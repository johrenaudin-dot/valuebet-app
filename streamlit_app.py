# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="ValueBet - Odds Consensus", page_icon=":soccer:", layout="wide")

# ==================== Config ====================
TZ = ZoneInfo("Europe/Paris")
TIMEOUT = 8  # secondes

# Clés sport The Odds API
SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_1",
    "UCL": "soccer_uefa_champs_league",
    "UEL": "soccer_uefa_europa_league",
}

# ==================== Utilitaires ====================
def kelly_fraction(p, o, cap=0.05):
    """Kelly fraction (capée). p en [0,1], o > 1."""
    if not (np.isfinite(p) and np.isfinite(o) and 0 <= p <= 1 and o > 1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def expected_value(p, o):
    """Edge = EV = p*o - 1."""
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p*o - 1.0)

def fair_probs_from_prices(prices):
    """
    prices: dict {'H': price, 'D': price, 'A': price} pour UN bookmaker
    Retourne des probabilités 'fair' normalisées (sans overround) ou None si données incomplètes.
    """
    try:
        oh, od, oa = float(prices["H"]), float(prices["D"]), float(prices["A"])
        if min(oh, od, oa) <= 1:
            return None
        imp = np.array([1/oh, 1/od, 1/oa], dtype=float)
        s = imp.sum()
        if s <= 0:
            return None
        fair = imp / s
        return {"H": float(fair[0]), "D": float(fair[1]), "A": float(fair[2])}
    except Exception:
        return None

def aggregate_consensus(book_fair_probs_list):
    """
    Agrège les probas 'fair' de plusieurs bookmakers.
    Stratégie: médiane par issue (H/D/A) pour robustesse.
    """
    if not book_fair_probs_list:
        return None
    H, D, A = [], [], []
    for p in book_fair_probs_list:
        if p is None:
            continue
        H.append(p["H"]); D.append(p["D"]); A.append(p["A"])
    if len(H) == 0 or len(D) == 0 or len(A) == 0:
        return None
    # médianes
    h, d, a = float(np.median(H)), float(np.median(D)), float(np.median(A))
    s = h + d + a
    if s <= 0:
        return None
    # re-normalise pour être sûr que h+d+a=1
    return {"H": h/s, "D": d/s, "A": a/s}

def best_prices_across_books(bookmakers):
    """
    Prend la meilleure cote H/D/A parmi tous les bookmakers renvoyés par l'API.
    Retourne {'H': best_oh, 'D': best_od, 'A': best_oa} + le détail par book (pour debug).
    """
    best = {"H": None, "D": None, "A": None}
    detail = []  # pour debug
    for bk in bookmakers:
        bk_name = bk.get("title") or bk.get("key")
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            oh = od = oa = None
            for o in m.get("outcomes", []):
                name = (o.get("name") or "").lower()
                price = o.get("price")
                if price is None:
                    continue
                if name in ("home", "draw", "away"):
                    key = {"home":"H", "draw":"D", "away":"A"}[name]
                    if best[key] is None or price > best[key]:
                        best[key] = float(price)
                    if key == "H": oh = float(price)
                    if key == "D": od = float(price)
                    if key == "A": oa = float(price)
                else:
                    # parfois le nom est l'équipe (au lieu de home/away)
                    # on ne sait pas distinguer Home/Away sans plus d'infos => on saute le 'detail' mais best sera traité plus haut
                    pass
            if oh and od and oa:
                detail.append({"book": bk_name, "H": oh, "D": od, "A": oa})
    return best, detail

def fetch_odds(sport_key, start_iso, end_iso, api_key):
    """
    Appelle The Odds API pour un sport donné, retourne la liste d'événements avec bookmakers complets.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": start_iso,
        "commenceTimeTo": end_iso,
        "bookmakers": "",   # vide => tous les books disponibles
    }
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, f"{r.status_code} - {r.text}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ==================== UI ====================
st.sidebar.header("Parametres")

bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=1000.0, step=100.0, format="%.0f")
min_edge = st.sidebar.slider("Edge minimum", 0.0, 0.20, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.20, 0.05, 0.01)
debug_mode = st.sidebar.checkbox("Mode debug (afficher tout)", value=True)

base_day = st.sidebar.date_input("Date de référence", datetime.now(TZ).date())
days_ahead = st.sidebar.slider("Jours à venir (0-7)", 0, 7, 2)
start_day = base_day
end_day = base_day + timedelta(days=days_ahead)
st.sidebar.write(f"Fenêtre: {start_day.isoformat()} → {end_day.isoformat()}")

if st.sidebar.button("Recharger les données (vider le cache)"):
    st.cache_data.clear()

leagues_selected = st.sidebar.multiselect(
    "Ligues",
    list(SPORT_KEYS.keys()),
    default=["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "UCL", "UEL"]
)

# ==================== Appels API (cache) ====================
@st.cache_data(show_spinner=False)
def load_odds_for_league(sport_key: str, start_iso: str, end_iso: str, api_key: str):
    return fetch_odds(sport_key, start_iso, end_iso, api_key)

# ==================== Corps ====================
st.title("ValueBet - Picks du jour (Consensus Marché)")

api_key = st.secrets.get("ODDS_API_KEY")
if not api_key:
    st.error("Aucune clé The Odds API trouvée dans les Secrets. Ajoute ODDS_API_KEY dans Settings → Secrets.")
    st.stop()

start_iso = datetime.combine(start_day, datetime.min.time(), tzinfo=TZ).isoformat()
end_iso   = datetime.combine(end_day,   datetime.max.time(), tzinfo=TZ).isoformat()

rows = []        # picks (value bets)
diag_rows = []   # diag par ligue
debug_rows = []  # debug détaillé

status = st.status("Chargement des ligues...", expanded=False)

for i, lname in enumerate(leagues_selected, 1):
    status.update(label=f"Chargement: {lname} ({i}/{len(leagues_selected)})", state="running")
    sport_key = SPORT_KEYS[lname]

    data, err = load_odds_for_league(sport_key, start_iso, end_iso, api_key)
    if err is not None or data is None:
        diag_rows.append({"Ligue": lname, "Matchs fenetre": 0, "Avec cotes": 0, "Erreur API": str(err)})
        continue

    match_count = 0
    with_prices = 0

    for ev in data:
        match_count += 1
        home = ev.get("home_team")
        away = ev.get("away_team")
        tiso = ev.get("commence_time")
        bks = ev.get("bookmakers", [])

        # Meilleures cotes
        best_prices, books_detail = best_prices_across_books(bks)
        # Probas "fair" par bookmaker
        fair_list = []
        for bd in books_detail:
            fair_list.append(fair_probs_from_prices({"H": bd["H"], "D": bd["D"], "A": bd["A"]}))
        consensus = aggregate_consensus(fair_list)

        # Debug toujours
        dbg = {"Date": tiso, "Ligue": lname, "Home": home, "Away": away,
               "Best_H": best_prices["H"], "Best_D": best_prices["D"], "Best_A": best_prices["A"]}
        if consensus is None or None in (best_prices["H"], best_prices["D"], best_prices["A"]):
            dbg.update({"pH": np.nan, "pD": np.nan, "pA": np.nan,
                        "Edge_H": np.nan, "Edge_D": np.nan, "Edge_A": np.nan,
                        "BestPick": None, "BestEdge": np.nan, "Motif": "Cotes insuffisantes"})
            debug_rows.append(dbg)
            continue

        with_prices += 1

        pH, pD, pA = consensus["H"], consensus["D"], consensus["A"]
        eH = expected_value(pH, best_prices["H"])
        eD = expected_value(pD, best_prices["D"])
        eA = expected_value(pA, best_prices["A"])
        pick_map = {"H": (pH, best_prices["H"], eH),
                    "D": (pD, best_prices["D"], eD),
                    "A": (pA, best_prices["A"], eA)}
        best_label, (p_star, o_star, edge_star) = max(
            pick_map.items(), key=lambda kv: (kv[1][2] if kv[1][2] is not None else -9e9)
        )

        dbg.update({"pH": round(pH,3), "pD": round(pD,3), "pA": round(pA,3),
                    "Edge_H": round(eH,3), "Edge_D": round(eD,3), "Edge_A": round(eA,3),
                    "BestPick": best_label, "BestEdge": round(edge_star,3), "Motif": "" if edge_star >= min_edge else "Edge < seuil"})
        debug_rows.append(dbg)

        if edge_star < min_edge:
            continue
        kelly = kelly_fraction(p_star, o_star, cap=kelly_cap)
        stake = bankroll * kelly
        rows.append({
            "Date": tiso,
            "Ligue": lname,
            "Home": home,
            "Away": away,
            "Pick": best_label,
            "Cote": round(o_star, 2),
            "ProbaConsensus": round(p_star, 3),
            "Edge": round(edge_star, 3),
            "MiseEUR": round(stake, 2),
            "KellyPct": round(100*kelly, 2),
        })

    diag_rows.append({"Ligue": lname, "Matchs fenetre": match_count, "Avec cotes": with_prices, "Erreur API": ""})

status.update(label="Terminé", state="complete")

# ==================== Rendu ====================
st.caption("Méthode : probas 'fair' par bookmaker (1/cote, normalisées), agrégées par médiane → consensus. "
           "On compare le consensus à la meilleure cote du marché (edge = p*odds - 1).")

if not rows:
    st.info("Aucun value bet trouvé pour la fenêtre et les paramètres actuels.")
else:
    dfp = pd.DataFrame(rows).sort_values(["Edge","MiseEUR"], ascending=[False,False]).reset_index(drop=True)
    st.metric("Value bets trouvés", len(dfp))
    st.dataframe(dfp, use_container_width=True, height=min(600, 100+35*len(dfp)))

st.subheader("Diagnostic par ligue")
if diag_rows:
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

if debug_mode:
    st.subheader("Debug - Tous les matchs dans la fenêtre")
    if debug_rows:
        cols = ["Date","Ligue","Home","Away","Best_H","Best_D","Best_A",
                "pH","pD","pA","Edge_H","Edge_D","Edge_A","BestPick","BestEdge","Motif"]
        dfd = pd.DataFrame(debug_rows)
        dfd = dfd.reindex(columns=[c for c in cols if c in dfd.columns])
        st.dataframe(dfd, use_container_width=True, height=min(700, 120+30*len(dfd)))
    else:
        st.write("Aucun match recensé (ou pas de cotes disponibles).")
