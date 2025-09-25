# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

# -----------------------------
# CONFIG & CONSTANTES
# -----------------------------
st.set_page_config(page_title="ValueBet - Picks du jour (Consensus Marché)", page_icon=":soccer:", layout="wide")
TZ = ZoneInfo("Europe/Paris")
TIMEOUT = 12

# Clés "sport" officielles The Odds API
SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_1",           # <-- correct
    "UCL": "soccer_uefa_champions_league",
    "UEL": "soccer_uefa_europa_league",
}

# -----------------------------
# FONCTIONS
# -----------------------------
def kelly_fraction(p, o, cap=0.05):
    """Kelly (capé) à partir d'une proba p et d'une cote o."""
    if not (np.isfinite(p) and np.isfinite(o) and 0 <= p <= 1 and o > 1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def expected_value(p, o):
    """Edge = p*odds - 1."""
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p*o - 1.0)

def fair_probs_from_prices(oh, od, oa):
    """Probas 'fair' normalisées (en retirant l'overround) à partir des 3 cotes H/D/A."""
    try:
        inv = np.array([1/oh, 1/od, 1/oa], dtype=float)
        s = inv.sum()
        if s <= 0:
            return None
        fair = inv / s
        return {"H": float(fair[0]), "D": float(fair[1]), "A": float(fair[2])}
    except Exception:
        return None

def aggregate_consensus(fair_list):
    """Agrège les probas 'fair' de plusieurs books via la médiane par issue."""
    H, D, A = [], [], []
    for p in fair_list:
        if not p:
            continue
        H.append(p["H"]); D.append(p["D"]); A.append(p["A"])
    if not H or not D or not A:
        return None
    h, d, a = float(np.median(H)), float(np.median(D)), float(np.median(A))
    s = h + d + a
    if s <= 0:
        return None
    return {"H": h/s, "D": d/s, "A": a/s}

def fetch_odds(sport_key: str, start_iso: str, end_iso: str, api_key: str):
    """Appelle The Odds API (h2h) pour un sport donné entre start_iso et end_iso (UTC 'Z')."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": start_iso,  # YYYY-MM-DDTHH:MM:SSZ
        "commenceTimeTo": end_iso,      # YYYY-MM-DDTHH:MM:SSZ
    }
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, f"{r.status_code} - {r.text}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def best_prices_and_fair_by_book(event):
    """
    Pour un event, extrait par bookmaker :
      - (oh, od, oa) = cotes H/D/A (via noms home/away/draw),
      - 'fair' = probas normalisées 1/cote.
    Retourne:
      - best: dict best odds {'H','D','A'}
      - fair_list: liste de dicts 'fair' (un par bookmaker)
    """
    home = event.get("home_team")
    away = event.get("away_team")
    best = {"H": None, "D": None, "A": None}
    fair_list = []

    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            oh = od = oa = None
            for o in m.get("outcomes", []):
                name = (o.get("name") or "").strip()
                price = o.get("price")
                if price is None:
                    continue
                if name == home:
                    oh = float(price)
                elif name == away:
                    oa = float(price)
                elif name.lower() == "draw":
                    od = float(price)
            if oh and od and oa:
                # best odds
                if best["H"] is None or oh > best["H"]:
                    best["H"] = oh
                if best["D"] is None or od > best["D"]:
                    best["D"] = od
                if best["A"] is None or oa > best["A"]:
                    best["A"] = oa
                fair_list.append(fair_probs_from_prices(oh, od, oa))

    return best, fair_list

# -----------------------------
# UI (SIDEBAR)
# -----------------------------
st.sidebar.header("Paramètres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=1000.0, step=100.0, format="%.0f")
min_edge = st.sidebar.slider("Edge minimum (p*odds - 1)", 0.0, 0.20, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.20, 0.05, 0.01)
debug_mode = st.sidebar.checkbox("Mode debug (afficher diagnostics)", value=True)

base_day = st.sidebar.date_input("Date de référence", datetime.now(TZ).date())
days_ahead = st.sidebar.slider("Jours à venir (0-7)", 0, 7, 2)
start_day = base_day
end_day = base_day + timedelta(days=days_ahead)
st.sidebar.write(f"Fenêtre: {start_day.isoformat()} → {end_day.isoformat()}")

if st.sidebar.button("Recharger (vider le cache)"):
    st.cache_data.clear()

leagues_selected = st.sidebar.multiselect(
    "Ligues",
    list(SPORT_KEYS.keys()),
    default=["Premier League","La Liga","Serie A","Bundesliga","Ligue 1","UCL","UEL"]
)

# -----------------------------
# MAIN
# -----------------------------
st.title("ValueBet - Picks du jour (Consensus Marché)")
st.caption("Méthode : pour chaque bookmaker → probas 'fair' (1/cote, normalisées). "
           "Consensus = médiane des probas par issue (H/D/A). On compare ce consensus à la meilleure cote du marché. "
           "Edge = p*odds - 1, mise = Kelly capé.")

api_key = st.secrets.get("ODDS_API_KEY")
if not api_key:
    st.error("❌ Aucune clé The Odds API trouvée (Settings → Secrets → ODDS_API_KEY).")
    st.stop()

# The Odds API requiert des timestamps UTC 'Z'
start_iso = f"{start_day.strftime('%Y-%m-%d')}T00:00:00Z"
end_iso   = f"{end_day.strftime('%Y-%m-%d')}T23:59:59Z"

rows = []        # picks retenus (value bets)
diag_rows = []   # diagnostic par ligue
debug_rows = []  # détails par match (si debug)

for lname in leagues_selected:
    sport_key = SPORT_KEYS[lname]
    data, err = fetch_odds(sport_key, start_iso, end_iso, api_key)
    if data is None:
        diag_rows.append({"Ligue": lname, "Matchs fenetre": 0, "Avec cotes": 0, "Erreur API": err})
        continue

    match_count = len(data)
    with_prices = 0

    for ev in data:
        best, fair_list = best_prices_and_fair_by_book(ev)
        consensus = aggregate_consensus(fair_list)

        # Debug (même si pas de pick)
        dbg = {
            "Date": ev.get("commence_time",""),
            "Ligue": lname,
            "Home": ev.get("home_team"),
            "Away": ev.get("away_team"),
            "Best_H": best["H"],
            "Best_D": best["D"],
            "Best_A": best["A"],
        }

        if consensus is None or None in (best["H"], best["D"], best["A"]):
            dbg.update({"pH": np.nan, "pD": np.nan, "pA": np.nan,
                        "Edge_H": np.nan, "Edge_D": np.nan, "Edge_A": np.nan,
                        "BestPick": None, "BestEdge": np.nan, "Motif": "Cotes insuffisantes"})
            debug_rows.append(dbg)
            continue

        with_prices += 1
        pH, pD, pA = consensus["H"], consensus["D"], consensus["A"]
        eH = expected_value(pH, best["H"])
        eD = expected_value(pD, best["D"])
        eA = expected_value(pA, best["A"])
        pick_map = {"H": (pH, best["H"], eH), "D": (pD, best["D"], eD), "A": (pA, best["A"], eA)}
        best_label, (p_star, o_star, edge_star) = max(
            pick_map.items(), key=lambda kv: (kv[1][2] if np.isfinite(kv[1][2]) else -9e9)
        )

        dbg.update({"pH": round(pH,3), "pD": round(pD,3), "pA": round(pA,3),
                    "Edge_H": round(eH,3), "Edge_D": round(eD,3), "Edge_A": round(eA,3),
                    "BestPick": best_label, "BestEdge": round(edge_star,3),
                    "Motif": "" if edge_star >= min_edge else "Edge < seuil"})
        debug_rows.append(dbg)

        if edge_star < min_edge:
            continue

        kelly = kelly_fraction(p_star, o_star, cap=kelly_cap)
        stake = bankroll * kelly
        rows.append({
            "Date": ev.get("commence_time",""),
            "Ligue": lname,
            "Home": ev.get("home_team"),
            "Away": ev.get("away_team"),
            "Pick": best_label,
            "Cote": round(o_star, 2),
            "ProbaConsensus": round(p_star, 3),
            "Edge": round(edge_star, 3),
            "MiseEUR": round(stake, 2),
            "KellyPct": round(100*kelly, 2),
        })

    diag_rows.append({"Ligue": lname, "Matchs fenetre": match_count, "Avec cotes": with_prices, "Erreur API": ""})

# -----------------------------
# RENDU
# -----------------------------
if not rows:
    st.info("Aucun value bet trouvé pour la fenêtre et les paramètres actuels.")
else:
    dfp = pd.DataFrame(rows).sort_values(["Edge","MiseEUR"], ascending=[False,False]).reset_index(drop=True)
    st.metric("Value bets trouvés", len(dfp))
    st.dataframe(dfp, use_container_width=True, height=min(600, 100+35*len(dfp)))

st.subheader("Diagnostic par ligue")
st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

if debug_mode:
    st.subheader("Debug - Tous les matchs scannés")
    if debug_rows:
        cols = ["Date","Ligue","Home","Away","Best_H","Best_D","Best_A",
                "pH","pD","pA","Edge_H","Edge_D","Edge_A","BestPick","BestEdge","Motif"]
        dfd = pd.DataFrame(debug_rows)
        dfd = dfd.reindex(columns=[c for c in cols if c in dfd.columns])
        st.dataframe(dfd, use_container_width=True, height=min(700, 120+30*len(dfd)))
    else:
        st.write("Aucun match (ou pas de cotes disponibles).")
