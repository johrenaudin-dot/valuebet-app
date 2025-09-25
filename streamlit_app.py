# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ValueBet - Picks du jour (Consensus Marché)",
                   page_icon=":soccer:", layout="wide")
TZ = ZoneInfo("Europe/Paris")
TIMEOUT = 12

# The Odds API sport keys (corrigées)
SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "UCL": "soccer_uefa_champs_league",
    "UEL": "soccer_uefa_europa_league",
}

# -----------------------------
# FONCTIONS UTILITAIRES
# -----------------------------
def expected_value(p, o):
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p * o - 1.0)

def kelly_fraction(p, o, cap=0.05):
    if not (np.isfinite(p) and np.isfinite(o) and 0 <= p <= 1 and o > 1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def fair_probs_from_prices(oh, od, oa):
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
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": start_iso,
        "commenceTimeTo": end_iso,
    }
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, f"{r.status_code} - {r.text}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def best_prices_and_fair_by_book(event):
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
                if best["H"] is None or oh > best["H"]:
                    best["H"] = oh
                if best["D"] is None or od > best["D"]:
                    best["D"] = od
                if best["A"] is None or oa > best["A"]:
                    best["A"] = oa
                fair_list.append(fair_probs_from_prices(oh, od, oa))
    return best, fair_list

def edge_color(edge: float) -> str:
    if edge >= 0.10: return "#1b5e20"
    if edge >= 0.07: return "#2e7d32"
    if edge >= 0.05: return "#43a047"
    if edge >= 0.03: return "#ffb300"
    return "#9e9e9e"

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Paramètres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=1000.0, step=100.0, format="%.0f")
min_edge = st.sidebar.slider("Edge minimum (p×odds - 1)", 0.0, 0.20, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.20, 0.05, 0.01)

# freins conservateurs
min_prob = st.sidebar.slider("Proba minimum du pick (modèle)", 0.00, 0.30, 0.10, 0.01)
min_books = st.sidebar.slider("Nb minimum de bookmakers", 1, 10, 4)
blend_w = st.sidebar.slider("Poids consensus vs marché (shrinkage)", 0.0, 1.0, 0.60, 0.05)
max_std = st.sidebar.slider("Dispersion max des probas (écart-type)", 0.00, 0.20, 0.06, 0.01)

# limite de cote optionnelle
limit_odds = st.sidebar.checkbox("Limiter la cote max ?", value=False)
max_odds = st.sidebar.number_input("Cote max (si coché)", min_value=1.01, value=12.0, step=0.5, disabled=not limit_odds)

debug_mode = st.sidebar.checkbox("Mode debug (diagnostics)", value=True)

base_day = st.sidebar.date_input("Date de référence", datetime.now(TZ).date())
days_ahead = st.sidebar.slider("Jours à venir (0-7)", 0, 7, 2)
start_day, end_day = base_day, base_day + timedelta(days=days_ahead)
st.sidebar.write(f"Fenêtre: {start_day.isoformat()} → {end_day.isoformat()}")

if st.sidebar.button("Recharger (vider le cache)"):
    st.cache_data.clear()

leagues_selected = st.sidebar.multiselect(
    "Ligues analysées", list(SPORT_KEYS.keys()),
    default=["Premier League","La Liga","Serie A","Bundesliga","Ligue 1","UCL","UEL"]
)

# -----------------------------
# MAIN
# -----------------------------
st.title("ValueBet - Picks du jour (Consensus Marché)")
st.caption(
    "Proba consensus = médiane des probas 'fair' (1/cote, normalisées). "
    "Proba **modèle** = mélange Consensus↔Marché (shrinkage) pour réduire les extrêmes. "
    "Edge = proba_modèle × meilleure cote − 1, mise via Kelly capé."
)

api_key = st.secrets.get("ODDS_API_KEY")
if not api_key:
    st.error("❌ Aucune clé The Odds API trouvée (Settings → Secrets → `ODDS_API_KEY`).")
    st.stop()

start_iso = f"{start_day.strftime('%Y-%m-%d')}T00:00:00Z"
end_iso   = f"{end_day.strftime('%Y-%m-%d')}T23:59:59Z"

rows, diag_rows, debug_rows = [], [], []

for lname in leagues_selected:
    skey = SPORT_KEYS[lname]
    data, err = fetch_odds(skey, start_iso, end_iso, api_key)
    if data is None:
        diag_rows.append({"Ligue": lname, "Matchs fenêtre": 0, "Avec cotes": 0, "Erreur API": err})
        continue

    match_count = len(data)
    with_prices = 0

    for ev in data:
        best, fair_list = best_prices_and_fair_by_book(ev)
        book_count = len([p for p in fair_list if p])
        if book_count < min_books:
            debug_rows.append({"Date (UTC)": ev.get("commence_time",""), "Ligue": lname,
                               "Home": ev.get("home_team"), "Away": ev.get("away_team"),
                               "Motif": f"Rejet: bookmakers<{min_books}"})
            continue

        consensus = aggregate_consensus(fair_list)
        if consensus is None or None in (best["H"], best["D"], best["A"]):
            debug_rows.append({"Date (UTC)": ev.get("commence_time",""), "Ligue": lname,
                               "Home": ev.get("home_team"), "Away": ev.get("away_team"),
                               "Motif": "Cotes insuffisantes"})
            continue

        # Dispersion (écart-type des probas par book) pour filtrer les cas instables
        arrH = [p["H"] for p in fair_list if p]; arrD = [p["D"] for p in fair_list if p]; arrA = [p["A"] for p in fair_list if p]
        std_max = float(max(np.std(arrH), np.std(arrD), np.std(arrA)))
        if std_max > max_std:
            debug_rows.append({"Date (UTC)": ev.get("commence_time",""), "Ligue": lname,
                               "Home": ev.get("home_team"), "Away": ev.get("away_team"),
                               "Motif": f"Rejet: dispersion>{max_std} (std={std_max:.3f})"})
            continue

        # Probas implicites du marché depuis les meilleures cotes
        inv = np.array([1/float(best["H"]), 1/float(best["D"]), 1/float(best["A"])], float)
        inv = inv / inv.sum()
        market_probs = {"H": float(inv[0]), "D": float(inv[1]), "A": float(inv[2])}

        # Shrinkage : proba modèle = blend_w * consensus + (1-blend_w) * marché
        probs_final = {k: blend_w*consensus[k] + (1.0-blend_w)*market_probs[k] for k in ("H","D","A")}

        with_prices += 1
        # Calcule edge pour chaque issue à partir des probas modèle
        eH = expected_value(probs_final["H"], best["H"])
        eD = expected_value(probs_final["D"], best["D"])
        eA = expected_value(probs_final["A"], best["A"])
        pick_map = {"H": (probs_final["H"], best["H"], eH),
                    "D": (probs_final["D"], best["D"], eD),
                    "A": (probs_final["A"], best["A"], eA)}
        best_label, (p_star, o_star, edge_star) = max(
            pick_map.items(), key=lambda kv: (kv[1][2] if np.isfinite(kv[1][2]) else -9e9)
        )

        # Garde liberté sur les grosses cotes : on ne coupe que si l’utilisateur coche la limite
        if limit_odds and o_star > max_odds:
            debug_rows.append({"Date (UTC)": ev.get("commence_time",""), "Ligue": lname,
                               "Home": ev.get("home_team"), "Away": ev.get("away_team"),
                               "Motif": f"Rejet: cote>{max_odds} (o={o_star:.2f})"})
            continue

        # proba min du pick (modèle)
        if p_star < min_prob:
            debug_rows.append({"Date (UTC)": ev.get("commence_time",""), "Ligue": lname,
                               "Home": ev.get("home_team"), "Away": ev.get("away_team"),
                               "Motif": f"Rejet: proba<{min_prob} (p={p_star:.3f})"})
            continue

        # Heure locale pour affichage
        try:
            date_local = pd.to_datetime(ev.get("commence_time")).tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_local = (ev.get("commence_time") or "")[:16]

        kelly = kelly_fraction(p_star, o_star, cap=kelly_cap)
        stake = bankroll * kelly

        rows.append({
            "DateLocal": date_local, "Ligue": lname,
            "Home": ev.get("home_team"), "Away": ev.get("away_team"),
            "Pick": best_label,
            "Cote": round(o_star, 2),
            # proba du pick selon modèle et selon consensus (demandé)
            "ProbaModele": round(p_star, 3),
            "ProbaConsensusPick": round(consensus[best_label], 3),
            # les 3 probas modèle (H/D/A) pour contexte
            "P_H": round(probs_final["H"], 3),
            "P_D": round(probs_final["D"], 3),
            "P_A": round(probs_final["A"], 3),
            "Edge": round(edge_star, 3),
            "MiseEUR": round(stake, 2),
            "KellyPct": round(100*kelly, 2),
        })

    diag_rows.append({"Ligue": lname, "Matchs fenêtre": match_count, "Avec cotes": with_prices, "Erreur API": ""})

# -----------------------------
# AFFICHAGE
# -----------------------------
st.subheader("🎯 Picks à jouer (clairs et colorés)")
if not rows:
    st.info("Aucun value bet trouvé pour la fenêtre et les paramètres actuels.")
else:
    dfp = pd.DataFrame(rows).sort_values(["Edge","MiseEUR"], ascending=[False, False]).reset_index(drop=True)

    # CARTES (TOP 10)
    for _, r in dfp.head(10).iterrows():
        col = edge_color(r["Edge"])
        pick_txt = {"H": "Victoire HOME", "D": "Match NUL", "A": "Victoire AWAY"}[r["Pick"]]
        # chips pour H/D/A (probas modèle)
        chips_hda = (
            f"<span style='background:#263238;padding:4px 8px;border-radius:8px;'>H {int(round(r['P_H']*100))}%</span>"
            f"<span style='background:#263238;padding:4px 8px;border-radius:8px;'>D {int(round(r['P_D']*100))}%</span>"
            f"<span style='background:#263238;padding:4px 8px;border-radius:8px;'>A {int(round(r['P_A']*100))}%</span>"
        )
        st.markdown(
            f"""
<div style="border-radius:12px;padding:14px 16px;margin:10px 0;
            background:#111;border-left:10px solid {col};color:#fff;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:18px;font-weight:600;">
      {r['DateLocal']} · <span style="opacity:.85">{r['Ligue']}</span>
    </div>
    <div style="font-size:14px;opacity:.8;">Edge <b>{int(round(r['Edge']*100))}%</b></div>
  </div>
  <div style="margin-top:6px;font-size:17px;">
    <b>{r['Home']} vs {r['Away']}</b>
  </div>
  <div style="margin-top:6px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
    <span style="background:{col};color:#fff;padding:4px 10px;border-radius:8px;">
      Pick : <b>{pick_txt}</b>
    </span>
    <span style="background:#263238;padding:4px 10px;border-radius:8px;">
      Cote : <b>{r['Cote']}</b>
    </span>
    <span style="background:#263238;padding:4px 10px;border-radius:8px;">
      Proba modèle : <b>{int(round(r['ProbaModele']*100))}%</b>
    </span>
    <span style="background:#263238;padding:4px 10px;border-radius:8px;">
      Proba consensus : <b>{int(round(r['ProbaConsensusPick']*100))}%</b>
    </span>
    <span style="background:#263238;padding:4px 10px;border-radius:8px;">
      Mise : <b>{r['MiseEUR']}€</b> (Kelly {r['KellyPct']}%)
    </span>
  </div>
  <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">
    {chips_hda}
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    st.subheader("🧾 Tableau récapitulatif")
    cols = ["DateLocal","Ligue","Home","Away","Pick","Cote",
            "ProbaModele","ProbaConsensusPick","P_H","P_D","P_A",
            "Edge","MiseEUR","KellyPct"]
    dfp = dfp.reindex(columns=cols)
    st.metric("Value bets trouvés", len(dfp))
    st.dataframe(dfp, use_container_width=True, height=min(600, 100 + 35*len(dfp)))

# DIAGNOSTIC & DEBUG
st.subheader("Diagnostic par ligue")
st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

if debug_mode:
    st.subheader("Debug — motifs de rejet / détails")
    if debug_rows:
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, height=500)
    else:
        st.write("Aucun match rejeté (ou pas de cotes disponibles).")
