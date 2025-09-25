# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ValueBet - Today's Picks", page_icon=":soccer:", layout="wide")
TZ = ZoneInfo("Europe/Paris")
MAX_GOALS = 10

SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "UCL": "soccer_uefa_champions_league",
    "UEL": "soccer_uefa_europa_league",
}

# -----------------------------
# FONCTIONS
# -----------------------------

def poisson_probs(lam_h, lam_a, max_goals=10):
    gh = np.arange(0, max_goals + 1)
    ga = np.arange(0, max_goals + 1)
    denom_h = np.array([math.factorial(int(i)) for i in gh], dtype=float)
    denom_a = np.array([math.factorial(int(i)) for i in ga], dtype=float)
    ph = np.exp(-lam_h) * np.power(lam_h, gh) / denom_h
    pa = np.exp(-lam_a) * np.power(lam_a, ga) / denom_a
    mat = np.outer(ph, pa)
    p_draw = float(np.trace(mat))
    p_home = float(np.tril(mat, -1).sum())
    p_away = float(np.triu(mat, 1).sum())
    return p_home, p_draw, p_away

def kelly_fraction(p, o, cap=0.05):
    if not (np.isfinite(p) and np.isfinite(o) and o > 1):
        return 0.0
    b = o - 1.0
    f = (p * b - (1 - p)) / b
    return float(max(0.0, min(cap, f)))

def expected_value(p, o):
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p * o - 1.0)

def fetch_odds(sport_key, date_from, date_to):
    """Appel à The Odds API"""
    url = "https://api.the-odds-api.com/v4/sports/{}/odds".format(sport_key)
    params = {
        "apiKey": st.secrets["ODDS_API_KEY"],
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "commenceTimeFrom": f"{date_from}T00:00:00Z",
        "commenceTimeTo": f"{date_to}T23:59:59Z",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None, f"{r.status_code} - {r.text}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Parametres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=1000.0, step=100.0, format="%.0f")
min_edge = st.sidebar.slider("Edge minimum", 0.0, 0.15, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.2, 0.05, 0.01)
debug_mode = st.sidebar.checkbox("Mode debug (afficher tout)", value=False)
ref_date = st.sidebar.text_input("Date de reference", datetime.now(TZ).date().isoformat())
days_ahead = st.sidebar.slider("Jours à venir (0-7)", 0, 7, 2)

date0 = datetime.fromisoformat(ref_date).date()
date1 = date0 + timedelta(days=days_ahead)
st.sidebar.write(f"Fenetre: {date0} → {date1}")

# -----------------------------
# MAIN
# -----------------------------
st.title("ValueBet - Picks du jour (Consensus Marche)")
st.caption("Méthode : probas 'fair' par bookmaker (1/cote, normalisées), agrégées par médiane → consensus. On compare le consensus à la meilleure cote du marché (edge = p*odds - 1).")

rows = []
diag = []

for lname, skey in SPORT_KEYS.items():
    data, err = fetch_odds(skey, date0, date1)
    if data is None:
        diag.append({"Ligue": lname, "Matchs fenetre": 0, "Avec cotes": 0, "Erreur API": err})
        continue

    match_count = len(data)
    bets_ok = 0

    for match in data:
        if "bookmakers" not in match or not match["bookmakers"]:
            continue

        outcomes = []
        for bm in match["bookmakers"]:
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
                    if len(prices) == 3:  # home/draw/away
                        total = sum(1/x for x in prices.values())
                        probs = {k: (1/v)/total for k, v in prices.items()}
                        outcomes.append({"probs": probs, "prices": prices})

        if not outcomes:
            continue

        probs_med = {}
        for side in ["Home", "Draw", "Away"]:
            vals = [o["probs"].get(side, np.nan) for o in outcomes]
            probs_med[side] = np.nanmedian(vals)

        best_odds = {}
        for side in ["Home", "Draw", "Away"]:
            vals = [o["prices"].get(side, np.nan) for o in outcomes]
            best_odds[side] = np.nanmax(vals)

        pick_map = {
            "H": (probs_med["Home"], best_odds["Home"]),
            "D": (probs_med["Draw"], best_odds["Draw"]),
            "A": (probs_med["Away"], best_odds["Away"]),
        }
        scored = {k: (p, o, expected_value(p, o)) for k, (p, o) in pick_map.items() if o is not None}
        best_label, (p_star, o_star, edge_star) = max(scored.items(), key=lambda kv: kv[1][2])

        if edge_star >= min_edge:
            kelly = kelly_fraction(p_star, o_star, cap=kelly_cap)
            stake = bankroll * kelly
            rows.append({
                "Date": match["commence_time"][:10],
                "Ligue": lname,
                "Home": match["home_team"],
                "Away": match["away_team"],
                "Pick": best_label,
                "Cote": round(o_star, 2),
                "ProbaConsensus": round(p_star, 3),
                "Edge": round(edge_star, 3),
                "MiseEUR": round(stake, 2),
                "KellyPct": round(100 * kelly, 2),
            })
            bets_ok += 1

    diag.append({"Ligue": lname, "Matchs fenetre": match_count, "Avec cotes": bets_ok, "Erreur API": err})

# -----------------------------
# DISPLAY
# -----------------------------
if not rows:
    st.info("Aucun value bet trouve pour la fenetre et les parametres actuels.")
else:
    dfp = pd.DataFrame(rows).sort_values(["Edge", "MiseEUR"], ascending=[False, False]).reset_index(drop=True)
    st.metric("Value bets trouves", len(dfp))
    st.dataframe(dfp, use_container_width=True, height=min(600, 100 + 35 * len(dfp)))

if debug_mode:
    st.subheader("Diagnostic par ligue")
    st.dataframe(pd.DataFrame(diag))
