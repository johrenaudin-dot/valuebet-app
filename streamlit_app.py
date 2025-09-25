# streamlit_app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="ValueBet - API Odds", page_icon="⚽", layout="wide")

# ==============================
# Paramètres utilisateur
# ==============================
st.sidebar.header("Paramètres")

bankroll = st.sidebar.number_input("Bankroll (€)", 0, 1_000_000, 1000)
edge_min = st.sidebar.slider("Edge minimum", 0.0, 0.20, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.20, 0.05, 0.01)
min_prob = st.sidebar.slider("Proba minimum du pick (modèle)", 0.0, 0.30, 0.08, 0.01)
min_books = st.sidebar.slider("Nb minimum de bookmakers", 1, 10, 3)

days_ahead = st.sidebar.slider("Fenêtre (jours à venir)", 0, 7, 3)

# ==============================
# Fonctions utilitaires
# ==============================
def expected_value(p, o): 
    return p*o - 1

def kelly_fraction(p, o, cap=0.05):
    b = o - 1
    if b <= 0: return 0
    f = (p*b - (1-p)) / b
    return max(0, min(cap, f))

def edge_color(edge):
    if edge >= 0.10: return "#1b5e20"
    if edge >= 0.07: return "#2e7d32"
    if edge >= 0.05: return "#ff9800"
    return "#9e9e9e"

# ==============================
# API Odds
# ==============================
API_KEY = st.secrets["ODDS_API_KEY"]
BASE_URL = "https://api.the-odds-api.com/v4/sports"

sports = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "UCL": "soccer_uefa_champs_league",
    "UEL": "soccer_uefa_europa_league",
}

# ==============================
# Récupération des cotes
# ==============================
st.title("🎯 ValueBet - Picks du jour (Consensus Marché)")

date_from = datetime.now(timezone.utc)
date_to = date_from + timedelta(days=days_ahead)

all_matches = []

for lig, sport_key in sports.items():
    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.warning(f"{lig}: Erreur API {r.status_code} - {r.text}")
        continue
    data = r.json()

    for match in data:
        date = match["commence_time"]
        home, away = match["home_team"], match["away_team"]
        odds_all = []
        for book in match.get("bookmakers", []):
            for mk in book["markets"]:
                if mk["key"] == "h2h":
                    prices = mk["outcomes"]
                    if len(prices) == 2:  # pas de draw
                        continue
                    odds = {o["name"]: o["price"] for o in prices}
                    odds_all.append(odds)
        if len(odds_all) < min_books:
            continue

        # meilleures cotes
        bestH = max([o.get(home, np.nan) for o in odds_all], default=np.nan)
        bestD = max([o.get("Draw", np.nan) for o in odds_all], default=np.nan)
        bestA = max([o.get(away, np.nan) for o in odds_all], default=np.nan)
        if np.isnan([bestH, bestD, bestA]).any(): 
            continue

        # probas "fair"
        probs = []
        for o in odds_all:
            if home in o and "Draw" in o and away in o:
                inv = [1/o[home], 1/o["Draw"], 1/o[away]]
                s = sum(inv)
                probs.append([x/s for x in inv])
        if not probs: continue
        probs = np.array(probs)

        p_cons = np.median(probs, axis=0)  # consensus
        p_mod = p_cons  # simplifié: modèle = consensus

        picks = {"H": (home, bestH, p_cons[0], p_mod[0]),
                 "D": ("Draw", bestD, p_cons[1], p_mod[1]),
                 "A": (away, bestA, p_cons[2], p_mod[2])}

        for res, (team, odds, p_c, p_m) in picks.items():
            if odds <= 1: continue
            edge_c = expected_value(p_c, odds)
            edge_m = expected_value(p_m, odds)
            if edge_c > 0 and edge_m >= edge_min and p_m >= p_c and p_m >= min_prob:
                kelly = kelly_fraction(p_m, odds, cap=kelly_cap)
                stake = kelly*bankroll
                all_matches.append({
                    "Date": date,
                    "Ligue": lig,
                    "Match": f"{home} vs {away}",
                    "Pick": team,
                    "Odds": odds,
                    "p_cons": round(p_c,3),
                    "p_mod": round(p_m,3),
                    "Edge": round(edge_m,3),
                    "Kelly%": round(kelly*100,2),
                    "Stake€": round(stake,2)
                })

# ==============================
# Résultats
# ==============================
if not all_matches:
    st.info("Aucun value bet trouvé avec ces paramètres.")
else:
    df = pd.DataFrame(all_matches)
    st.metric("🎲 Nombre de picks trouvés", len(df))
    for _, r in df.iterrows():
        st.markdown(
            f"""
            <div style="background:{edge_color(r['Edge'])};padding:10px;border-radius:10px;margin:5px">
            <b>{r['Date']} · {r['Ligue']}</b><br>
            {r['Match']}<br>
            ✅ Pick: <b>{r['Pick']}</b><br>
            Cote: {r['Odds']} | p_cons: {r['p_cons']} | p_mod: {r['p_mod']} | Edge: {r['Edge']}<br>
            Mise: {r['Stake€']}€ (Kelly {r['Kelly%']}%)
            </div>
            """, unsafe_allow_html=True
        )
