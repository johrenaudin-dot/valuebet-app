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
# FONCTIONS
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
    if edge >= 0.10: return "#1b5e20"   # très fort
    if edge >= 0.07: return "#2e7d32"
    if edge >= 0.05: return "#43a047"
    if edge >= 0.03: return "#ffb300"   # borderline
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

# garde-fous "value" explicites
enforce_model_ge_consensus = st.sidebar.checkbox("Exiger p_mod ≥ p_cons (issue retenue)", value=True)
enforce_consensus_edge = st.sidebar.checkbox("Exiger edge (consensus) ≥ edge min", value=True)

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
    "Proba **modèle** = mélange consensus↔marché (shrinkage) pour réduire les extrêmes. "
    "Edge modèle = p_mod × meilleure cote − 1.  "
    "Les garde-fous garantissent un value vs consensus (
