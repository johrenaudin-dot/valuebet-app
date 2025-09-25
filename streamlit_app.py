# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------
# CONFIG GÉNÉRALE
# -----------------------------
st.set_page_config(page_title="ValueBet - Picks du jour (Consensus Marché)",
                   page_icon=":soccer:",
                   layout="wide")

TZ = ZoneInfo("Europe/Paris")
TIMEOUT = 12

# Clés officielles The Odds API (corrigées)
SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",          # <- corrigé
    "UCL": "soccer_uefa_champs_league",            # <- corrigé
    "UEL": "soccer_uefa_europa_league",
}

# -----------------------------
# OUTILS
# -----------------------------
def expected_value(p, o):
    """Edge = p*odds - 1."""
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p * o - 1.0)

def kelly_fraction(p, o, cap=0.05):
    """Kelly capé."""
    if not (np.isfinite(p) and np.isfinite(o) and 0 <= p <= 1 and o > 1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def fair_probs_from_prices(oh, od, oa):
    """Probas 'fair' normalisées (retire l'overround)."""
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
    """Médiane par issue (H/D/A) puis renormalisation."""
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
    """
    Pour un event, extrait par bookmaker :
      - (oh, od, oa) = cotes H/D/A (via noms home/away/draw),
      - fair = probas normalisées 1/cote par book.
    Retour:
      - best: {'H','D','A'} meilleures cotes
      - fair_list: liste de dicts 'fair'
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
                if best["H"] is None or oh > best["H"]:
                    best["H"] = oh
                if best["D"] is None or od > best["D"]:
                    best["D"] = od
                if best["A"] is None or oa > best["A"]:
                    best["A"] = oa
                fair_list.append(fair_probs_from_prices(oh, od, oa))
    return best, fair_list

def edge_color(edge: float) -> str:
    """Couleurs lisibles selon l'edge (match cards)."""
    if edge >= 0.10:
        return "#1b5e20"   # vert très f
