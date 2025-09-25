# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="ValueBet - Today's Picks", page_icon=":soccer:", layout="wide")

# ---------- Config ----------
SEASON_START = 2025  # pour l'historique (forces d'équipes)
MAX_GOALS = 10
TZ = ZoneInfo("Europe/Paris")

# football-data (historique pour le modèle)
FD_CODES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "UCL": "EC",
    "UEL": "EL",
}

# The Odds API (fixtures + cotes à venir)
ODDS_SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_1",
    "UCL": "soccer_uefa_champs_league",
    "UEL": "soccer_uefa_europa_league",
}

# ---------- Utils ----------
def yy(y: int) -> str:
    return f"{y%100:02d}"

def fd_url(season_start: int, code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{yy(season_start)}{yy(season_start+1)}/{code}.csv"

def fetch_csv(url: str) -> pd.DataFrame | None:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        from io import StringIO
        return pd.read_csv(StringIO(r.text))
    except Exception:
        return None

def parse_date_series(s: pd.Series) -> pd.Series:
    def parse_one(x):
        for fmt in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d","%Y/%m/%d"):
            try:
                return datetime.strptime(str(x), fmt).date()
            except Exception:
                continue
        return pd.NaT
    return s.apply(parse_one)

def poisson_probs(lam_h, lam_a, max_goals=10):
    gh = np.arange(0, max_goals+1)
    ga = np.arange(0, max_goals+1)
    ph = np.exp(-lam_h) * np.power(lam_h, gh) / np.array([np.math.factorial(i) for i in gh])
    pa = np.exp(-lam_a) * np.power(lam_a, ga) / np.array([np.math.factorial(i) for i in ga])
    mat = np.outer(ph, pa)
    p_draw = np.trace(mat)
    p_home = np.tril(mat, -1).sum()
    p_away = np.triu(mat, 1).sum()
    return float(p_home), float(p_draw), float(p_away)

def expected_value(p, o):  # edge
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p*o - 1.0)

def kelly_fraction(p, o, cap=0.05):
    if not (np.isfinite(p) and np.isfinite(o) and o>1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def team_strengths_poisson(df: pd.DataFrame, today):
    df = df.copy()
    if "Date" not in df.columns:
        return None
    df["Date"] = parse_date_series(df["Date"])
    hist = df[df["Date"] < today]
    if hist.empty:
        return None
    needed = ["HomeTeam","AwayTeam","FTHG","FTAG"]
    if not all(c in hist.columns for c in needed):
        return None
    base_h = hist["FTHG"].mean()
    base_a = hist["FTAG"].mean()
    ha = hist.groupby("HomeTeam")["FTHG"].mean() / (base_h if base_h>0 else 1)
    hd = hist.groupby("HomeTeam")["FTAG"].mean() / (base_a if base_a>0 else 1)
    aa = hist.groupby("AwayTeam")["FTAG"].mean() / (base_a if base_a>0 else 1)
    ad = hist.groupby("AwayTeam")["FTHG"].mean() / (base_h if base_h>0 else 1)
    return dict(base_h=base_h, base_a=base_a, ha=ha, hd=hd, aa=aa, ad=ad, hist=hist)

def adjust_with_home_away(S, home, away):
    hist = S["hist"]
    lg_home_win = (hist["FTR"]=="H").mean()
    lg_away_win = (hist["FTR"]=="A").mean()
    team_home_win = hist[hist["HomeTeam"]==home]["FTR"].eq("H").mean()
    team_away_win = hist[hist["AwayTeam"]==away]["FTR"].eq("A").mean()
    factor_home = 1.0 + ((team_home_win - lg_home_win) * 0.5)
    factor_away = 1.0 + ((team_away_win - lg_away_win) * 0.5)
    return factor_home, factor_away

def adjust_with_h2h(S, home, away):
    hist = S["hist"]
    h2h = hist[((hist["HomeTeam"]==home)&(hist["AwayTeam"]==away))|((hist["HomeTeam"]==away)&(hist["AwayTeam"]==home))]
    h2h = h2h.sort_values("Date", ascending=False).head(5)
    if h2h.empty:
        return 1.0, 1.0
    wins_home = ((h2h["HomeTeam"]==home)&(h2h["FTR"]=="H")).sum() + ((h2h["AwayTeam"]==home)&(h2h["FTR"]=="A")).sum()
    wins_away = ((h2h["HomeTeam"]==away)&(h2h["FTR"]=="H")).sum() + ((h2h["AwayTeam"]==away)&(h2h["FTR"]=="A")).sum()
    total = len(h2h)
    ratio_home = wins_home/total if total else 0.5
    ratio_away = wins_away/total if total else 0.5
    factor_home = 1.0 + ((ratio_home - 0.5)*0.1)
    factor_away = 1.0 + ((ratio_away - 0.5)*0.1)
    return factor_home, factor_away

# ---------- Odds API ----------
def fetch_odds_matches(sport_key: str, start_iso: str, end_iso: str) -> list:
    """
    Retourne une liste de matchs avec cotes (eu decimal) via The Odds API.
    Chaque item: {home, away, commence_time, odds: {H: , D: , A: }}
    """
    api_key = st.secrets.get("ODDS_API_KEY")
    if not api_key:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": start_is_
