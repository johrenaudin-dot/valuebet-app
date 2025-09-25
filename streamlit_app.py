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

# football-data (historique)
FD_CODES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "UCL": "EC",
    "UEL": "EL",
}

# The Odds API (fixtures + cotes)
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
        r = requests.get(url, timeout=8)  # timeout court anti-spinner
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
    Chaque item: {home, away, time(ISO), odds: {H,D,A}}
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
        "commenceTimeFrom": start_iso,
        "commenceTimeTo": end_iso,
    }
    try:
        r = requests.get(url, params=params, timeout=8)  # timeout court
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    out = []
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        t = ev.get("commence_time")
        best = {"H": None, "D": None, "A": None}
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name = (o.get("name") or "").lower()
                    price = o.get("price")
                    if name in ("home","draw","away"):
                        key = {"home":"H","draw":"D","away":"A"}[name]
                        if price and (best[key] is None or price > best[key]):
                            best[key] = price
                    else:
                        # parfois le nom est l'equipe
                        if home and o.get("name") == home and price and (best["H"] is None or price > best["H"]):
                            best["H"] = price
                        if away and o.get("name") == away and price and (best["A"] is None or price > best["A"]):
                            best["A"] = price
        out.append({"home": home, "away": away, "time": t, "odds": best})
    return out

# ---------- UI ----------
st.sidebar.header("Parametres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=1000.0, step=100.0, format="%.0f")
min_edge = st.sidebar.slider("Edge minimum", 0.0, 0.15, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.2, 0.05, 0.01)
debug_mode = st.sidebar.checkbox("Mode debug (afficher tous les matchs)", value=True)

base_day = st.sidebar.date_input("Date de reference", datetime.now(TZ).date())
days_ahead = st.sidebar.slider("Jours a venir (0-7)", 0, 7, 2)
start_day = base_day
end_day = base_day + timedelta(days=days_ahead)
st.sidebar.write(f"Fenetre: {start_day.isoformat()} -> {end_day.isoformat()}")

if st.sidebar.button("Recharger les donnees (vider le cache)"):
    st.cache_data.clear()

leagues_selected = st.sidebar.multiselect(
    "Ligues",
    list(FD_CODES.keys()),
    default=["Premier League","La Liga","Serie A","Bundesliga","Ligue 1","UCL","UEL"]
)

# ---------- Chargement historiques (forces) ----------
@st.cache_data(show_spinner=False)
def load_fd_with_fallback(code: str, season_start: int):
    for s in (season_start, season_start-1):
        url = fd_url(s, code)
        df = fetch_csv(url)
        if df is not None:
            return df, f"{s}-{s+1}"
    return None, None

# ---------- Corps ----------
rows = []
diag_rows = []
debug_rows = []

start_iso = datetime.combine(start_day, datetime.min.time(), tzinfo=TZ).isoformat()
end_iso = datetime.combine(end_day, datetime.max.time(), tzinfo=TZ).isoformat()

status = st.status("Chargement des ligues...", expanded=False)

for idx, lname in enumerate(leagues_selected, 1):
    status.update(label=f"Chargement: {lname} ({idx}/{len(leagues_selected)})", state="running")

    # forces via football-data
    fd_code = FD_CODES[lname]
    df_hist, season_used = load_fd_with_fallback(fd_code, SEASON_START)

    if df_hist is None or "Date" not in df_hist.columns:
        diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": bool(df_hist is not None),
                          "Matchs fenetre": 0, "Avec cotes": 0})
        continue

    df_hist["Date"] = parse_date_series(df_hist["Date"])
    S = team_strengths_poisson(df_hist, base_day)
    if S is None:
        diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": True,
                          "Matchs fenetre": 0, "Avec cotes": 0})
        continue

    # fixtures + cotes via The Odds API
    sport_key = ODDS_SPORT_KEYS[lname]
    odds_events = fetch_odds_matches(sport_key, start_iso, end_iso)

    with_odds_count = 0
    for ev in odds_events:
        home = ev["home"]; away = ev["away"]
        odds = ev["odds"]; oh, od, oa = odds["H"], odds["D"], odds["A"]
        if oh is not None and od is not None and oa is not None:
            with_odds_count += 1

        dbg = {"Date": ev["time"], "Ligue": lname, "Home": home, "Away": away,
               "Odds_H": oh, "Odds_D": od, "Odds_A": oa}

        if oh is None or od is None or oa is None:
            dbg.update({"Proba_H": np.nan, "Proba_D": np.nan, "Proba_A": np.nan,
                        "Edge_H": np.nan, "Edge_D": np.nan, "Edge_A": np.nan,
                        "BestPick": None, "BestEdge": np.nan, "Motif": "Cotes manquantes"})
            debug_rows.append(dbg)
            continue

        lam_h = (S["base_h"] if S["base_h"]>0 else 1)*float(S["ha"].get(home,1.0))*float(S["ad"].get(away,1.0))
        lam_a = (S["base_a"] if S["base_a"]>0 else 1)*float(S["aa"].get(away,1.0))*float(S["hd"].get(home,1.0))
        f_home, f_away_dom = adjust_with_home_away(S, home, away)
        f_home_h2h, f_away_h2h = adjust_with_h2h(S, home, away)
        lam_h *= f_home * f_home_h2h
        lam_a *= f_away_dom * f_away_h2h

        ph, pd_, pa = poisson_probs(lam_h, lam_a, MAX_GOALS)
        eh, ed, ea = expected_value(ph, oh), expected_value(pd_, od), expected_value(pa, oa)

        pick_map = {"H": (ph, oh, eh), "D": (pd_, od, ed), "A": (pa, oa, ea)}
        best_label, (p_star, o_star, edge_star) = max(
            pick_map.items(), key=lambda kv: (kv[1][2] if kv[1][2] is not None else -9e9)
        )

        dbg.update({
            "Proba_H": round(ph,3), "Proba_D": round(pd_,3), "Proba_A": round(pa,3),
            "Edge_H": round(eh,3), "Edge_D": round(ed,3), "Edge_A": round(ea,3),
            "BestPick": best_label, "BestEdge": round(edge_star,3) if edge_star is not None else np.nan,
            "Motif": "" if (edge_star is not None and edge_star >= min_edge) else "Edge < seuil"
        })
        debug_rows.append(dbg)

        if edge_star is None or edge_star < min_edge:
            continue
        kelly = kelly_fraction(p_star, o_star, cap=kelly_cap)
        stake = bankroll * kelly
        rows.append({
            "Date": ev["time"],
            "Ligue": lname,
            "Home": home,
            "Away": away,
            "Pick": best_label,
            "Cote": round(o_star,2),
            "ProbaModele": round(p_star,3),
            "Edge": round(edge_star,3),
            "MiseEUR": round(stake,2),
            "KellyPct": round(100*kelly,2),
        })

    diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": True,
                      "Matchs fenetre": len(odds_events), "Avec cotes": with_odds_count})

status.update(label="Terminé", state="complete")

# ---------- Rendu ----------
st.title("ValueBet - Picks du jour")
st.caption("Poisson (forces via football-data) + fixtures/cotes temps reel via The Odds API. Mode debug et diagnostics inclus.")

if not rows:
    st.info("Aucun value bet trouve pour la fenetre et les parametres actuels.")
else:
    dfp = pd.DataFrame(rows).sort_values(["Edge","MiseEUR"], ascending=[False,False]).reset_index(drop=True)
    st.metric("Value bets trouves", len(dfp))
    st.dataframe(dfp, use_container_width=True, height=min(600, 100+35*len(dfp)))

st.subheader("Diagnostic par ligue")
if diag_rows:
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

if debug_mode:
    st.subheader("Debug - Tous les matchs dans la fenetre")
    if debug_rows:
        cols = ["Date","Ligue","Home","Away","Odds_H","Odds_D","Odds_A",
                "Proba_H","Proba_D","Proba_A","Edge_H","Edge_D","Edge_A","BestPick","BestEdge","Motif"]
        dfd = pd.DataFrame(debug_rows)
        dfd = dfd.reindex(columns=[c for c in cols if c in dfd.columns])
        st.dataframe(dfd, use_container_width=True, height=min(700, 120+30*len(dfd)))
    else:
        st.write("Aucun match recense dans la fenetre (ou pas de cotes).")
