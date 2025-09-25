# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="ValueBet - Today's Picks", page_icon=":soccer:", layout="wide")

# --------- Config ----------
SEASON_START = 2025  # essaiera 2025-26 puis 2024-25 si besoin
LEAGUE_CODES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "UCL": "EC",
    "UEL": "EL",
}
MAX_GOALS = 10
TZ = ZoneInfo("Europe/Paris")

# --------- Utils ----------
def yy(y: int) -> str:
    return f"{y%100:02d}"

def season_path(season_start: int, code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{yy(season_start)}{yy(season_start+1)}/{code}.csv"

def fetch_csv(url: str) -> pd.DataFrame | None:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        return df
    except Exception:
        return None

def coerce_date_col(s: pd.Series) -> pd.Series:
    def parse_one(x):
        for fmt in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d"):
            try:
                return datetime.strptime(str(x), fmt).date()
            except Exception:
                continue
        return pd.NaT
    return s.apply(parse_one)

def pick_odds_row(row):
    cols_ps = ["PSH","PSD","PSA"]
    cols_b = ["B365H","B365D","B365A"]
    if all(c in row.index for c in cols_ps) and not pd.isna(row.get("PSH")):
        return row["PSH"], row["PSD"], row["PSA"]
    if all(c in row.index for c in cols_b) and not pd.isna(row.get("B365H")):
        return row["B365H"], row["B365D"], row["B365A"]
    return None, None, None

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

def kelly_fraction(p, o, cap=0.05):
    if not (np.isfinite(p) and np.isfinite(o) and o>1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def expected_value(p, o):
    if not (np.isfinite(p) and np.isfinite(o)):
        return np.nan
    return float(p*o - 1.0)

def team_strengths_poisson(df: pd.DataFrame, today: datetime.date):
    df = df.copy()
    if "Date" not in df.columns:
        return None
    df["Date"] = coerce_date_col(df["Date"])
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

# --------- Chargement ligue avec fallback de saison ----------
@st.cache_data(show_spinner=False)
def load_league_with_fallback(code: str, season_start: int):
    # essaie saison courante puis saison precedente
    for s in (season_start, season_start-1):
        url = season_path(s, code)
        df = fetch_csv(url)
        if df is not None:
            return df, f"{s}-{s+1}"
    return None, None

# ================= UI =================
st.sidebar.header("Parametres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=1000.0, step=100.0, format="%.0f")
min_edge = st.sidebar.slider("Edge minimum", 0.0, 0.15, 0.03, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap (mise max %)", 0.0, 0.2, 0.05, 0.01)
debug_mode = st.sidebar.checkbox("Mode debug (afficher tous les matchs)")

base_day = st.sidebar.date_input("Date de reference", datetime.now(TZ).date())
days_ahead = st.sidebar.slider("Jours a venir (0-7)", 0, 7, 2)
start_day = base_day
end_day = base_day + timedelta(days=days_ahead)

if st.sidebar.button("Recharger les donnees (vider le cache)"):
    st.cache_data.clear()

leagues_selected = st.sidebar.multiselect(
    "Ligues",
    list(LEAGUE_CODES.keys()),
    default=["Premier League","La Liga","Serie A","Bundesliga","Ligue 1","UCL","UEL"]
)
st.sidebar.write(f"Fenetre: {start_day.isoformat()} -> {end_day.isoformat()}")

# ================= Core =================
rows = []          # picks (value bets)
diag_rows = []     # resume par ligue
debug_rows = []    # details match par match

for lname in leagues_selected:
    code = LEAGUE_CODES[lname]
    df, season_used = load_league_with_fallback(code, SEASON_START)
    if df is None:
        diag_rows.append({"Ligue": lname, "Saison": None, "CSV charge": False,
                          "MinDate": None, "MaxDate": None, "Matchs fenetre": 0, "Avec cotes": 0})
        continue

    if "Date" not in df.columns:
        diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": True,
                          "MinDate": None, "MaxDate": None, "Matchs fenetre": 0, "Avec cotes": 0})
        continue

    df["Date"] = coerce_date_col(df["Date"])
    df = df[df["Date"].notna()]

    min_d = df["Date"].min()
    max_d = df["Date"].max()

    S = team_strengths_poisson(df, base_day)
    if S is None:
        diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": True,
                          "MinDate": min_d, "MaxDate": max_d, "Matchs fenetre": 0, "Avec cotes": 0})
        continue

    window = df[(df["Date"] >= start_day) & (df["Date"] <= end_day)].copy()
    if window.empty:
        diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": True,
                          "MinDate": min_d, "MaxDate": max_d, "Matchs fenetre": 0, "Avec cotes": 0})
        continue

    with_odds_count = 0

    for _, r in window.iterrows():
        home, away, d = r["HomeTeam"], r["AwayTeam"], r["Date"]
        oh, od, oa = pick_odds_row(r)
        if oh is not None and od is not None and oa is not None:
            with_odds_count += 1

        # Debug row (toujours)
        dbg_line = {"Date": d.isoformat(), "Ligue": lname, "Home": home, "Away": away,
                    "Odds_H": oh, "Odds_D": od, "Odds_A": oa}
        # Si pas de cotes -> on continue pour debug mais pas de pick
        if oh is None or od is None or oa is None:
            dbg_line.update({"Proba_H": np.nan, "Proba_D": np.nan, "Proba_A": np.nan,
                             "Edge_H": np.nan, "Edge_D": np.nan, "Edge_A": np.nan,
                             "BestPick": None, "BestEdge": np.nan, "Motif": "Cotes manquantes"})
            debug_rows.append(dbg_line)
            continue

        # Lambda et ajustements
        lam_h = (S["base_h"] if S["base_h"]>0 else 1)*float(S["ha"].get(home,1.0))*float(S["ad"].get(away,1.0))
        lam_a = (S["base_a"] if S["base_a"]>0 else 1)*float(S["aa"].get(away,1.0))*float(S["hd"].get(home,1.0))
        f_home, f_away_dom = adjust_with_home_away(S, home, away)
        f_home_h2h, f_away_h2h = adjust_with_h2h(S, home, away)
        lam_h *= f_home * f_home_h2h
        lam_a *= f_away_dom * f_away_h2h

        ph, pd_, pa = poisson_probs(lam_h, lam_a, MAX_GOALS)
        edge_h = expected_value(ph, oh)
        edge_d = expected_value(pd_, od)
        edge_a = expected_value(pa, oa)

        pick_map = {"H": (ph, oh, edge_h), "D": (pd_, od, edge_d), "A": (pa, oa, edge_a)}
        best_label, (p_star, o_star, edge_star) = max(
            pick_map.items(),
            key=lambda kv: (kv[1][2] if kv[1][2] is not None else -9e9)
        )

        # ligne debug
        dbg_line.update({
            "Proba_H": round(ph,3), "Proba_D": round(pd_,3), "Proba_A": round(pa,3),
            "Edge_H": round(edge_h,3), "Edge_D": round(edge_d,3), "Edge_A": round(edge_a,3),
            "BestPick": best_label, "BestEdge": round(edge_star,3) if edge_star is not None else np.nan,
            "Motif": "" if (edge_star is not None and edge_star >= min_edge) else "Edge < seuil"
        })
        debug_rows.append(dbg_line)

        # applique le filtre value bet
        if edge_star is None or edge_star < min_edge:
            continue
        kelly = kelly_fraction(p_star, o_star, cap=kelly_cap)
        stake = bankroll * kelly
        rows.append({
            "Date": d.isoformat(),
            "Ligue": lname,
            "Home": home,
            "Away": away,
            "Pick": best_label,
            "Cote": round(o_star, 2),
            "ProbaModele": round(p_star, 3),
            "Edge": round(edge_star, 3),
            "MiseEUR": round(stake, 2),
            "KellyPct": round(100*kelly, 2),
        })

    diag_rows.append({"Ligue": lname, "Saison": season_used, "CSV charge": True,
                      "MinDate": min_d, "MaxDate": max_d,
                      "Matchs fenetre": len(window), "Avec cotes": with_odds_count})

# ================= UI rendu =================
st.title("ValueBet - Picks du jour")
st.caption("Poisson ajuste (domicile/exterieur, H2H). Utilise la fenetre de dates, le rechargement des donnees et le mode debug pour diagnostiquer.")

# Tableau principal
if not rows:
    st.info("Aucun value bet trouve pour la fenetre et les parametres actuels.")
else:
    dfp = pd.DataFrame(rows).sort_values(["Edge","MiseEUR"], ascending=[False,False]).reset_index(drop=True)
    st.metric("Value bets trouves", len(dfp))
    st.dataframe(dfp, use_container_width=True, height=min(600, 100+35*len(dfp)))

# Diagnostics par ligue
st.subheader("Diagnostic par ligue")
if diag_rows:
    dsum = pd.DataFrame(diag_rows)
    # formate dates
    for c in ("MinDate","MaxDate"):
        if c in dsum.columns:
            dsum[c] = dsum[c].astype(str)
    st.dataframe(dsum, use_container_width=True)

# Debug detaille
if st.sidebar.checkbox("Afficher le tableau debug detaille (tous les matchs)", value=True):
    st.subheader("Debug - Tous les matchs dans la fenetre")
    if debug_rows:
        dfd = pd.DataFrame(debug_rows)
        cols = ["Date","Ligue","Home","Away","Odds_H","Odds_D","Odds_A",
                "Proba_H","Proba_D","Proba_A","Edge_H","Edge_D","Edge_A","BestPick","BestEdge","Motif"]
        dfd = dfd.reindex(columns=[c for c in cols if c in dfd.columns])
        st.dataframe(dfd, use_container_width=True, height=min(700, 120+30*len(dfd)))
    else:
        st.write("Aucun match recense dans la fenetre pour les ligues selectionnees.")
