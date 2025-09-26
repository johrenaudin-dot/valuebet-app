# -*- coding: utf-8 -*-
import re
import unicodedata
from difflib import get_close_matches
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ValueBet — Poisson + Forme + Bonus/Malus", page_icon="⚽", layout="wide")

# ===========================
# Sidebar — paramètres
# ===========================
st.sidebar.header("Paramètres picks")
bankroll   = st.sidebar.number_input("Bankroll (€)", 0, 1_000_000, 1000)
edge_min   = st.sidebar.slider("Edge minimum (p×odds - 1)", 0.00, 0.20, 0.03, 0.01)
kelly_cap  = st.sidebar.slider("Kelly Cap (mise max %)", 0.00, 0.20, 0.05, 0.01)
min_prob   = st.sidebar.slider("Proba minimum du pick (modèle)", 0.00, 0.30, 0.08, 0.01)
min_books  = st.sidebar.slider("Nb minimum de bookmakers", 1, 10, 3)
days_ahead = st.sidebar.slider("Fenêtre (jours à venir)", 0, 7, 3)

st.sidebar.header("Paramètres modèle")
blend_w    = st.sidebar.slider("Mix p_mod (Poisson) vs p_cons (shrinkage)", 0.0, 1.0, 0.70, 0.05)
max_goals  = st.sidebar.slider("Poisson: nb buts max (matrice)", 4, 12, 10)

st.sidebar.header("Forme récente (déjà intégrée)")
use_form        = st.sidebar.checkbox("Utiliser la forme récente (buts)", True)
form_window     = st.sidebar.slider("Taille fenêtre (derniers matches)", 3, 10, 5)
form_decay      = st.sidebar.slider("Décroissance (0.50–0.95)", 0.50, 0.95, 0.75, 0.01)
form_weight     = st.sidebar.slider("Poids de la forme (0–1)", 0.0, 1.0, 0.50, 0.05)

# BONUS/MALUS légers
st.sidebar.header("Bonus/Malus (léger)")
w_home_adv   = st.sidebar.slider("Poids avantage domicile (0–0.15)", 0.00, 0.15, 0.06, 0.01)
w_fatigue    = st.sidebar.slider("Poids fatigue (0–0.15)", 0.00, 0.15, 0.05, 0.01)
w_ppm_form   = st.sidebar.slider("Poids forme en points (0–0.15)", 0.00, 0.15, 0.06, 0.01)

debug_mode = st.sidebar.checkbox("Mode debug (diagnostic)", True)

st.title("🎯 ValueBet — Poisson + Forme + Bonus/Malus vs Marché")

# ===========================
# Utilitaires
# ===========================
def fair_probs_from_odds(oh, od, oa):
    inv = np.array([1/oh, 1/od, 1/oa], dtype=float)
    s = inv.sum()
    if s <= 0:
        return None
    return inv / s

def expected_value(p, o):
    return p*o - 1 if (np.isfinite(p) and np.isfinite(o)) else np.nan

def kelly_fraction(p, o, cap=0.05):
    if not (np.isfinite(p) and np.isfinite(o) and 0 <= p <= 1 and o > 1):
        return 0.0
    b = o - 1.0
    f = (p*b - (1-p)) / b
    return float(max(0.0, min(cap, f)))

def edge_color(edge):
    if edge >= 0.10: return "#1b5e20"
    if edge >= 0.07: return "#2e7d32"
    if edge >= 0.05: return "#ff9800"
    return "#9e9e9e"

def poisson_probs(lam_h, lam_a, max_goals=10):
    gh = np.arange(0, max_goals+1); ga = np.arange(0, max_goals+1)
    fact_h = np.array([np.math.factorial(int(i)) for i in gh])
    fact_a = np.array([np.math.factorial(int(i)) for i in ga])
    ph = np.exp(-lam_h) * (lam_h**gh) / fact_h
    pa = np.exp(-lam_a) * (lam_a**ga) / fact_a
    mat = np.outer(ph, pa)
    p_home = np.tril(mat, -1).sum(); p_draw = np.trace(mat); p_away = np.triu(mat, 1).sum()
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s
    return float(p_home), float(p_draw), float(p_away)

# ----- Normalisation / alias / matching proche -----
def normalize_name(name: str) -> str:
    if not isinstance(name, str): return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

ALIASES = {
    "psg": "paris saint germain",
    "man city": "manchester city",
    "man utd": "manchester united",
    "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "ath bilbao": "athletic bilbao",
    "inter": "internazionale",
    "monchengladbach": "borussia monchengladbach",
    "bayern munchen": "bayern munich",
    "leverkusen": "bayer leverkusen",
    "roma": "as roma",
    "milan": "ac milan",
}

def resolve_team_name(name: str, known_names: list[str]) -> str:
    norm = normalize_name(name)
    norm = ALIASES.get(norm, norm)
    mapping = {normalize_name(n): n for n in known_names}
    if norm in mapping:
        return mapping[norm]
    close = get_close_matches(norm, list(mapping.keys()), n=1, cutoff=0.90)
    if close:
        return mapping[close[0]]
    return name

# ===========================
# Cartographie ligues & sources
# ===========================
sports = {
    "Premier League": ("soccer_epl",    "E0"),
    "La Liga":        ("soccer_spain_la_liga", "SP1"),
    "Serie A":        ("soccer_italy_serie_a", "I1"),
    "Bundesliga":     ("soccer_germany_bundesliga", "D1"),
    "Ligue 1":        ("soccer_france_ligue_one",   "F1"),
    "UCL":            ("soccer_uefa_champs_league", "EC"),
    "UEL":            ("soccer_uefa_europa_league", "EL"),
}

def season_path(season_start, code):
    def yy(y): return f"{y%100:02d}"
    return f"https://www.football-data.co.uk/mmz4281/{yy(season_start)}{yy(season_start+1)}/{code}.csv"

def fetch_hist(code, season_start):
    url = season_path(season_start, code)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        needed = {"Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"}
        if not needed.issubset(df.columns):
            return None
        def parse_date(x):
            for fmt in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d"):
                try:
                    return datetime.strptime(str(x), fmt).date()
                except:
                    pass
            return None
        df["Date"] = df["Date"].apply(parse_date)
        df = df.dropna(subset=["Date"])
        for c in ["FTHG","FTAG"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["FTHG","FTAG"])
        return df
    except Exception:
        return None

def build_poisson_strengths(hist):
    base_h = hist["FTHG"].mean()
    base_a = hist["FTAG"].mean()
    if base_h <= 0: base_h = 1e-6
    if base_a <= 0: base_a = 1e-6
    ha = hist.groupby("HomeTeam")["FTHG"].mean() / base_h
    hd = hist.groupby("HomeTeam")["FTAG"].mean() / base_a
    aa = hist.groupby("AwayTeam")["FTAG"].mean() / base_a
    ad = hist.groupby("AwayTeam")["FTHG"].mean() / base_h
    return dict(base_h=base_h, base_a=base_a, ha=ha, hd=hd, aa=aa, ad=ad)

# --------- Forme récente (buts) ----------
def _weighted_recent_mean(vals, weights):
    vals = np.array(vals, dtype=float)
    weights = np.array(weights, dtype=float)
    m = np.sum(vals * weights); s = np.sum(weights)
    return m / s if s > 0 else np.nan

def compute_form_multipliers(hist, today_dt, base_h, base_a, k=5, decay=0.75):
    H_att, H_def, A_att, A_def = {}, {}, {}, {}
    hist2 = hist[hist["Date"] < today_dt.date()].copy()
    if hist2.empty: return H_att, H_def, A_att, A_def
    hist2 = hist2.sort_values("Date")
    teams = list(set(hist2["HomeTeam"]) | set(hist2["AwayTeam"]))
    for team in teams:
        hmat = hist2[hist2["HomeTeam"] == team].tail(k)
        if not hmat.empty:
            n = len(hmat); w = np.array([decay**(n-1-i) for i in range(n)])
            att = _weighted_recent_mean(hmat["FTHG"].values, w)
            dff = _weighted_recent_mean(hmat["FTAG"].values, w)
            H_att[team] = max(att / base_h, 1e-6)
            H_def[team] = max(dff / base_a, 1e-6)
        amat = hist2[hist2["AwayTeam"] == team].tail(k)
        if not amat.empty:
            n = len(amat); w = np.array([decay**(n-1-i) for i in range(n)])
            att = _weighted_recent_mean(amat["FTAG"].values, w)
            dff = _weighted_recent_mean(amat["FTHG"].values, w)
            A_att[team] = max(att / base_a, 1e-6)
            A_def[team] = max(dff / base_h, 1e-6)
    return H_att, H_def, A_att, A_def

def apply_form(m, weight):
    return (1 - weight) + weight * float(m if np.isfinite(m) else 1.0)

def lambda_match_with_form(S, home, away, form_dicts, weight):
    lam_h, lam_a = lambda_match(S, home, away)
    if form_dicts is None or weight <= 0:
        return lam_h, lam_a
    H_att, H_def, A_att, A_def = form_dicts
    mh_att = apply_form(H_att.get(home, 1.0), weight)
    mh_def = apply_form(H_def.get(home, 1.0), weight)
    ma_att = apply_form(A_att.get(away, 1.0), weight)
    ma_def = apply_form(A_def.get(away, 1.0), weight)
    lam_h *= mh_att * ma_def
    lam_a *= ma_att * mh_def
    return max(lam_h, 1e-6), max(lam_a, 1e-6)

def lambda_match(S, home, away):
    bh, ba = S["base_h"], S["base_a"]
    lam_h = bh * float(S["ha"].get(home, 1.0)) * float(S["ad"].get(away, 1.0))
    lam_a = ba * float(S["aa"].get(away, 1.0)) * float(S["hd"].get(home, 1.0))
    lam_h = max(lam_h, 1e-6); lam_a = max(lam_a, 1e-6)
    return lam_h, lam_a

# --------- BONUS/MALUS légers à partir d'historiques ----------
def compute_home_advantage_boost(hist, k=6):
    """Winrate domicile récente vs winrate ligue → multiplicateur ~1±(w_home_adv * delta)."""
    h = hist.sort_values("Date")
    # winrate ligue
    lg_home_wr = (h["FTR"] == "H").mean()
    # winrate domicile par équipe (sur les k derniers matches à domicile)
    boost = {}
    for team, dfh in h.groupby("HomeTeam"):
        last = dfh.tail(k)
        if last.empty: 
            continue
        wr = (last["FTR"] == "H").mean()
        boost[team] = wr - lg_home_wr  # delta ∈ [-1,1], mais réalistiquement ~[-0.4,0.4]
    return boost, lg_home_wr

def compute_fatigue_factor(hist, today_dt, min_rest=4, cap=0.08):
    """
    Malus si repos < min_rest jours. 
    Retourne dict team -> factor in [1-cap, 1], ex 0.92–1.00
    """
    h = hist[hist["Date"] < today_dt.date()].copy().sort_values("Date")
    factor = {}
    # dates du dernier match
    last_date = {}
    for _, r in h.iterrows():
        d = r["Date"]
        th, ta = r["HomeTeam"], r["AwayTeam"]
        last_date[th] = d
        last_date[ta] = d
    for team, d in last_date.items():
        rest = (today_dt.date() - d).days
        if rest < min_rest:
            # pénalité linéaire jusqu'à cap (ex 8%) si 0 repos ; 0 si repos >= min_rest
            pen = min(cap, cap * (min_rest - rest) / max(min_rest,1))
            factor[team] = 1.0 - pen
        else:
            factor[team] = 1.0
    return factor

def compute_points_form(hist, k=6):
    """
    Points par match récents vs ppm ligue → multiplicateur ~1±(w_ppm_form * delta_norm)
    """
    h = hist.sort_values("Date")
    # points par match ligue
    pts = []
    for _, r in h.iterrows():
        res = r["FTR"]
        pts_home = 3 if res == "H" else (1 if res == "D" else 0)
        pts_away = 3 if res == "A" else (1 if res == "D" else 0)
        pts.append(("H", r["HomeTeam"], pts_home))
        pts.append(("A", r["AwayTeam"], pts_away))
    dfp = pd.DataFrame(pts, columns=["Side","Team","Pts"])
    lg_ppm = dfp["Pts"].mean()  # ~1.3 typiquement

    # ppm récents par équipe (domicile et extérieur confondus)
    ppm = {}
    for team in pd.unique(dfp["Team"]):
        # Prend les k dernières apparitions (home/away)
        sel = []
        for _, r in h.iterrows():
            if r["HomeTeam"] == team:
                sel.append(3 if r["FTR"]=="H" else (1 if r["FTR"]=="D" else 0))
            elif r["AwayTeam"] == team:
                sel.append(3 if r["FTR"]=="A" else (1 if r["FTR"]=="D" else 0))
        if sel:
            sel = sel[-k:]
            ppm[team] = np.mean(sel)
    return ppm, lg_ppm

def apply_bonus_malus(lam_h, lam_a, home, away,
                      home_boosts, fatigue_factors, ppm_team, lg_ppm,
                      w_home_adv=0.06, w_fatigue=0.05, w_ppm=0.06):
    """
    Applique de petits multiplicateurs (tous ~1±qch de faible):
      - Avantage domicile récent
      - Fatigue (jours de repos)
      - Forme PPM récente
    """
    # Avantage domicile récent (bonus home, micro malus away)
    if w_home_adv > 0:
        delta = float(home_boosts.get(home, 0.0))  # peut être négatif
        m = 1.0 + w_home_adv * delta
        lam_h *= m
        lam_a /= max(1e-6, m**0.25)  # petit contre-effet

    # Fatigue (malus si peu de repos)
    if w_fatigue > 0:
        fh = float(fatigue_factors.get(home, 1.0))
        fa = float(fatigue_factors.get(away, 1.0))
        lam_h *= fh
        lam_a *= fa

    # Forme en points (bonus/malus)
    if w_ppm > 0 and lg_ppm > 0:
        mh = 1.0 + w_ppm * ((float(ppm_team.get(home, lg_ppm)) - lg_ppm) / lg_ppm)
        ma = 1.0 + w_ppm * ((float(ppm_team.get(away, lg_ppm)) - lg_ppm) / lg_ppm)
        lam_h *= mh
        lam_a *= ma

    return max(lam_h, 1e-6), max(lam_a, 1e-6)

# ===========================
# Odds API (fixtures + cotes)
# ===========================
API_KEY = st.secrets.get("ODDS_API_KEY", "")
if not API_KEY:
    st.error("❌ ODDS_API_KEY manquante dans les *Secrets* Streamlit.")
    st.stop()

BASE_URL = "https://api.the-odds-api.com/v4/sports"
date_from = datetime.now(timezone.utc)
date_to   = date_from + timedelta(days=days_ahead)

# ===========================
# Historiques & forces + BONUS/MALUS
# ===========================
st.subheader("Téléchargement des historiques (football-data) & calcul forces / forme / bonus-malus")
season_start_default = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
season_start = st.number_input("Saison (année de début)", season_start_default-1, season_start_default, season_start_default)

strengths_by_league = {}
form_by_league      = {}
home_boost_by_league = {}
fatigue_by_league    = {}
ppm_by_league        = {}
lg_ppm_by_league     = {}

load_msgs = []
for lig, (_, code) in sports.items():
    df_hist = fetch_hist(code, season_start)
    if df_hist is None or df_hist.empty:
        load_msgs.append(f"⚠️ {lig}: historique indisponible.")
        continue
    S = build_poisson_strengths(df_hist)
    strengths_by_league[lig] = S

    # forme (buts)
    if use_form:
        H_att, H_def, A_att, A_def = compute_form_multipliers(
            df_hist, today_dt=date_from, base_h=S["base_h"], base_a=S["base_a"],
            k=form_window, decay=form_decay
        )
        form_by_league[lig] = (H_att, H_def, A_att, A_def)
    else:
        form_by_league[lig] = None

    # BONUS/MALUS:
    hb, lg_hw = compute_home_advantage_boost(df_hist, k=6)
    home_boost_by_league[lig] = hb
    fatigue_by_league[lig]    = compute_fatigue_factor(df_hist, date_from, min_rest=4, cap=0.08)
    ppm_map, lg_ppm = compute_points_form(df_hist, k=6)
    ppm_by_league[lig]    = ppm_map
    lg_ppm_by_league[lig] = lg_ppm

    load_msgs.append(f"✅ {lig}: {len(df_hist)} matches — Forme:{'on' if use_form else 'off'}; bonus/malus actifs")

for m in load_msgs:
    st.caption(m)

# ===========================
# Calcul des picks
# ===========================
st.subheader("Calcul des value bets (Poisson + Forme + Bonus/Malus vs Marché)")
all_rows, diag_rows = [], []

for lig, (sport_key, _) in sports.items():
    if lig not in strengths_by_league:
        diag_rows.append({"Ligue": lig, "Matches fenêtre": 0, "Avec cotes": 0, "Erreur API": "no_hist"})
        continue

    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            diag_rows.append({"Ligue": lig, "Matches fenêtre": 0, "Avec cotes": 0, "Erreur API": f"{r.status_code} {r.text[:60]}"})
            continue
        data = r.json()
    except Exception as e:
        diag_rows.append({"Ligue": lig, "Matches fenêtre": 0, "Avec cotes": 0, "Erreur API": f"{type(e).__name__}"})
        continue

    nb_matches, nb_with_odds = 0, 0
    S = strengths_by_league[lig]
    form_dicts = form_by_league[lig]
    home_boosts = home_boost_by_league[lig]
    fatigue_f   = fatigue_by_league[lig]
    ppm_team    = ppm_by_league[lig]
    lg_ppm      = lg_ppm_by_league[lig]

    known_home_names = list(S["ha"].index)
    known_away_names = list(S["aa"].index)

    for mt in data:
        nb_matches += 1
        try:
            date_iso = mt["commence_time"]
            home     = mt["home_team"]
            away     = mt["away_team"]
        except KeyError:
            continue

        odds_all = []
        for bk in mt.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "h2h":
                    prices = mk.get("outcomes", [])
                    names = [p.get("name") for p in prices]
                    if {"Draw", home, away}.issubset(set(names)):
                        o = {p["name"]: p["price"] for p in prices}
                        odds_all.append(o)

        if len(odds_all) < min_books:
            continue
        nb_with_odds += 1

        bestH = max([o.get(home, np.nan) for o in odds_all])
        bestD = max([o.get("Draw", np.nan) for o in odds_all])
        bestA = max([o.get(away, np.nan) for o in odds_all])
        if np.isnan([bestH, bestD, bestA]).any():
            continue

        # consensus marché
        fair = []
        for o in odds_all:
            oh, od, oa = o[home], o["Draw"], o[away]
            pro = fair_probs_from_odds(oh, od, oa)
            if pro is not None:
                fair.append(pro)
        if not fair:
            continue
        fair = np.array(fair)
        p_cons = np.median(fair, axis=0)

        # matching noms
        home_res = resolve_team_name(home, known_home_names)
        away_res = resolve_team_name(away, known_away_names)

        # λ Poisson avec forme
        lam_h, lam_a = lambda_match_with_form(S, home_res, away_res, form_dicts, form_weight if use_form else 0.0)
        # BONUS/MALUS légers
        lam_h, lam_a = apply_bonus_malus(lam_h, lam_a, home_res, away_res,
                                         home_boosts, fatigue_f, ppm_team, lg_ppm,
                                         w_home_adv=w_home_adv, w_fatigue=w_fatigue, w_ppm=w_ppm_form)

        # Probas Poisson → mélange avec consensus
        pH_p, pD_p, pA_p = poisson_probs(lam_h, lam_a, max_goals=max_goals)
        p_mod_poisson = np.array([pH_p, pD_p, pA_p])
        p_mod = blend_w * p_mod_poisson + (1 - blend_w) * p_cons

        picks = {
            "H": (home, bestH, p_cons[0], p_mod[0]),
            "D": ("Draw", bestD, p_cons[1], p_mod[1]),
            "A": (away, bestA, p_cons[2], p_mod[2]),
        }

        for label, (team, odds, pc, pm) in picks.items():
            if odds <= 1:
                continue
            edge_c = expected_value(pc, odds)
            edge_m = expected_value(pm, odds)

            # Garde-fous “laser focus”
            if edge_c <= 0:      # marché doit déjà être >0
                continue
            if edge_m < edge_min:
                continue
            if pm < pc:
                continue
            if pm < min_prob:
                continue

            kelly = kelly_fraction(pm, odds, cap=kelly_cap)
            stake = bankroll * kelly

            all_rows.append({
                "Date": date_iso,
                "Ligue": lig,
                "Match": f"{home} vs {away}",
                "Pick": team,
                "Odds": round(float(odds),2),
                "p_cons": round(float(pc),3),
                "p_mod": round(float(pm),3),
                "Edge": round(float(edge_m),3),
                "Kelly%": round(kelly*100,2),
                "Stake€": round(stake,2),
            })

    diag_rows.append({"Ligue": lig, "Matches fenêtre": nb_matches, "Avec cotes": nb_with_odds, "Erreur API": ""})

# ===========================
# Résultats
# ===========================
if not all_rows:
    st.info("Aucun value bet trouvé avec ces paramètres.")
else:
    df = pd.DataFrame(all_rows).sort_values(["Edge","Stake€"], ascending=[False, False]).reset_index(drop=True)
    st.metric("🎲 Nombre de picks trouvés", len(df))
    for _, r in df.iterrows():
        bg = edge_color(r["Edge"])
        st.markdown(
            f"""
<div style="background:{bg};padding:14px;border-radius:12px;margin:8px 0;color:white">
  <div style="font-weight:700">{r['Date']} · {r['Ligue']}</div>
  <div style="font-size:18px;margin:2px 0">{r['Match']}</div>
  <div>✅ Pick: <b>{r['Pick']}</b></div>
  <div>Cote: {r['Odds']} | p_cons: {r['p_cons']} | p_mod: {r['p_mod']} | Edge: {r['Edge']}</div>
  <div>Mise: {r['Stake€']}€ (Kelly {r['Kelly%']}%)</div>
</div>
            """,
            unsafe_allow_html=True
        )

if debug_mode:
    st.subheader("Diagnostic par ligue")
    st.dataframe(pd.DataFrame(diag_rows))
