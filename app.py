"""
World Cup Match Outcome Predictor — Streamlit App
==================================================
Run after train_model.py has been executed:

    streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark pitch-green header bar */
    .stApp { background-color: #0f1117; }
    h1 { color: #f5f5f5; font-family: 'Georgia', serif; letter-spacing: -1px; }
    h3 { color: #cccccc; }
    .block-container { padding-top: 2rem; }

    /* Result cards */
    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-home  { background: linear-gradient(135deg, #1a3a5c, #2563ab); }
    .card-draw  { background: linear-gradient(135deg, #2d2d2d, #4a4a4a); }
    .card-away  { background: linear-gradient(135deg, #5c1a1a, #ab2525); }

    .card-label { font-size: 0.85rem; color: #aaa; text-transform: uppercase; letter-spacing: 2px; }
    .card-prob  { font-size: 3rem; font-weight: 800; color: #fff; line-height: 1.1; }
    .card-name  { font-size: 1rem; color: #ddd; margin-top: 4px; }
    .winner-tag { font-size: 0.75rem; background: #f0c040; color: #111;
                  border-radius: 20px; padding: 2px 10px; display: inline-block;
                  margin-top: 6px; font-weight: 700; }

    /* Stat comparison table */
    .stat-row { display: flex; justify-content: space-between;
                padding: 6px 0; border-bottom: 1px solid #2a2a2a; }
    .stat-label { color: #888; font-size: 0.85rem; flex: 1; text-align: center; }
    .stat-val   { color: #f0f0f0; font-size: 0.95rem; font-weight: 600; flex: 1; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model      = joblib.load("wc_predictor_model.pkl")
    team_stats = json.load(open("team_stats.json"))
    teams      = json.load(open("teams.json"))
    h2h_raw    = json.load(open("h2h_stats.json"))
    # rebuild h2h lookup: (date_str, home, away) -> float  — we'll use latest
    h2h = {}
    for key, val in h2h_raw.items():
        parts = key.split("|")
        if len(parts) == 3:
            _, h, a = parts
            h2h[(h, a)] = val          # keep latest (file is sorted chronologically)
    return model, team_stats, teams, h2h

try:
    model, team_stats, teams, h2h = load_artefacts()
except FileNotFoundError as e:
    st.error(
        f"**Missing file:** `{e.filename}`\n\n"
        "Please run `train_model.py` first to generate the model artefacts."
    )
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚽ World Cup Match Predictor")
st.markdown(
    "Predict the outcome of any international match using historical FIFA data "
    "and machine learning."
)
st.divider()

# ── Team selectors ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🏠 Home Team")
    home_team = st.selectbox("Select home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0, label_visibility="collapsed")

with col2:
    st.markdown("### ✈️ Away Team")
    away_options = [t for t in teams if t != home_team]
    default_away = "Argentina" if "Argentina" in away_options else away_options[0]
    away_team = st.selectbox("Select away team", away_options, index=away_options.index(default_away), label_visibility="collapsed")

# ── Match context ─────────────────────────────────────────────────────────────
st.markdown("### 🏟️ Match Settings")
c1, c2, c3 = st.columns(3)
with c1:
    is_neutral  = st.toggle("Neutral venue", value=True)
with c2:
    is_world_cup = st.toggle("FIFA World Cup", value=True)
with c3:
    is_wc_qual = st.toggle("Qualification match", value=False) if not is_world_cup else st.toggle("Qualification match", value=False, disabled=True)

# ── Feature builder ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_win_rate",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_win_rate",
    "h2h_home_win_rate", "is_neutral", "is_world_cup", "is_wc_qual", "form_diff",
]

DEFAULTS = {
    "goals_scored_avg"   : 1.3,
    "goals_conceded_avg" : 1.3,
    "win_rate"           : 0.33,
}

def get_stats(team):
    s = team_stats.get(team, {})
    return {
        "goals_scored_avg"   : s.get("goals_scored_avg",   DEFAULTS["goals_scored_avg"]),
        "goals_conceded_avg" : s.get("goals_conceded_avg", DEFAULTS["goals_conceded_avg"]),
        "win_rate"           : s.get("win_rate",           DEFAULTS["win_rate"]),
    }

def build_features(home, away, neutral, wc, qual):
    hs  = get_stats(home)
    as_ = get_stats(away)
    h2h_rate = h2h.get((home, away), h2h.get((away, home), 0.5))
    if (away, home) in h2h and (home, away) not in h2h:
        h2h_rate = 1 - h2h_rate           # flip perspective

    form_diff = (
        hs["goals_scored_avg"] - hs["goals_conceded_avg"]
      - (as_["goals_scored_avg"] - as_["goals_conceded_avg"])
    )

    return np.array([[
        hs["goals_scored_avg"],   hs["goals_conceded_avg"],  hs["win_rate"],
        as_["goals_scored_avg"],  as_["goals_conceded_avg"], as_["win_rate"],
        h2h_rate,
        int(neutral), int(wc and not qual), int(qual),
        form_diff,
    ]])

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Outcome", use_container_width=True, type="primary"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        X = build_features(home_team, away_team, is_neutral, is_world_cup, is_wc_qual)
        probs  = model.predict_proba(X)[0]
        labels = model.classes_

        prob_map = dict(zip(labels, probs))
        p_home = prob_map.get("home_win", 0)
        p_draw = prob_map.get("draw",     0)
        p_away = prob_map.get("away_win", 0)

        best = max(prob_map, key=prob_map.get)
        best_label = {
            "home_win": f"{home_team} Win",
            "draw"    : "Draw",
            "away_win": f"{away_team} Win",
        }[best]

        st.divider()
        st.markdown("### 📊 Prediction")

        r1, r2, r3 = st.columns(3)

        with r1:
            winner_tag = '<div class="winner-tag">Most likely</div>' if best == "home_win" else ""
            st.markdown(f"""
            <div class="result-card card-home">
                <div class="card-label">Home Win</div>
                <div class="card-prob">{p_home:.0%}</div>
                <div class="card-name">{home_team}</div>
                {winner_tag}
            </div>""", unsafe_allow_html=True)

        with r2:
            winner_tag = '<div class="winner-tag">Most likely</div>' if best == "draw" else ""
            st.markdown(f"""
            <div class="result-card card-draw">
                <div class="card-label">Draw</div>
                <div class="card-prob">{p_draw:.0%}</div>
                <div class="card-name">—</div>
                {winner_tag}
            </div>""", unsafe_allow_html=True)

        with r3:
            winner_tag = '<div class="winner-tag">Most likely</div>' if best == "away_win" else ""
            st.markdown(f"""
            <div class="result-card card-away">
                <div class="card-label">Away Win</div>
                <div class="card-prob">{p_away:.0%}</div>
                <div class="card-name">{away_team}</div>
                {winner_tag}
            </div>""", unsafe_allow_html=True)

        # ── Team stat comparison ───────────────────────────────────────────────
        st.divider()
        st.markdown("### 📈 Team Form Comparison")

        hs  = get_stats(home_team)
        as_ = get_stats(away_team)

        stat_rows = [
            ("Avg Goals Scored",   f"{hs['goals_scored_avg']:.2f}",   f"{as_['goals_scored_avg']:.2f}"),
            ("Avg Goals Conceded", f"{hs['goals_conceded_avg']:.2f}", f"{as_['goals_conceded_avg']:.2f}"),
            ("Recent Win Rate",    f"{hs['win_rate']:.1%}",            f"{as_['win_rate']:.1%}"),
            ("H2H Win Rate",       f"{h2h.get((home_team, away_team), 0.5):.1%}", "—"),
        ]

        header_cols = st.columns([2, 2, 2])
        header_cols[0].markdown(f"**{home_team}**")
        header_cols[1].markdown("**Stat**")
        header_cols[2].markdown(f"**{away_team}**")

        for label, hval, aval in stat_rows:
            c1, c2, c3 = st.columns([2, 2, 2])
            c1.markdown(hval)
            c2.markdown(f"*{label}*")
            c3.markdown(aval)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Gradient Boosting Classifier trained on 49 000+ international matches (1872–2026). "
    "Features include rolling form, head-to-head history, and tournament context. "
    "Predictions are probabilistic estimates, not guarantees."
)
