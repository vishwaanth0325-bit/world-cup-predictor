"""
World Cup Match Outcome Predictor — Model Training
====================================================
Run this ONCE in Colab to train the model and save it.
Then run app.py with Streamlit to deploy the UI.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
results     = pd.read_csv("results.csv")
goalscorers = pd.read_csv("goalscorers.csv")
shootouts   = pd.read_csv("shootouts.csv")
former_names = pd.read_csv("former_names.csv")

# ── 2. Normalise team names (handle former names) ─────────────────────────────
name_map = dict(zip(former_names["former"], former_names["current"]))

def normalise(name):
    return name_map.get(name, name)

for col in ["home_team", "away_team"]:
    results[col]     = results[col].map(normalise)
    goalscorers[col] = goalscorers[col].map(normalise)
    shootouts[col]   = shootouts[col].map(normalise)

# ── 3. Label outcomes ─────────────────────────────────────────────────────────
results["date"] = pd.to_datetime(results["date"])

def label_outcome(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

results["outcome"] = results.apply(label_outcome, axis=1)

# ── 4. Build per-team rolling stats (last N matches, all competitions) ────────
print("Engineering features...")

results_sorted = results.sort_values("date").reset_index(drop=True)

def team_rolling_stats(df, team_col, score_col, opp_score_col, window=10):
    """Return a dict: {(date, team): {goals_scored_avg, goals_conceded_avg, win_rate}}"""
    records = {}
    history = {}  # team -> list of (goals_scored, goals_conceded, won)

    for _, row in df.iterrows():
        team  = row[team_col]
        gs    = row[score_col]
        gc    = row[opp_score_col]
        won   = 1 if gs > gc else 0

        past = history.get(team, [])
        if len(past) >= 3:                         # need at least 3 past games
            window_data = past[-window:]
            records[(row["date"], team)] = {
                "goals_scored_avg"   : np.mean([x[0] for x in window_data]),
                "goals_conceded_avg" : np.mean([x[1] for x in window_data]),
                "win_rate"           : np.mean([x[2] for x in window_data]),
                "n_games"            : len(window_data),
            }
        history.setdefault(team, []).append((gs, gc, won))

    return records

home_stats = team_rolling_stats(results_sorted, "home_team", "home_score", "away_score")
away_stats = team_rolling_stats(results_sorted, "away_team", "away_score", "home_score")

# ── 5. Build head-to-head win rate ────────────────────────────────────────────
def h2h_stats(df):
    """Return dict: {(date, home_team, away_team): home_h2h_win_rate}"""
    records = {}
    history = {}  # (teamA, teamB) -> list of outcomes from teamA's perspective

    for _, row in df.iterrows():
        h, a  = row["home_team"], row["away_team"]
        key   = (h, a)
        rkey  = (a, h)

        past = history.get(key, []) + [1 - x for x in history.get(rkey, [])]
        if len(past) >= 2:
            records[(row["date"], h, a)] = np.mean(past[-10:])

        outcome = 1 if row["home_score"] > row["away_score"] else 0
        history.setdefault(key, []).append(outcome)

    return records

h2h = h2h_stats(results_sorted)

# ── 6. Build goal features from goalscorers (penalty / own-goal rates) ────────
goalscorers["date"] = pd.to_datetime(goalscorers["date"])
pen_rate = (
    goalscorers.groupby(["date", "home_team", "away_team", "team"])["penalty"]
    .mean()
    .reset_index()
    .rename(columns={"penalty": "pen_rate"})
)

# ── 7. Assemble feature matrix ────────────────────────────────────────────────
rows = []

for _, row in results_sorted.iterrows():
    d, h, a = row["date"], row["home_team"], row["away_team"]

    hs = home_stats.get((d, h))
    as_ = away_stats.get((d, a))
    if hs is None or as_ is None:
        continue                                   # skip if no history yet

    feat = {
        # home rolling stats
        "home_goals_scored_avg"   : hs["goals_scored_avg"],
        "home_goals_conceded_avg" : hs["goals_conceded_avg"],
        "home_win_rate"           : hs["win_rate"],
        # away rolling stats
        "away_goals_scored_avg"   : as_["goals_scored_avg"],
        "away_goals_conceded_avg" : as_["goals_conceded_avg"],
        "away_win_rate"           : as_["win_rate"],
        # head-to-head
        "h2h_home_win_rate"       : h2h.get((d, h, a), 0.5),
        # match context
        "is_neutral"              : int(row["neutral"]),
        "is_world_cup"            : int("FIFA World Cup" in row["tournament"]),
        "is_wc_qual"              : int("qualification" in row["tournament"].lower()),
        # goal difference proxy
        "form_diff"               : hs["goals_scored_avg"] - hs["goals_conceded_avg"]
                                  - (as_["goals_scored_avg"] - as_["goals_conceded_avg"]),
        # target
        "outcome"                 : row["outcome"],
        # metadata (not used as features)
        "_date"                   : d,
        "_home_team"              : h,
        "_away_team"              : a,
    }
    rows.append(feat)

df_feat = pd.DataFrame(rows)
print(f"Feature matrix shape: {df_feat.shape}")
print(f"Outcome distribution:\n{df_feat['outcome'].value_counts()}")

# ── 8. Train / test split (chronological) ────────────────────────────────────
FEATURE_COLS = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_win_rate",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_win_rate",
    "h2h_home_win_rate", "is_neutral", "is_world_cup", "is_wc_qual", "form_diff",
]

df_feat = df_feat.sort_values("_date")
split   = int(len(df_feat) * 0.8)
train   = df_feat.iloc[:split]
test    = df_feat.iloc[split:]

X_train = train[FEATURE_COLS]
y_train = train["outcome"]
X_test  = test[FEATURE_COLS]
y_test  = test["outcome"]

# ── 9. Train model ────────────────────────────────────────────────────────────
print("\nTraining Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
print(f"\nTest accuracy: {acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, preds))

# ── 10. Save artefacts ────────────────────────────────────────────────────────
joblib.dump(model, "wc_predictor_model.pkl")
print("\n✅  Model saved → wc_predictor_model.pkl")

# Save per-team latest stats so the app can look them up quickly
team_latest = {}
for _, row in results_sorted.iterrows():
    d, h, a = row["date"], row["home_team"], row["away_team"]
    hs  = home_stats.get((d, h))
    as_ = away_stats.get((d, a))
    if hs:
        team_latest[h] = {**hs, "last_seen": str(d)}
    if as_:
        team_latest[a] = {**as_, "last_seen": str(d)}

import json
with open("team_stats.json", "w") as f:
    json.dump(team_latest, f)
print("✅  Team stats saved → team_stats.json")

# Save sorted unique team list for the UI dropdowns
all_teams = sorted(
    set(results["home_team"].tolist() + results["away_team"].tolist())
)
with open("teams.json", "w") as f:
    json.dump(all_teams, f)
print("✅  Team list saved → teams.json")

# Save h2h lookup
h2h_serialisable = {f"{d}|{h}|{a}": v for (d, h, a), v in h2h.items()}
with open("h2h_stats.json", "w") as f:
    json.dump(h2h_serialisable, f, default=str)
print("✅  H2H stats saved → h2h_stats.json")

print("\nAll done! Now run:  streamlit run app.py")
