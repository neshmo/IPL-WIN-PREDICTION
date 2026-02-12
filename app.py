import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ----------------------------------
# Paths
# ----------------------------------
MODEL_PATH = "models/prematch_model.pkl"
DATA_PATH = "data/processed/match_features.parquet"

# ----------------------------------
# Load model & data
# ----------------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_parquet(DATA_PATH)

# ----------------------------------
# Feature list (MUST match training)
# ----------------------------------
FEATURES = [
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "team1_recent_win_rate_5",
    "team2_recent_win_rate_5",
    "chasing_win_rate",
    "team1_pp_run_rate",
    "team2_pp_run_rate",
    "team1_middle_wicket_rate",
    "team2_middle_wicket_rate",
    "team1_death_run_rate",
    "team2_death_run_rate",
]

# ----------------------------------
# Active IPL Teams (ONLY)
# ----------------------------------
ACTIVE_TEAMS = {
    "Chennai Super Kings",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Punjab Kings",
    "Delhi Capitals",
    "Lucknow Super Giants",
    "Gujarat Titans",
    "Royal Challengers Bangalore"
}

all_teams = set(df["team1"].unique()) | set(df["team2"].unique())
teams = sorted(all_teams.intersection(ACTIVE_TEAMS))

# ----------------------------------
# Helper: latest stats for a team
# ----------------------------------
def get_latest_team_stats(team_name, prefix):
    team_matches = df[
        (df["team1"] == team_name) | (df["team2"] == team_name)
    ].sort_values("id")

    if team_matches.empty:
        return None

    last = team_matches.iloc[-1]

    stats = {}
    for col in FEATURES:
        if col.startswith(prefix):
            stats[col] = last[col]

    return stats

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(
    page_title="IPL Winner Predictor",
    layout="centered"
)

st.title("🏏 IPL Match Winner Predictor")
st.caption("Explainable ML model using form, phase-wise performance & ELO")

st.markdown("---")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

if team1 == team2:
    st.warning("Please select two different teams.")
    st.stop()

if st.button("Predict Winner"):
    t1_stats = get_latest_team_stats(team1, "team1")
    t2_stats = get_latest_team_stats(team2, "team2")

    if t1_stats is None or t2_stats is None:
        st.error("Not enough historical data for selected teams.")
        st.stop()

    # ----------------------------------
    # Build input row
    # ----------------------------------
    input_data = {}

    for f in FEATURES:
        if f.startswith("team1"):
            input_data[f] = t1_stats.get(f, 0)
        elif f.startswith("team2"):
            input_data[f] = t2_stats.get(f, 0)

    # Derived features
    input_data["elo_diff"] = (
        input_data["team1_elo"] - input_data["team2_elo"]
    )

    input_data["chasing_win_rate"] = df["chasing_win_rate"].mean()

    X = pd.DataFrame([input_data])[FEATURES]

    # ----------------------------------
    # Prediction
    # ----------------------------------
    prob_team1 = model.predict_proba(X)[0][1]
    prob_team2 = 1 - prob_team1

    winner = team1 if prob_team1 >= 0.5 else team2
    confidence = max(prob_team1, prob_team2)

    # ----------------------------------
    # Display Results
    # ----------------------------------
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric("Predicted Winner", winner)

    st.progress(int(confidence * 100))

    col1, col2 = st.columns(2)
    col1.metric(team1, f"{prob_team1:.2%}")
    col2.metric(team2, f"{prob_team2:.2%}")

    if confidence >= 0.65:
        st.success("High confidence prediction")
    elif confidence >= 0.55:
        st.info("Medium confidence prediction")
    else:
        st.warning("Low confidence prediction")

    st.caption(
        "Prediction is based on historical performance, recent form, and phase-wise strength."
    )
