import joblib
import pandas as pd

model = joblib.load("models/live_model.pkl")

def predict_live(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]

    return {
        "batting_team_win_probability": round(float(prob), 3),
        "bowling_team_win_probability": round(1 - float(prob), 3)
    }
