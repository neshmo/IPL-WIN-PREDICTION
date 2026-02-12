import joblib
import pandas as pd

model = joblib.load("models/prematch_model.pkl")

def predict_prematch(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]

    return {
        "team1_win_probability": round(float(prob), 3),
        "team2_win_probability": round(1 - float(prob), 3)
    }
