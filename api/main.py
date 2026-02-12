from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/prematch_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]

    return {
        "team1_win_probability": round(float(prob), 3),
        "team2_win_probability": round(1 - float(prob), 3)
    }
