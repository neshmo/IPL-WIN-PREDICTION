import lightgbm as lgb
import joblib
from sklearn.metrics import log_loss

def train_live(df):
    features = [
        "runs_remaining",
        "balls_remaining",
        "wickets_in_hand",
        "current_run_rate",
        "required_run_rate"
    ]

    X = df[features]
    y = df["target"]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        objective="binary"
    )

    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]

    print("Live Model LogLoss:", log_loss(y, preds))
    joblib.dump(model, "models/live_model.pkl")
