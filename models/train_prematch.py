import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import joblib

def train(df):
    features = [
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



    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    print("Log Loss:", log_loss(y_test, preds))
    joblib.dump(model, "models/prematch_model.pkl")
