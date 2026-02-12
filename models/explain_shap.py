import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


def run_shap():
    # -----------------------------
    # Load model and data
    # -----------------------------
    model = joblib.load("models/prematch_model.pkl")

    df = pd.read_parquet("data/processed/match_features.parquet")

    # Same features used in training
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

    # -----------------------------
    # SHAP EXPLAINER
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # -----------------------------
    # GLOBAL EXPLANATION
    # -----------------------------
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_global_importance.png")
    plt.close()

    # -----------------------------
    # LOCAL EXPLANATION (1 MATCH)
    # -----------------------------
    idx = 0  # change this to inspect different matches

    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_single_match.png")
    plt.close()


if __name__ == "__main__":
    run_shap()
