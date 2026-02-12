from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss
)

def evaluate(y_true, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_prob > 0.5),
        "log_loss": log_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob)
    }
