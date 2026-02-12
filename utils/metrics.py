import numpy as np

def calibration_curve(y_true, y_prob, bins=10):
    bins = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    results = []
    for i in range(len(bins) - 1):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        results.append({
            "bin": i,
            "predicted_prob": y_prob[mask].mean(),
            "actual_win_rate": y_true[mask].mean()
        })
    return results
