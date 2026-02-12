import pandas as pd
from collections import defaultdict, deque


def compute_recent_form(matches: pd.DataFrame, window: int = 5):
    """
    Computes rolling recent win-rate for each team BEFORE each match.
    """

    history = defaultdict(lambda: deque(maxlen=window))
    records = []

    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]

        # recent win rate BEFORE match
        t1_rate = sum(history[t1]) / len(history[t1]) if history[t1] else 0.5
        t2_rate = sum(history[t2]) / len(history[t2]) if history[t2] else 0.5

        records.append({
            "match_id": row["id"],
            "team1_recent_win_rate_5": t1_rate,
            "team2_recent_win_rate_5": t2_rate
        })

        # update history AFTER match
        if row["winner"] == t1:
            history[t1].append(1)
            history[t2].append(0)
        elif row["winner"] == t2:
            history[t1].append(0)
            history[t2].append(1)
        else:
            # no result / tie
            history[t1].append(0.5)
            history[t2].append(0.5)

    return pd.DataFrame(records)
