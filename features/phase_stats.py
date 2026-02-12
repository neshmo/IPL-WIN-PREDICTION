import pandas as pd

def assign_phase(over):
    if over <= 6:
        return "powerplay"
    elif over <= 15:
        return "middle"
    else:
        return "death"


def compute_team_phase_stats(deliveries: pd.DataFrame):
    """
    Computes historical phase-wise stats per team per match.
    """

    df = deliveries.copy()
    df["phase"] = df["over"].apply(assign_phase)

    agg = (
        df.groupby(["match_id", "batting_team", "phase"])
        .agg(
            runs=("total_runs", "sum"),
            balls=("ball", "count"),
            wickets=("is_wicket", "sum")
        )
        .reset_index()
    )

    agg["run_rate"] = agg["runs"] / (agg["balls"] / 6)

    return agg
