import os
import pandas as pd

from features.elo import compute_elo
from features.venue_stats import compute_venue_stats
from features.phase_stats import compute_team_phase_stats
from features.recent_form import compute_recent_form


def build_match_features(matches: pd.DataFrame, deliveries: pd.DataFrame):

    # -------------------------------------------------
    # SORT (NO LEAKAGE)
    # -------------------------------------------------
    matches = matches.sort_values("id").reset_index(drop=True)

    # -------------------------------------------------
    # ELO FEATURES
    # -------------------------------------------------
    elo_df = pd.DataFrame(compute_elo(matches))
    matches = matches.merge(elo_df, left_on="id", right_on="match_id", how="left")
    matches.drop(columns=["match_id"], inplace=True)

    # -------------------------------------------------
    # RECENT FORM FEATURES
    # -------------------------------------------------
    recent_df = compute_recent_form(matches, window=5)
    matches = matches.merge(recent_df, left_on="id", right_on="match_id", how="left")
    matches.drop(columns=["match_id"], inplace=True)

    # -------------------------------------------------
    # VENUE FEATURES
    # -------------------------------------------------
    venue_df = compute_venue_stats(matches)
    matches = matches.merge(venue_df, on="venue", how="left")

    # -------------------------------------------------
    # PHASE-WISE FEATURES
    # -------------------------------------------------
    phase_df = compute_team_phase_stats(deliveries)

    phase_pivot = (
        phase_df
        .pivot_table(
            index=["match_id", "batting_team"],
            columns="phase",
            values=["run_rate", "wickets"],
            aggfunc="mean"
        )
        .reset_index()
    )

    # flatten columns
    phase_pivot.columns = [
        f"{a}_{b}" if b else a
        for a, b in phase_pivot.columns
    ]

    # ---------- TEAM 1 ----------
    t1_phase = phase_pivot.rename(columns={
        "run_rate_powerplay": "team1_pp_run_rate",
        "wickets_middle": "team1_middle_wicket_rate",
        "run_rate_death": "team1_death_run_rate"
    })

    matches = matches.merge(
        t1_phase,
        left_on=["id", "team1"],
        right_on=["match_id", "batting_team"],
        how="left"
    )

    matches.drop(columns=["match_id", "batting_team"], inplace=True)

    # ---------- TEAM 2 ----------
    t2_phase = phase_pivot.rename(columns={
        "run_rate_powerplay": "team2_pp_run_rate",
        "wickets_middle": "team2_middle_wicket_rate",
        "run_rate_death": "team2_death_run_rate"
    })

    matches = matches.merge(
        t2_phase,
        left_on=["id", "team2"],
        right_on=["match_id", "batting_team"],
        how="left"
    )

    matches.drop(columns=["match_id", "batting_team"], inplace=True)

    # -------------------------------------------------
    # TARGET
    # -------------------------------------------------
    matches["target"] = (matches["winner"] == matches["team1"]).astype(int)

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------
    matches.fillna(0, inplace=True)

    object_cols = matches.select_dtypes(include=["object"]).columns
    for col in object_cols:
        matches[col] = matches[col].astype(str)

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)
    matches.to_parquet("data/processed/match_features.parquet", index=False)

    return matches
