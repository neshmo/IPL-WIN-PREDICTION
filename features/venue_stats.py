def compute_venue_stats(matches):
    """
    Computes venue-level chasing bias WITHOUT relying on win_by_wickets.
    """

    df = matches.copy()

    # A team is chasing if toss_decision == "field"
    df["is_chasing"] = df["toss_decision"].str.lower().eq("field")

    # Chasing win happens if the chasing team is the winner
    df["chasing_win"] = (
        (df["is_chasing"]) &
        (df["toss_winner"] == df["winner"])
    )

    venue_stats = (
        df.groupby("venue")
        .agg(
            matches_played=("id", "count"),
            chasing_wins=("chasing_win", "sum")
        )
        .reset_index()
    )

    venue_stats["chasing_win_rate"] = (
        venue_stats["chasing_wins"] / venue_stats["matches_played"]
    )

    return venue_stats
