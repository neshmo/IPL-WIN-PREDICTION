from collections import defaultdict

INITIAL_ELO = 1500
K_FACTOR = 20

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, result):
    exp_a = expected_score(elo_a, elo_b)
    elo_a_new = elo_a + K_FACTOR * (result - exp_a)
    elo_b_new = elo_b + K_FACTOR * ((1 - result) - (1 - exp_a))
    return elo_a_new, elo_b_new

def compute_elo(matches_df):
    elo = defaultdict(lambda: INITIAL_ELO)
    elo_features = []

    for _, row in matches_df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        elo_t1, elo_t2 = elo[t1], elo[t2]

        elo_features.append({
            "match_id": row["id"],
            "team1_elo": elo_t1,
            "team2_elo": elo_t2,
            "elo_diff": elo_t1 - elo_t2
        })

        if row["winner"] == t1:
            result = 1
        elif row["winner"] == t2:
            result = 0
        else:
            continue  # no result

        elo[t1], elo[t2] = update_elo(elo_t1, elo_t2, result)

    return elo_features
