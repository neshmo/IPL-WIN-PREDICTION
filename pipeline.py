import pandas as pd
from features.feature_builder import build_match_features
from models.train_prematch import train as train_prematch

def run():
    matches = pd.read_csv("C:/Users/wmsys/OneDrive/Pictures/ipl_win_predictor/data/raw/matches.csv")
    deliveries = pd.read_csv("C:/Users/wmsys/OneDrive/Pictures/ipl_win_predictor/data/raw/deliveries.csv")

    df = build_match_features(matches, deliveries)
    train_prematch(df)

if __name__ == "__main__":
    run()
