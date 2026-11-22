"""Shared data loading and feature engineering for training and prediction.

Every feature only uses games *before* the row's game (via shift(1)), so the
same function works for historical rows and for upcoming games whose stats
are still unknown (target = NaN).
"""
import pandas as pd

TARGET = "fantasy_points_ppr"
POSITIONS = ["QB", "RB", "WR", "TE"]
POS_COLS = [f"is_{p}" for p in POSITIONS]

FEATURES = [
    f"{TARGET}_roll3",
    "targets_roll3",
    "carries_roll3",
    "ppr_career_avg",
    "games_played",
    "opp_allowed_avg",
] + POS_COLS


def load_weekly(path="data/weekly.csv"):
    """Load weekly stats, keeping regular-season skill-position rows."""
    df = pd.read_csv(path, low_memory=False)
    return df[
        (df["season_type"] == "REG")
        & (df["position"].isin(POSITIONS))
    ].copy()


def add_features(df):
    """Return a copy of df with all FEATURES columns added."""
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby("player_id")

    # Recent form: average of the previous 3 games
    for col in [TARGET, "targets", "carries"]:
        df[f"{col}_roll3"] = g[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    df["ppr_career_avg"] = g[TARGET].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["games_played"] = g.cumcount()

    # Opponent strength: avg PPR points this defense allowed to this position
    # over its previous 5 games
    opp = (
        df.groupby(["opponent_team", "position", "season", "week"])[TARGET]
        .mean()
        .reset_index()
        .sort_values(["opponent_team", "position", "season", "week"])
    )
    opp["opp_allowed_avg"] = opp.groupby(["opponent_team", "position"])[TARGET].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df = df.merge(
        opp.drop(columns=TARGET),
        on=["opponent_team", "position", "season", "week"],
    )

    # One-hot encode position
    for p, col in zip(POSITIONS, POS_COLS):
        df[col] = (df["position"] == p).astype(int)

    return df
