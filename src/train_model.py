import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

TEST_SEASON = 2025  # train on seasons before, test on this one
TARGET = "fantasy_points_ppr"
POSITIONS = ["QB", "RB", "WR", "TE"]


#Load and clean#
df = pd.read_csv("data/weekly.csv", low_memory=False)
df = df[
    (df["season_type"] == "REG")
    & (df["position"].isin(POSITIONS))
].copy()
df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

#Features#
g = df.groupby("player_id")

for col in [TARGET, "targets", "carries"]:
    df[f"{col}_roll3"] = g[col].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

df["ppr_career_avg"] = g[TARGET].transform(
    lambda s: s.shift(1).expanding().mean()
)
df["games_played"] = g.cumcount()

# Opponent strength: avg PPR points this defense allowed to this position recently
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

# One-hot encode position into explicitly named numeric columns
pos_cols = []
for p in POSITIONS:
    col = f"is_{p}"
    df[col] = (df["position"] == p).astype(int)
    pos_cols.append(col)

features = [
    f"{TARGET}_roll3",
    "targets_roll3",
    "carries_roll3",
    "ppr_career_avg",
    "games_played",
    "opp_allowed_avg",
] + pos_cols

df = df.dropna(subset=features + [TARGET])

#Time-based split, cant check what they actualy did with the prediction#
train = df[df["season"] < TEST_SEASON]
test = df[df["season"] == TEST_SEASON]
print(f"Train rows: {len(train)}, Test rows: {len(test)}")

#Baseline of 3 game avg#
baseline_mae = mean_absolute_error(test[TARGET], test[f"{TARGET}_roll3"])

#Train model#
model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, random_state=42
)
model.fit(train[features], train[TARGET])
preds = model.predict(test[features])
model_mae = mean_absolute_error(test[TARGET], preds)

print(f"Baseline MAE (rolling avg): {baseline_mae:.2f}")
print(f"Model MAE:                  {model_mae:.2f}")