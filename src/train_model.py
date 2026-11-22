from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from features import FEATURES, POSITIONS, TARGET, add_features, load_weekly

TEST_SEASON = 2025  # train on seasons before, test on this one

#Load, clean, and build features#
df = add_features(load_weekly())
df = df.dropna(subset=FEATURES + [TARGET])

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
model.fit(train[FEATURES], train[TARGET])
preds = model.predict(test[FEATURES])
model_mae = mean_absolute_error(test[TARGET], preds)

print(f"Baseline MAE (rolling avg): {baseline_mae:.2f}")
print(f"Model MAE:                  {model_mae:.2f}")

#Per-position evaluation#
results = test[["position", TARGET, f"{TARGET}_roll3"]].copy()
results["pred"] = preds

print(f"\n{'Pos':<4} {'Rows':>6} {'Baseline':>9} {'Model':>7} {'Improve':>8}")
for p in POSITIONS:
    r = results[results["position"] == p]
    if r.empty:
        continue
    b = mean_absolute_error(r[TARGET], r[f"{TARGET}_roll3"])
    m = mean_absolute_error(r[TARGET], r["pred"])
    print(f"{p:<4} {len(r):>6} {b:>9.2f} {m:>7.2f} {b - m:>+8.2f}")

#Save model#
# Refit on all seasons up through TEST_SEASON so the saved model has the most data
final_model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, random_state=42
)
final_df = df[df["season"] <= TEST_SEASON]
final_model.fit(final_df[FEATURES], final_df[TARGET])

Path("models").mkdir(exist_ok=True)
joblib.dump(
    {
        "model": final_model,
        "features": FEATURES,
        "target": TARGET,
        "positions": POSITIONS,
        "trained_through_season": TEST_SEASON,
        "test_mae": model_mae,
    },
    "models/model.joblib",
)
print(f"\nSaved models/model.joblib (trained on seasons through {TEST_SEASON})")
