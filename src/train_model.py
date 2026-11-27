from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from features import FEATURES, POSITIONS, REQUIRED_FEATURES, TARGET, add_features, load_weekly

TEST_SEASON = 2025  # train on seasons before, test on this one

# Quantile models give a range of outcomes: floor (10th pct), median, ceiling (90th pct)
QUANTILES = [0.1, 0.5, 0.9]


def make_model(quantile=None):
    # early_stopping is off: sklearn otherwise holds out a random 10% split and
    # can stop at a bad point (it produced floors of 0 for many star players).
    # A fixed 300 rounds is reproducible and slightly more accurate here.
    if quantile is None:
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=42, early_stopping=False,
        )
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile,
        max_iter=300, learning_rate=0.05, random_state=42, early_stopping=False,
    )


def fit_all(data):
    """Fit the mean model and one model per quantile."""
    mean_model = make_model().fit(data[FEATURES], data[TARGET])
    q_models = {q: make_model(q).fit(data[FEATURES], data[TARGET]) for q in QUANTILES}
    return mean_model, q_models


def predict_range(q_models, X):
    """Floor/median/ceiling predictions, sorted so they never cross."""
    preds = np.vstack([q_models[q].predict(X) for q in QUANTILES])
    return np.sort(preds, axis=0)


#Load, clean, and build features#
df = add_features(load_weekly())
df = df.dropna(subset=REQUIRED_FEATURES + [TARGET])

#Time-based split, cant check what they actualy did with the prediction#
train = df[df["season"] < TEST_SEASON]
test = df[df["season"] == TEST_SEASON]
print(f"Train rows: {len(train)}, Test rows: {len(test)}")

#Baseline of 3 game avg#
baseline_mae = mean_absolute_error(test[TARGET], test[f"{TARGET}_roll3"])

#Train models#
model, q_models = fit_all(train)
preds = model.predict(test[FEATURES])
floor, median, ceiling = predict_range(q_models, test[FEATURES])
model_mae = mean_absolute_error(test[TARGET], preds)

print(f"Baseline MAE (rolling avg): {baseline_mae:.2f}")
print(f"Model MAE:                  {model_mae:.2f}")

#Per-position evaluation#
results = test[["position", TARGET, f"{TARGET}_roll3"]].copy()
results["pred"] = preds
results["floor"], results["ceiling"] = floor, ceiling
results["covered"] = (results[TARGET] >= floor) & (results[TARGET] <= ceiling)

print(f"\n{'Pos':<4} {'Rows':>6} {'Baseline':>9} {'Model':>7} {'Improve':>8}"
      f" {'80% range hit':>14} {'Avg range':>10}")
for p in POSITIONS + ["All"]:
    r = results if p == "All" else results[results["position"] == p]
    if r.empty:
        continue
    b = mean_absolute_error(r[TARGET], r[f"{TARGET}_roll3"])
    m = mean_absolute_error(r[TARGET], r["pred"])
    width = (r["ceiling"] - r["floor"]).mean()
    print(f"{p:<4} {len(r):>6} {b:>9.2f} {m:>7.2f} {b - m:>+8.2f}"
          f" {r['covered'].mean():>14.1%} {width:>10.1f}")
print("(80% range hit = share of games where actual points landed between floor and"
      " ceiling; ideally about 80%)")

coverage = results["covered"].mean()

#Save model#
# Refit on all seasons up through TEST_SEASON so the saved models have the most data
final_df = df[df["season"] <= TEST_SEASON]
final_model, final_q_models = fit_all(final_df)

Path("models").mkdir(exist_ok=True)
joblib.dump(
    {
        "model": final_model,
        "quantile_models": final_q_models,
        "features": FEATURES,
        "target": TARGET,
        "positions": POSITIONS,
        "trained_through_season": TEST_SEASON,
        "test_mae": model_mae,
        "test_range_coverage": coverage,
    },
    "models/model.joblib",
)
print(f"\nSaved models/model.joblib (trained on seasons through {TEST_SEASON})")
