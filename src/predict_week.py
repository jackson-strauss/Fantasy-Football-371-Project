"""Project PPR fantasy points for every active QB/RB/WR/TE in a given week.

Usage (from the project root):
    python src/predict_week.py                       # next unplayed week
    python src/predict_week.py --season 2026 --week 12
    python src/predict_week.py --position WR --top 30

Only games before the requested week are used to build features. If that
week has already been played, actual points are shown next to projections.
"""
import argparse
from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd

from features import TARGET, add_features, load_weekly
from injuries import add_availability

ACTIVE_WINDOW = 4  # player must have played within this many weeks to be projected


def get_matchups(season, week):
    """Return one row per team playing that week: team, opponent_team."""
    sched = nfl.load_schedules([season]).to_pandas()
    sched = sched[(sched["game_type"] == "REG") & (sched["week"] == week)]
    home = sched.rename(columns={"home_team": "team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent_team"})
    return pd.concat([home, away])[["team", "opponent_team"]]


# Columns returned alongside player/team/opponent. The feature columns are kept
# so the assistant can explain each projection.
OUTPUT_COLS = [
    "projected", "floor", "median", "ceiling", "risk",
    "matchup_effect", "form_effect", "opp_rank", "opp_count",
    f"{TARGET}_roll3", "targets_roll3", "carries_roll3", "ppr_career_avg",
    "ppr_std6", "games_played", "opp_allowed_avg", "player_id",
]


# Roughly how many players per position are startable in a 12-team league;
# risk labels are calibrated against this group
RELEVANT_PER_POSITION = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}


def add_projection_details(rows, model, q_models, features):
    """Add projection, outcome range, risk label, and explanation columns."""
    rows = rows.copy()
    X = rows[features]
    rows["projected"] = model.predict(X)

    # Range of outcomes from the quantile models (10th / 50th / 90th percentile)
    q = np.sort(np.vstack([q_models[k].predict(X) for k in sorted(q_models)]), axis=0)
    rows["floor"] = np.minimum(q[0], rows["projected"])
    rows["median"] = q[1]
    rows["ceiling"] = np.maximum(q[2], rows["projected"])

    # How much the matchup moves the projection: re-predict against a
    # league-median defense for the position
    neutral = X.copy()
    neutral["opp_allowed_avg"] = rows.groupby("position")["opp_allowed_avg"].transform("median")
    rows["matchup_effect"] = rows["projected"] - model.predict(neutral)

    # How much recent form moves it: re-predict with last-3 average = career average
    typical = X.copy()
    typical[f"{TARGET}_roll3"] = rows["ppr_career_avg"]
    rows["form_effect"] = rows["projected"] - model.predict(typical)

    # Rank each defense by points allowed to the position (1 = most allowed)
    opp = rows.drop_duplicates(["position", "opponent_team"])[
        ["position", "opponent_team", "opp_allowed_avg"]].copy()
    opp["opp_rank"] = opp.groupby("position")["opp_allowed_avg"].rank(
        ascending=False, method="min").astype(int)
    opp["opp_count"] = opp.groupby("position")["opponent_team"].transform("count")
    rows = rows.merge(opp.drop(columns="opp_allowed_avg"),
                      on=["position", "opponent_team"], how="left")

    # Risk label: is this player's range of outcomes wider or narrower than is
    # typical for a similar projection? "Typical" is a line fit through the
    # fantasy-relevant players at the position this week.
    spread = rows["ceiling"] - rows["floor"]
    rows["risk"] = "Moderate"
    for pos, idx in rows.groupby("position").groups.items():
        relevant = rows.loc[idx, "projected"].nlargest(
            RELEVANT_PER_POSITION.get(pos, 30)).index
        if len(relevant) < 6:
            continue
        slope, intercept = np.polyfit(rows.loc[relevant, "projected"], spread.loc[relevant], 1)
        excess = spread.loc[idx] - (slope * rows.loc[idx, "projected"] + intercept)
        lo, hi = excess.loc[relevant].quantile([1 / 3, 2 / 3])
        rows.loc[idx, "risk"] = np.select([excess <= lo, excess > hi],
                                          ["Steady", "Boom/bust"], "Moderate")

    for col in ["projected", "floor", "median", "ceiling", "matchup_effect", "form_effect"]:
        rows[col] = rows[col].round(1)
    return rows


def predict_week(season=None, week=None, data_path="data/weekly.csv",
                 model_path="models/model.joblib"):
    bundle = joblib.load(model_path)
    model, features = bundle["model"], bundle["features"]

    df = load_weekly(data_path)
    team_col = "team" if "team" in df.columns else "recent_team"

    if season is None:
        season = int(df["season"].max())
    if week is None:
        played = df.loc[df["season"] == season, "week"]
        week = int(played.max()) + 1 if not played.empty else 1

    # Only use games strictly before the target week
    before = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    history = df[before]
    actual = df[(df["season"] == season) & (df["week"] == week)][["player_id", TARGET]]

    # Each player's most recent game tells us their current team and position
    last = (
        history.sort_values(["season", "week"])
        .groupby("player_id")
        .tail(1)
    )
    if week > 1:
        active = (last["season"] == season) & (last["week"] >= week - ACTIVE_WINDOW)
    else:
        active = last["season"] == season - 1
    last = last[active]

    matchups = get_matchups(season, week)
    if matchups.empty:
        raise SystemExit(f"No regular-season games found for {season} week {week}.")

    # Build placeholder rows for the upcoming game (stats unknown -> NaN).
    # Players whose team is on bye drop out in this merge.
    upcoming = (
        last[["player_id", "player_display_name", "position", team_col]]
        .rename(columns={team_col: "team"})
        .merge(matchups, on="team")
        .rename(columns={"team": team_col})
        .assign(season=season, week=week, season_type="REG")
    )

    combined = pd.concat([history, upcoming], ignore_index=True)
    feats = add_features(combined)
    rows = feats[(feats["season"] == season) & (feats["week"] == week)].copy()

    q_models = bundle.get("quantile_models")
    if q_models is None:
        raise SystemExit("models/model.joblib is out of date. Re-run: python src/train_model.py")

    rows = add_projection_details(rows, model, q_models, features)
    out = rows[["player_display_name", "position", team_col, "opponent_team"]
               + OUTPUT_COLS].rename(columns={team_col: "team"})
    out = out.merge(actual.rename(columns={TARGET: "actual"}), on="player_id", how="left")
    out = add_availability(out, season, week)
    out = out.sort_values("projected", ascending=False).reset_index(drop=True)

    if season <= bundle["trained_through_season"]:
        print(f"Note: the model was trained on data through {bundle['trained_through_season']}, "
              f"so projections for {season} are not a fair test.")
    return out, season, week


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--position", choices=["QB", "RB", "WR", "TE"])
    parser.add_argument("--top", type=int, default=25, help="rows to print per position")
    parser.add_argument("--show-injured", action="store_true",
                        help="include players ruled out / on IR in the printed rankings")
    args = parser.parse_args()

    out, season, week = predict_week(args.season, args.week)

    Path("data").mkdir(exist_ok=True)
    out_path = f"data/predictions_{season}_wk{week}.csv"
    out.to_csv(out_path, index=False)

    show_actual = out["actual"].notna().any()
    cols = ["player_display_name", "team", "opponent_team", "projected", "floor", "ceiling",
            "risk", "injury_status"]
    if show_actual:
        cols.append("actual")

    shown = out if args.show_injured else out[out["available"]]
    positions = [args.position] if args.position else ["QB", "RB", "WR", "TE"]
    print(f"\nProjected PPR points - {season} week {week}")
    for p in positions:
        print(f"\n== {p} ==")
        print(shown[shown["position"] == p][cols].head(args.top).to_string(index=False))

    hidden = (~out["available"]).sum()
    if hidden and not args.show_injured:
        print(f"\n{hidden} players ruled out / on IR hidden (use --show-injured to list them)")
    print(f"Saved all {len(out)} projections to {out_path}")


if __name__ == "__main__":
    main()
