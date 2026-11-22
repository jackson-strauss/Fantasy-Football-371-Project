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
import pandas as pd

from features import TARGET, add_features, load_weekly

ACTIVE_WINDOW = 4  # player must have played within this many weeks to be projected


def get_matchups(season, week):
    """Return one row per team playing that week: team, opponent_team."""
    sched = nfl.load_schedules([season]).to_pandas()
    sched = sched[(sched["game_type"] == "REG") & (sched["week"] == week)]
    home = sched.rename(columns={"home_team": "team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent_team"})
    return pd.concat([home, away])[["team", "opponent_team"]]


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

    rows["projected"] = model.predict(rows[features]).round(1)
    out = rows[["player_display_name", "position", team_col, "opponent_team",
                "projected", "player_id"]].rename(columns={team_col: "team"})
    out = out.merge(actual.rename(columns={TARGET: "actual"}), on="player_id", how="left")
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
    args = parser.parse_args()

    out, season, week = predict_week(args.season, args.week)

    Path("data").mkdir(exist_ok=True)
    out_path = f"data/predictions_{season}_wk{week}.csv"
    out.to_csv(out_path, index=False)

    show_actual = out["actual"].notna().any()
    cols = ["player_display_name", "team", "opponent_team", "projected"]
    if show_actual:
        cols.append("actual")

    positions = [args.position] if args.position else ["QB", "RB", "WR", "TE"]
    print(f"\nProjected PPR points - {season} week {week}")
    for p in positions:
        print(f"\n== {p} ==")
        print(out[out["position"] == p][cols].head(args.top).to_string(index=False))

    print(f"\nSaved all {len(out)} projections to {out_path}")


if __name__ == "__main__":
    main()
