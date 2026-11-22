"""Player availability from NFL injury reports and weekly roster status.

Two sources are combined, because neither is complete on its own:
  * Weekly injury reports (Out / Doubtful / Questionable). Players on injured
    reserve are NOT listed here.
  * Weekly rosters, whose status column marks injured reserve (RES), practice
    squad (DEV), released (CUT), etc.

For each source we use the most recent report at or before the target week,
so a week whose report isn't published yet falls back to the previous one.
"""
import warnings

import nflreadpy as nfl
import pandas as pd

# Injury-report designations that mean the player is not expected to play
OUT_REPORT_STATUSES = {"Out", "Doubtful"}

# Weekly-roster statuses that mean the player is not available to play
INACTIVE_ROSTER_STATUSES = {
    "RES": "Injured reserve",
    "DEV": "Practice squad",
    "CUT": "Released",
    "RET": "Retired",
    "EXE": "Exempt",
    "SUS": "Suspended",
}


def _latest_report(inj, week):
    """Rows from each team's most recent injury report at or before `week`."""
    inj = inj[(inj["game_type"] == "REG") & (inj["week"] <= week)]
    if inj.empty:
        return inj
    latest_week = inj.groupby("team")["week"].transform("max")
    return inj[inj["week"] == latest_week]


def get_availability(season, week):
    """Return DataFrame[player_id, injury_status, available] for players with an issue.

    Players not in the result have no reported issue and are available.
    """
    frames = []

    try:
        inj = nfl.load_injuries([season]).to_pandas()
        rep = _latest_report(inj, week)
        rep = rep[rep["report_status"].notna()]
        if not rep.empty:
            frames.append(pd.DataFrame({
                "player_id": rep["gsis_id"],
                "injury_status": rep["report_status"] + rep["week"].map(
                    lambda w: "" if w == week else f" (wk {w} report)"),
                "available": ~rep["report_status"].isin(OUT_REPORT_STATUSES),
                "_priority": 1,
            }))
    except Exception as e:  # data not published yet, network error, etc.
        warnings.warn(f"Could not load injury reports for {season}: {e}")

    try:
        ros = nfl.load_rosters_weekly([season]).to_pandas()
        ros = ros[(ros["game_type"] == "REG") & (ros["week"] <= week)]
        ros = ros.sort_values("week").groupby("gsis_id").tail(1)
        ros = ros[ros["status"].isin(INACTIVE_ROSTER_STATUSES)]
        if not ros.empty:
            frames.append(pd.DataFrame({
                "player_id": ros["gsis_id"],
                "injury_status": ros["status"].map(INACTIVE_ROSTER_STATUSES),
                "available": False,
                "_priority": 0,  # roster status (e.g. IR) wins over the injury report
            }))
    except Exception as e:
        warnings.warn(f"Could not load weekly rosters for {season}: {e}")

    if not frames:
        return pd.DataFrame(columns=["player_id", "injury_status", "available"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("_priority").drop_duplicates("player_id")
    return out.drop(columns="_priority").reset_index(drop=True)


def add_availability(projections, season, week):
    """Add injury_status ('' if healthy) and available (bool) columns."""
    status = get_availability(season, week)
    out = projections.merge(status, on="player_id", how="left")
    out["injury_status"] = out["injury_status"].fillna("")
    out["available"] = out["available"].fillna(True).astype(bool)
    return out
