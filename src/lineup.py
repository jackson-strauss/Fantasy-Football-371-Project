"""Recommend which players to start and bench for a fantasy roster.

Usage (from the project root):
    python src/lineup.py --roster my_roster.txt
    python src/lineup.py --players "Josh Allen, Bijan Robinson, Puka Nacua, ..."
    python src/lineup.py --roster my_roster.txt --season 2026 --week 10

A roster file has one player name per line (blank lines and lines starting
with # are ignored).
"""
import argparse
import difflib
import re

import pandas as pd

from predict_week import predict_week

# Standard lineup. FLEX can be filled by any of the listed positions.
DEFAULT_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
FLEX_POSITIONS = {"RB", "WR", "TE"}


def normalize(name):
    """Lowercase, drop suffixes, spaces and punctuation.

    'Kenneth Walker III' -> 'kennethwalker', "Ja'Marr Chase" / 'Ja Marr Chase' -> 'jamarrchase'.
    """
    name = re.sub(r"[^a-z ]", "", name.lower().replace("-", " "))
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return name.replace(" ", "")


def match_roster(projections, roster_names):
    """Match user-entered names to projection rows.

    Returns (matched DataFrame, list of (name, suggestions) that didn't match).
    """
    proj = projections.copy()
    proj["_key"] = proj["player_display_name"].map(normalize)
    keys = proj["_key"].unique().tolist()

    rows, unmatched = [], []
    for name in roster_names:
        key = normalize(name)
        hits = proj[proj["_key"] == key]
        if hits.empty:
            close = difflib.get_close_matches(key, keys, n=3, cutoff=0.75)
            suggestions = proj[proj["_key"].isin(close)]["player_display_name"].unique().tolist()
            unmatched.append((name, suggestions))
            continue
        # If two players share a name, take the one with the larger projection
        rows.append(hits.sort_values("projected", ascending=False).iloc[0])

    matched = pd.DataFrame(rows).drop(columns="_key") if rows else proj.iloc[0:0].drop(columns="_key")
    return matched.drop_duplicates("player_id"), unmatched


def recommend_lineup(roster, slots=DEFAULT_SLOTS):
    """Fill lineup slots with the highest-projected eligible players.

    Fixed position slots are filled first, then FLEX takes the best remaining
    RB/WR/TE. This greedy order is optimal for standard lineups.
    Players marked unavailable (Out, Doubtful, IR, ...) are never started.
    Returns (starters DataFrame with a 'slot' column, bench DataFrame).
    """
    pool = roster.sort_values("projected", ascending=False)
    eligible = pool[pool["available"]] if "available" in pool.columns else pool
    used = set()
    starters = []

    fixed = [s for s in slots if s != "FLEX"]
    flex_count = len(slots) - len(fixed)

    for slot in fixed:
        options = eligible[(eligible["position"] == slot) & (~eligible["player_id"].isin(used))]
        if options.empty:
            starters.append({"slot": slot, "player_display_name": "(empty)"})
            continue
        pick = options.iloc[0]
        used.add(pick["player_id"])
        starters.append({"slot": slot, **pick.to_dict()})

    for _ in range(flex_count):
        options = eligible[eligible["position"].isin(FLEX_POSITIONS)
                           & (~eligible["player_id"].isin(used))]
        if options.empty:
            starters.append({"slot": "FLEX", "player_display_name": "(empty)"})
            continue
        pick = options.iloc[0]
        used.add(pick["player_id"])
        starters.append({"slot": "FLEX", **pick.to_dict()})

    starters = pd.DataFrame(starters)
    bench = pool[~pool["player_id"].isin(used)]
    return starters, bench


def read_roster(args):
    if args.players:
        return [n.strip() for n in args.players.split(",") if n.strip()]
    with open(args.roster) as f:
        return [line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--roster", help="text file with one player name per line")
    src.add_argument("--players", help="comma-separated player names")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    args = parser.parse_args()

    roster_names = read_roster(args)
    projections, season, week = predict_week(args.season, args.week)
    roster, unmatched = match_roster(projections, roster_names)
    starters, bench = recommend_lineup(roster)

    has_actual = "actual" in roster.columns and roster["actual"].notna().any()
    cols = ["slot", "player_display_name", "position", "team", "opponent_team",
            "projected", "injury_status"]
    if has_actual:
        cols.append("actual")

    print(f"\nRecommended lineup - {season} week {week}")
    print(starters.reindex(columns=cols).astype(object).fillna("").to_string(index=False))
    print(f"\nProjected total: {starters['projected'].sum():.1f}")
    if has_actual:
        print(f"Actual total:    {starters['actual'].sum():.1f}")

    print("\nBench")
    if bench.empty:
        print("  (none)")
    else:
        print(bench.reindex(columns=cols[1:]).astype(object).fillna("").to_string(index=False))

    if unmatched:
        print("\nNo projection for these players (on bye, inactive/injured, or name not recognized):")
        for name, suggestions in unmatched:
            hint = f"  did you mean: {', '.join(suggestions)}?" if suggestions else ""
            print(f"  - {name}{hint}")


if __name__ == "__main__":
    main()
