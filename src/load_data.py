from pathlib import Path

import nflreadpy as nfl

Path("data").mkdir(exist_ok=True)

current = nfl.get_current_season()
seasons = list(range(2018, current + 1))

print(f"Downloading weekly player stats for {seasons[0]}-{seasons[-1]}...")

# Weekly player stats: one row per player per game
weekly = nfl.load_player_stats(seasons).to_pandas()
weekly.to_csv("data/weekly.csv", index=False)

print("Saved data/weekly.csv")
print("Shape:", weekly.shape)
print("Columns:", weekly.columns.tolist())
print(
    weekly[
        ["player_display_name", "position", "season", "week",
         "opponent_team", "fantasy_points_ppr"]
    ].head()
)