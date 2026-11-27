# AI Fantasy Football Assistant

A machine-learning assistant that projects NFL players' PPR fantasy points for an
upcoming week and recommends which players on your roster to start or bench.

The model is trained on weekly NFL player stats since 2018 (via
[nflreadpy](https://github.com/nflverse/nflreadpy)), using recent form, career
averages, usage (targets and carries), scoring volatility, and how many points each
opponent's defense has allowed to the position. Injury reports and roster status are
used to keep injured players out of the starting lineup.

What it does beyond a normal fantasy app's projections:

- **Personalized recommendations**: builds the best lineup from your whole roster,
  explains the closest start/sit calls, and suggests free-agent pickups that would
  improve your lineup (or your chance of winning).
- **Explainable projections**: every player gets plain-English reasons, e.g. *"Averaging
  18.1 PPR points over the last 3 games... CLE has allowed the 10th-most PPR points to
  TEs... the matchup adds 0.9 pts compared with an average defense."*
- **Risk analysis**: a floor (10th percentile), ceiling (90th percentile), and a
  Steady / Moderate / Boom-bust label for each player, plus Safe and Upside lineup strategies.
- **Opponent-specific strategy**: enter your opponent's roster and the assistant simulates
  the week 10,000 times to estimate your win probability, then picks the lineup that
  gives you the best chance of winning: safer when you're favored, higher-upside when
  you're the underdog.

## Setup

Requires Python 3.10+. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then download the data and train the model. The data and model files aren't
stored in git, so everyone needs to run these once after cloning:

```bash
python src/load_data.py      # downloads stats to data/weekly.csv (~1 min)
python src/train_model.py    # trains the model, saves models/model.joblib
```

Re-run both each week to pick up the latest games.

## Running the app

```bash
streamlit run src/app.py
```

Then open http://localhost:8501. The site is only available while this command is
running; press Ctrl+C in the terminal to stop it.

- **My lineup**: add your roster to get recommended starters with reasons, the closest
  start/sit calls, and your bench. Optionally add your opponent's roster (for win
  probability and strategy) and your league's free agents (for pickup suggestions).
- **Player rankings**: projections, floors, ceilings, and risk for every player, with an
  "Explain a player" section.
- **How it works**: a short description of the methods.
- Use the sidebar to pick the season, week, strategy (Auto / Balanced / Safe / Upside),
  and your league's lineup slots. Choosing a past week also shows the points each player
  actually scored.

## Command-line tools

```bash
# Projections for the next unplayed week (or pick one)
python src/predict_week.py
python src/predict_week.py --season 2026 --week 12 --position WR --top 30

# Start/sit recommendations for a roster (one name per line, see example_roster.txt)
python src/lineup.py --roster example_roster.txt
python src/lineup.py --players "Josh Allen, Bijan Robinson, Puka Nacua"
```

## Project structure

| File | Purpose |
|---|---|
| `src/load_data.py` | Download weekly player stats |
| `src/features.py` | Shared feature engineering used for training and prediction |
| `src/train_model.py` | Train and evaluate the model (overall and per position) and save it |
| `src/predict_week.py` | Project points for every player in a given week |
| `src/injuries.py` | Player availability from injury reports and weekly roster status |
| `src/lineup.py` | Match a roster and pick the best starting lineup |
| `src/explain.py` | Plain-English explanations of projections and start/sit calls |
| `src/strategy.py` | Simulations, win probability, risk-based lineups, and pickup suggestions |
| `src/app.py` | Streamlit web interface |

## Model performance

Tested on the 2025 season after training on 2018–2024. MAE is the average
error in PPR points per player per game (lower is better). The baseline predicts
each player's average over their last 3 games. "Range hit" is how often the actual
score landed between the predicted floor and ceiling (the target is 80%).

| Position | Baseline MAE | Model MAE | Range hit |
|---|---|---|---|
| QB | 7.03 | 6.77 | 74.3% |
| RB | 4.70 | 4.46 | 81.5% |
| WR | 4.57 | 4.30 | 79.0% |
| TE | 3.87 | 3.65 | 81.2% |
| **All** | **4.72** | **4.47** | **79.6%** |

Limitations: players are simulated independently (a QB and their own receivers tend
to have good or bad weeks together, which isn't modeled), scoring is PPR only, and
QB ranges are a little too narrow.

## Troubleshooting

- **`command not found: streamlit`**: activate the virtual environment first (`source .venv/bin/activate`).
- **"Missing data or model" in the app**: run `load_data.py` and `train_model.py` from the project root.
- **"Site can't be reached"**: the `streamlit run` command must be running in a terminal.
- **Port 8501 already in use**: run `streamlit run src/app.py --server.port 8502`.
