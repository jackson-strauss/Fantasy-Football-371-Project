"""Streamlit interface for the AI Fantasy Football Assistant.

Run from the project root:
    streamlit run src/app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import load_weekly
from lineup import FLEX_POSITIONS, recommend_lineup
from predict_week import predict_week

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "weekly.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
POSITIONS = ["QB", "RB", "WR", "TE"]

st.set_page_config(page_title="Fantasy Football Assistant", layout="wide")


# ---------- Cached data loading ----------

@st.cache_data(show_spinner=False)
def load_history():
    return load_weekly(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_model_info():
    bundle = joblib.load(MODEL_PATH)
    return {"trained_through": bundle["trained_through_season"],
            "test_mae": bundle["test_mae"]}


@st.cache_data(show_spinner="Projecting players and checking injury reports...")
def get_projections(season, week):
    out, _, _ = predict_week(season, week, data_path=DATA_PATH, model_path=MODEL_PATH)
    return out


@st.cache_data(show_spinner=False)
def player_options(season):
    """Everyone who played this season or last, so bye-week players can still be picked."""
    df = load_history()
    recent = df[df["season"] >= season - 1].sort_values(["season", "week"])
    last = recent.groupby("player_id").tail(1)
    last = last.sort_values("player_display_name")
    labels = last["player_display_name"] + " (" + last["position"] + ", " + last["team"] + ")"
    return dict(zip(last["player_id"], labels)), last.set_index("player_id")


# ---------- Setup checks ----------

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Missing data or model. From the project root run:\n\n"
        "```\npython src/load_data.py\npython src/train_model.py\n```"
    )
    st.stop()

history = load_history()
model_info = load_model_info()

latest_season = int(history["season"].max())
played_weeks = history.loc[history["season"] == latest_season, "week"]
default_week = min(int(played_weeks.max()) + 1, 18)


# ---------- Sidebar: week and lineup settings ----------

with st.sidebar:
    st.header("Week")
    seasons = sorted(history["season"].unique().tolist(), reverse=True)
    season = st.selectbox("Season", seasons, index=0)
    week = st.number_input("Week", min_value=1, max_value=18,
                           value=default_week if season == latest_season else 1)

    st.header("Lineup slots")
    slot_counts = {
        "QB": st.number_input("QB", 0, 3, 1),
        "RB": st.number_input("RB", 0, 4, 2),
        "WR": st.number_input("WR", 0, 4, 2),
        "TE": st.number_input("TE", 0, 3, 1),
        "FLEX": st.number_input("FLEX (RB/WR/TE)", 0, 3, 1),
    }
    slots = [pos for pos, n in slot_counts.items() for _ in range(n)]

    st.divider()
    st.caption(
        f"Model trained on 2018-{model_info['trained_through']} seasons. "
        f"Average error on the {model_info['trained_through']} test season: "
        f"{model_info['test_mae']:.1f} PPR points per player."
    )
    if st.button("Refresh injury reports"):
        get_projections.clear()

try:
    projections = get_projections(int(season), int(week))
except SystemExit as e:  # predict_week raises this when there are no games that week
    st.warning(str(e))
    st.stop()

has_actual = projections["actual"].notna().any()

st.title("Fantasy Football Assistant")
st.caption(f"PPR projections for {season} week {week}")

lineup_tab, rankings_tab = st.tabs(["My lineup", "Player rankings"])


# ---------- My lineup ----------

with lineup_tab:
    labels, player_info = player_options(int(season))
    roster_ids = st.multiselect(
        "Your roster",
        options=list(labels.keys()),
        format_func=labels.get,
        key="roster",
        placeholder="Start typing a player's name...",
    )

    if not roster_ids:
        st.info("Add the players on your roster to get start/sit recommendations.")
    else:
        roster = projections[projections["player_id"].isin(roster_ids)]
        starters, bench = recommend_lineup(roster, slots)

        # Roster players with no projection this week (bye, or not active recently)
        missing = player_info.loc[[pid for pid in roster_ids
                                   if pid not in set(roster["player_id"])]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Projected points", f"{starters['projected'].sum():.1f}")
        if has_actual:
            c2.metric("Actual points", f"{starters['actual'].sum():.1f}")
        empty = (starters["player_display_name"] == "(empty)").sum()
        if empty:
            c3.metric("Empty slots", int(empty))

        cols = ["slot", "player_display_name", "position", "team", "opponent_team",
                "projected", "injury_status"] + (["actual"] if has_actual else [])
        col_config = {
            "slot": "Slot",
            "player_display_name": "Player",
            "position": "Pos",
            "team": "Team",
            "opponent_team": "Opp",
            "projected": st.column_config.NumberColumn("Projected", format="%.1f"),
            "injury_status": "Status",
            "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
        }

        st.subheader("Start")
        st.dataframe(starters.reindex(columns=cols), column_config=col_config,
                     hide_index=True, width="stretch")

        questionable = starters[starters["injury_status"].fillna("").str.startswith("Questionable")]
        for name in questionable["player_display_name"]:
            st.warning(f"{name} is questionable. Check the final injury report before kickoff.")

        st.subheader("Bench")
        bench_rows = bench.reindex(columns=cols[1:])
        if not missing.empty:
            bench_rows = pd.concat([bench_rows, pd.DataFrame({
                "player_display_name": missing["player_display_name"],
                "position": missing["position"],
                "team": missing["team"],
                "injury_status": "No game this week / inactive",
            })], ignore_index=True)
        if bench_rows.empty:
            st.write("Nobody on the bench.")
        else:
            st.dataframe(bench_rows, column_config=col_config,
                         hide_index=True, width="stretch")

        if empty:
            needed = [s for s in starters.loc[starters["player_display_name"] == "(empty)", "slot"]]
            st.error(f"Not enough healthy players to fill: {', '.join(needed)}. "
                     "Check the rankings tab for options to pick up.")


# ---------- Player rankings ----------

with rankings_tab:
    f1, f2, f3 = st.columns([1, 2, 1])
    pos = f1.selectbox("Position", ["All"] + POSITIONS + ["FLEX"])
    search = f2.text_input("Search player")
    show_injured = f3.checkbox("Show injured players")

    view = projections
    if pos == "FLEX":
        view = view[view["position"].isin(FLEX_POSITIONS)]
    elif pos != "All":
        view = view[view["position"] == pos]
    if search:
        view = view[view["player_display_name"].str.contains(search, case=False, regex=False)]
    if not show_injured:
        view = view[view["available"]]

    view = view.reset_index(drop=True)
    view.insert(0, "rank", view.index + 1)
    rank_cols = ["rank", "player_display_name", "position", "team", "opponent_team",
                 "projected", "injury_status"] + (["actual"] if has_actual else [])
    st.dataframe(
        view[rank_cols],
        column_config={
            "rank": "#",
            "player_display_name": "Player",
            "position": "Pos",
            "team": "Team",
            "opponent_team": "Opp",
            "projected": st.column_config.NumberColumn("Projected", format="%.1f"),
            "injury_status": "Status",
            "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
        },
        hide_index=True,
        width="stretch",
        height=600,
    )
