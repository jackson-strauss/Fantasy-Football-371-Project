"""Streamlit interface for the AI Fantasy Football Assistant.

Run from the project root:
    streamlit run src/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from explain import compare, headline, player_reasons
from features import load_weekly
from lineup import FLEX_POSITIONS, recommend_lineup
from predict_week import predict_week
from strategy import (STRATEGIES, Simulation, choose_lineup, close_calls, starter_ids,
                      suggest_pickups)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "weekly.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
POSITIONS = ["QB", "RB", "WR", "TE"]

STRATEGY_HELP = {
    "Auto": "Maximize your chance of winning if you enter an opponent; otherwise Balanced.",
    "Balanced": "Highest total projected points.",
    "Safe": "Highest floors. Good when you're favored and just need to avoid a dud.",
    "Upside": "Highest ceilings. Good when you're an underdog and need a big week.",
}

st.set_page_config(page_title="Fantasy Football Assistant", layout="wide")


# ---------- Cached data loading ----------

@st.cache_data(show_spinner=False)
def load_history():
    return load_weekly(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_model_info():
    bundle = joblib.load(MODEL_PATH)
    return {"trained_through": bundle["trained_through_season"],
            "test_mae": bundle["test_mae"],
            "coverage": bundle.get("test_range_coverage"),
            "has_ranges": "quantile_models" in bundle}


@st.cache_data(show_spinner="Projecting players and checking injury reports...")
def get_projections(season, week):
    out, _, _ = predict_week(season, week, data_path=DATA_PATH, model_path=MODEL_PATH)
    return out


@st.cache_resource(show_spinner="Simulating the week...")
def get_simulation(season, week):
    return Simulation(get_projections(season, week))


@st.cache_data(show_spinner=False)
def player_options(season):
    """Everyone who played this season or last, so bye-week players can still be picked."""
    df = load_history()
    recent = df[df["season"] >= season - 1].sort_values(["season", "week"])
    last = recent.groupby("player_id").tail(1).sort_values("player_display_name")
    labels = last["player_display_name"] + " (" + last["position"] + ", " + last["team"] + ")"
    return dict(zip(last["player_id"], labels)), last.set_index("player_id")


# ---------- Formatting helpers ----------

COL_CONFIG = {
    "slot": "Slot",
    "player_display_name": "Player",
    "position": "Pos",
    "team": "Team",
    "opponent_team": "Opp",
    "projected": st.column_config.NumberColumn("Projected", format="%.1f"),
    "floor": st.column_config.NumberColumn("Floor", format="%.1f",
                                           help="10th percentile: a bad week"),
    "ceiling": st.column_config.NumberColumn("Ceiling", format="%.1f",
                                             help="90th percentile: a great week"),
    "risk": st.column_config.TextColumn("Risk", help="Range of outcomes compared with "
                                        "similar players at the position"),
    "why": "Why",
    "injury_status": "Status",
    "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
}
PLAYER_COLS = ["player_display_name", "position", "team", "opponent_team",
               "projected", "floor", "ceiling", "risk", "why", "injury_status"]


def with_why(df):
    df = df.copy()
    if not df.empty:
        df["why"] = [headline(r) if pd.notna(r.get("projected")) else ""
                     for _, r in df.iterrows()]
    return df


def show_table(df, cols, **kw):
    st.dataframe(df.reindex(columns=cols), column_config=COL_CONFIG,
                 hide_index=True, width="stretch", **kw)


def show_reasons(row):
    st.markdown(f"**{row['player_display_name']}** ({row['position']}, {row['team']} vs "
                f"{row['opponent_team']}): **{row['projected']:.1f}** projected, "
                f"range {row['floor']:.1f} to {row['ceiling']:.1f}")
    st.markdown("\n".join(f"- {r}" for r in player_reasons(row)))


def pct(x):
    return f"{x:.0%}" if x is not None else "-"


# ---------- Setup checks ----------

if not DATA_PATH.exists() or not MODEL_PATH.exists() or not load_model_info()["has_ranges"]:
    st.error(
        "Missing or outdated data/model. From the project root run:\n\n"
        "```\npython src/load_data.py\npython src/train_model.py\n```"
    )
    st.stop()

history = load_history()
model_info = load_model_info()

latest_season = int(history["season"].max())
played_weeks = history.loc[history["season"] == latest_season, "week"]
default_week = min(int(played_weeks.max()) + 1, 18)


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Week")
    seasons = sorted(history["season"].unique().tolist(), reverse=True)
    season = st.selectbox("Season", seasons, index=0)
    week = st.number_input("Week", min_value=1, max_value=18,
                           value=default_week if season == latest_season else 1)

    st.header("Strategy")
    strategy = st.radio("Lineup strategy", list(STRATEGY_HELP), index=0,
                        captions=list(STRATEGY_HELP.values()), label_visibility="collapsed")

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
    coverage = model_info["coverage"]
    st.caption(
        f"Model trained on 2018-{model_info['trained_through']} seasons. On the "
        f"{model_info['trained_through']} test season its projections were off by "
        f"{model_info['test_mae']:.1f} PPR points per player on average"
        + (f", and {coverage:.0%} of scores landed inside the floor-to-ceiling range."
           if coverage else ".")
    )
    if st.button("Refresh injury reports"):
        get_projections.clear()
        get_simulation.clear()

try:
    projections = get_projections(int(season), int(week))
except SystemExit as e:  # predict_week raises this when there are no games that week
    st.warning(str(e))
    st.stop()
sim = get_simulation(int(season), int(week))
has_actual = projections["actual"].notna().any()
by_id = projections.set_index("player_id")

st.title("Fantasy Football Assistant")
st.caption(f"PPR projections for {season} week {week}")

lineup_tab, rankings_tab, how_tab = st.tabs(["My lineup", "Player rankings", "How it works"])


# ---------- My lineup ----------

with lineup_tab:
    labels, player_info = player_options(int(season))
    all_ids = list(labels.keys())

    roster_ids = st.multiselect("Your roster", all_ids, format_func=labels.get,
                                key="roster", placeholder="Start typing a player's name...")

    c1, c2 = st.columns(2)
    with c1.expander("Opponent's roster (optional): win probability and strategy"):
        opp_ids = st.multiselect(
            "Opponent's players", [p for p in all_ids if p not in roster_ids],
            format_func=labels.get, key="opponent",
            placeholder="Add your opponent's players...")
        st.caption("We assume your opponent starts their highest-projected lineup.")
    with c2.expander("Free agents (optional): pickup suggestions"):
        fa_ids = st.multiselect(
            "Players available in your league",
            [p for p in all_ids if p not in roster_ids and p not in opp_ids],
            format_func=labels.get, key="free_agents",
            placeholder="Add players on your waiver wire...")
        scan_all = st.checkbox("Or scan every player not on either roster", value=not fa_ids,
                               help="Some of these may already be on other teams in your league.")

    if not roster_ids:
        st.info("Add the players on your roster to get a personalized lineup. "
                "Add your opponent's roster too, to see your chance of winning.")
    else:
        roster = projections[projections["player_id"].isin(roster_ids)]
        missing = player_info.loc[[p for p in roster_ids if p not in by_id.index]]

        # Opponent
        opp_totals, opp_starters = None, None
        if opp_ids:
            opp_roster = projections[projections["player_id"].isin(opp_ids)]
            opp_starters, _ = recommend_lineup(opp_roster, slots)
            opp_totals = sim.totals(starter_ids(opp_starters))

        chosen, candidates = choose_lineup(roster, slots, sim, strategy, opp_totals)
        starters, bench = with_why(chosen["starters"]), with_why(chosen["bench"])
        balanced = candidates[0]

        # ----- Summary -----
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected points", f"{chosen['projected']:.1f}")
        m2.metric("Likely range", f"{chosen['p10']:.0f} to {chosen['p90']:.0f}",
                  help="10th to 90th percentile of 10,000 simulated weeks")
        if opp_totals is not None:
            m3.metric("Opponent projected", f"{opp_starters['projected'].sum():.1f}")
            m4.metric("Win probability", pct(chosen["win_prob"]),
                      delta=(f"{(chosen['win_prob'] - balanced['win_prob']) * 100:+.1f}% vs "
                             "max-projection lineup") if chosen is not balanced else None)
        if has_actual:
            (m3 if opp_totals is None else m1).metric(
                "Actual points", f"{starters['actual'].sum():.1f}")

        # ----- Strategy explanation -----
        if opp_totals is not None:
            favored = balanced["win_prob"] >= 0.5
            if strategy == "Auto":
                if chosen is balanced:
                    msg = (f"You're {'favored' if favored else 'the underdog'} "
                           f"({pct(balanced['win_prob'])} win chance). Your highest-projected "
                           "lineup is also your best shot at winning, so no changes needed.")
                else:
                    safer = chosen["p90"] - chosen["p10"] < balanced["p90"] - balanced["p10"]
                    style = "safer" if safer else "higher-upside"
                    if safer and favored:
                        why = ("When you're favored, avoiding a bad week matters more than "
                               "raw points.")
                    elif not safer and not favored:
                        why = ("When you're the underdog, you need a big week, so upside "
                               "matters more than raw points.")
                    else:
                        why = (f"This matchup is close, and in the simulations the {style} "
                               "lineup wins slightly more often.")
                    msg = (f"You're {'favored' if favored else 'the underdog'} "
                           f"({pct(balanced['win_prob'])} win chance with your highest-projected "
                           f"lineup). {why} This {style} lineup raises your chance to "
                           f"**{pct(chosen['win_prob'])}**.")
                st.info(msg)

            with st.expander("Compare lineup strategies"):
                comp = []
                for c in candidates:
                    swapped_in = c["starters"][c["starters"]["player_id"].isin(
                        c["ids"] - balanced["ids"])]["player_display_name"].tolist()
                    comp.append({"Lineup": c["label"] + ("  ✓" if c is chosen else ""),
                                 "Projected": c["projected"],
                                 "Range": f"{c['p10']:.0f} to {c['p90']:.0f}",
                                 "Win %": c["win_prob"] * 100,
                                 "Changes vs. Balanced": ", ".join(swapped_in) or "-"})
                st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch",
                             column_config={"Projected": st.column_config.NumberColumn(format="%.1f"),
                                            "Win %": st.column_config.NumberColumn(format="%.1f%%")})

                # Distribution of the score margin for the chosen lineup
                margin = chosen["totals"] - opp_totals
                bins = np.arange(np.floor(margin.min() / 5) * 5, margin.max() + 5, 5)
                counts, edges = np.histogram(margin, bins=bins)
                st.caption("Simulated margin of victory (your points minus opponent's)")
                st.bar_chart(pd.DataFrame({"margin": edges[:-1], "share of weeks": counts / counts.sum()})
                             .set_index("margin"), height=200)

            with st.expander("Opponent's projected lineup"):
                show_table(with_why(opp_starters), ["slot"] + PLAYER_COLS)
        elif strategy == "Auto":
            st.caption("Tip: add your opponent's roster above to optimize for your chance "
                       "of winning instead of just projected points.")

        # ----- Starters -----
        cols = ["slot"] + PLAYER_COLS + (["actual"] if has_actual else [])
        st.subheader("Start")
        show_table(starters, cols)

        for _, r in starters.iterrows():
            status = r.get("injury_status")
            if isinstance(status, str) and status.startswith("Questionable"):
                st.warning(f"{r['player_display_name']} is questionable. "
                           "Check the final injury report before kickoff.")

        empty = starters[starters["player_display_name"] == "(empty)"]
        if not empty.empty:
            st.error(f"Not enough healthy players to fill: {', '.join(empty['slot'])}. "
                     "See the pickup suggestions below.")

        with st.expander("Why these starters?"):
            for _, r in starters.dropna(subset=["player_id"]).iterrows():
                show_reasons(r)

        # ----- Close calls -----
        score_col = STRATEGIES.get(strategy, "projected")
        calls = close_calls(chosen["starters"], chosen["bench"], score_col)
        if calls:
            st.subheader("Closest calls")
            for s, b, margin in calls:
                st.markdown("- " + compare(s, b, score_col))

        # ----- Bench -----
        st.subheader("Bench")
        bench_rows = bench.reindex(columns=cols[1:])
        if not missing.empty:
            bench_rows = pd.concat([bench_rows, pd.DataFrame({
                "player_display_name": missing["player_display_name"],
                "position": missing["position"], "team": missing["team"],
                "injury_status": "No game this week / inactive",
            })], ignore_index=True)
        if bench_rows.empty:
            st.write("Nobody on the bench.")
        else:
            show_table(bench_rows, cols[1:])

        # ----- Pickups -----
        if fa_ids or scan_all:
            pool = (projections[projections["player_id"].isin(fa_ids)] if fa_ids else
                    projections[~projections["player_id"].isin(roster_ids + opp_ids)])
            pickups = suggest_pickups(roster, pool, slots, sim, strategy, opp_totals)
            st.subheader("Pickup suggestions")
            if pickups.empty:
                st.write("No available player would improve your lineup this week.")
            else:
                win_mode = strategy == "Auto" and opp_totals is not None
                if win_mode:
                    pickups["gain"] = pickups["gain"] * 100
                gain_label = {"Safe": "Floor gain", "Upside": "Ceiling gain"}.get(
                    strategy, "Win % gain" if win_mode else "Points gain")
                st.dataframe(
                    with_why(pickups).reindex(columns=PLAYER_COLS[:-1] + ["gain", "replaces"]),
                    column_config=COL_CONFIG | {
                        "gain": st.column_config.NumberColumn(
                            gain_label, format="+%.1f%%" if win_mode else "+%.1f"),
                        "replaces": "Would start over"},
                    hide_index=True, width="stretch")
                if not fa_ids:
                    st.caption("Scanned all players not on your or your opponent's roster. "
                               "Add your league's free agents above for accurate suggestions.")


# ---------- Player rankings ----------

with rankings_tab:
    f1, f2, f3, f4 = st.columns([1, 1, 2, 1])
    pos = f1.selectbox("Position", ["All"] + POSITIONS + ["FLEX"])
    risk_filter = f2.selectbox("Risk", ["Any", "Steady", "Moderate", "Boom/bust"])
    search = f3.text_input("Search player")
    show_injured = f4.checkbox("Show injured players")

    view = projections
    if pos == "FLEX":
        view = view[view["position"].isin(FLEX_POSITIONS)]
    elif pos != "All":
        view = view[view["position"] == pos]
    if risk_filter != "Any":
        view = view[view["risk"] == risk_filter]
    if search:
        view = view[view["player_display_name"].str.contains(search, case=False, regex=False)]
    if not show_injured:
        view = view[view["available"]]

    view = with_why(view.reset_index(drop=True))
    view.insert(0, "rank", view.index + 1)
    rank_cols = ["rank"] + PLAYER_COLS + (["actual"] if has_actual else [])
    st.dataframe(view.reindex(columns=rank_cols),
                 column_config=COL_CONFIG | {"rank": "#"},
                 hide_index=True, width="stretch", height=500)

    st.subheader("Explain a player")
    pick = st.selectbox("Player", view["player_id"].tolist(), index=None,
                        format_func=lambda pid: by_id.loc[pid, "player_display_name"],
                        placeholder="Choose a player to see why they're projected this way")
    if pick:
        show_reasons(by_id.loc[pick].to_dict() | {"player_id": pick})


# ---------- How it works ----------

with how_tab:
    st.markdown(f"""
**Projections.** A gradient-boosted model trained on every QB, RB, WR and TE game since 2018
predicts PPR points from recent form (last 3 games), career average, usage (targets and
carries), scoring volatility (last 6 games) and how many points the opponent has allowed to
the position over its last 5 games.

**Range of outcomes.** Three more models predict each player's *floor* (10th percentile),
*median* and *ceiling* (90th percentile). On the {model_info['trained_through']} test season,
{pct(model_info['coverage'])} of actual scores landed between floor and ceiling (the target is 80%).
The *Risk* label compares a player's range with players at the same position who have a similar projection.

**Explanations.** Reasons come from the model's inputs. "The matchup adds X pts" is measured by
re-running the model with a league-average defense in place of the real opponent; recent form
is measured the same way, using the player's career average.

**Win probability.** Each player's week is simulated 10,000 times from their floor, median and
ceiling. Your lineup's simulated totals are compared with your opponent's to estimate how often
you'd win. In *Auto* mode the assistant tries lineups from safest to highest-upside and picks
the one with the best chance of winning. Players are simulated independently, so the link
between a QB and their own receivers isn't modeled.

**Injuries.** Players who are Out, Doubtful, or on injured reserve are never started.
""")
