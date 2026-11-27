"""Risk-aware lineup strategy: simulations, win probability, and pickups.

Each player's outcome is simulated from the model's floor (10th percentile),
median and ceiling (90th percentile) using a split normal distribution, which
allows a longer upside tail than downside. Players are simulated
independently (teammate correlation is ignored).

All lineups share the same simulated weeks ("common random numbers"), so
comparisons between lineups aren't affected by simulation noise.
"""
import numpy as np
import pandas as pd

from lineup import FLEX_POSITIONS, recommend_lineup

Z90 = 1.2816  # z-score of the 90th percentile
N_SIMS = 10_000
MIN_SCORE = -3.0  # fantasy scores below this are essentially impossible

# Named strategies -> the column the lineup builder maximizes
STRATEGIES = {"Balanced": "projected", "Safe": "floor", "Upside": "ceiling"}

# Extra candidate lineups between safe and high-upside:
# score = projected + lambda * (half the floor-to-ceiling range)
LAMBDAS = [-0.6, -0.3, 0.3, 0.6]


class Simulation:
    """Simulated fantasy points for every projected player, N_SIMS weeks each."""

    def __init__(self, projections, n_sims=N_SIMS, seed=42):
        c = projections["median"].to_numpy(float)
        sd_low = np.maximum((c - projections["floor"].to_numpy(float)) / Z90, 0.5)
        sd_high = np.maximum((projections["ceiling"].to_numpy(float) - c) / Z90, 0.5)
        z = np.random.default_rng(seed).standard_normal((n_sims, len(c)))
        samples = c + np.where(z < 0, z * sd_low, z * sd_high)
        self.samples = np.clip(samples, MIN_SCORE, None).astype(np.float32)
        self.col = {pid: i for i, pid in enumerate(projections["player_id"])}
        self.n_sims = n_sims

    def totals(self, player_ids):
        """Simulated lineup totals (one per simulated week). Unknown players score 0."""
        idx = [self.col[p] for p in player_ids if p in self.col]
        if not idx:
            return np.zeros(self.n_sims, dtype=np.float32)
        return self.samples[:, idx].sum(axis=1)


def starter_ids(starters):
    if starters.empty or "player_id" not in starters:
        return []
    return starters["player_id"].dropna().tolist()


def win_probability(my_totals, opp_totals):
    return float((my_totals > opp_totals).mean() + 0.5 * (my_totals == opp_totals).mean())


def candidate_lineups(roster, slots):
    """Distinct lineups ranging from safest to highest-upside."""
    r = roster.copy()
    half_range = (r["ceiling"] - r["floor"]) / 2
    specs = list(STRATEGIES.items())
    for lam in LAMBDAS:
        col = f"_lam{lam}"
        r[col] = r["projected"] + lam * half_range
        specs.append(("Safer mix" if lam < 0 else "Higher-upside mix", col))

    seen, out = set(), []
    for label, col in specs:
        starters, bench = recommend_lineup(r, slots, score=col)
        ids = frozenset(starter_ids(starters))
        if ids in seen:
            continue
        seen.add(ids)
        drop = [c for c in starters.columns if c.startswith("_lam")]
        out.append({"label": label, "starters": starters.drop(columns=drop),
                    "bench": bench.drop(columns=drop), "ids": ids})
    return out


def evaluate(candidate, sim, opp_totals=None):
    totals = sim.totals(candidate["ids"])
    candidate["projected"] = float(candidate["starters"]["projected"].sum())
    candidate["p10"], candidate["p90"] = (float(x) for x in np.percentile(totals, [10, 90]))
    candidate["win_prob"] = (win_probability(totals, opp_totals)
                             if opp_totals is not None else None)
    candidate["totals"] = totals
    return candidate


def choose_lineup(roster, slots, sim, strategy="Auto", opp_totals=None):
    """Pick a lineup. Returns (chosen candidate, all evaluated candidates).

    strategy: "Balanced" / "Safe" / "Upside" use that objective directly.
    "Auto" maximizes win probability when an opponent is given, otherwise
    it's the same as "Balanced".
    """
    cands = [evaluate(c, sim, opp_totals) for c in candidate_lineups(roster, slots)]
    if strategy in STRATEGIES:
        chosen = next(c for c in cands if c["label"] == strategy)
    elif opp_totals is not None:
        # Highest win probability; ties go to the highest projection
        chosen = max(cands, key=lambda c: (round(c["win_prob"], 3), c["projected"]))
    else:
        chosen = cands[0]  # Balanced
    return chosen, cands


def objective(candidate, strategy, has_opponent):
    """The number a strategy is trying to maximize, for comparing rosters."""
    if strategy == "Auto" and has_opponent:
        return candidate["win_prob"]
    return {"Safe": candidate["p10"], "Upside": candidate["p90"]}.get(
        strategy, candidate["projected"])


def close_calls(starters, bench, score="projected", n=3):
    """The closest start/sit decisions: (starter row, best bench alternative, margin)."""
    if bench.empty:
        return []
    alts = bench[bench["available"]] if "available" in bench else bench
    calls = []
    for _, s in starters.iterrows():
        if pd.isna(s.get("player_id")):
            continue
        ok = FLEX_POSITIONS if s["slot"] == "FLEX" else {s["slot"]}
        options = alts[alts["position"].isin(ok)]
        if options.empty:
            continue
        best = options.sort_values(score, ascending=False).iloc[0]
        calls.append((s, best, float(s[score] - best[score])))
    calls.sort(key=lambda c: c[2])
    # One entry per bench player, closest first
    seen, out = set(), []
    for c in calls:
        if c[1]["player_id"] not in seen:
            seen.add(c[1]["player_id"])
            out.append(c)
    return out[:n]


def suggest_pickups(roster, pool, slots, sim, strategy="Auto", opp_totals=None,
                    per_position=8, n=5):
    """Free agents who would most improve the lineup.

    pool: projections for players the user could add. Returns a DataFrame with
    the player, the gain in the strategy's objective, and who they'd replace.
    """
    has_opp = opp_totals is not None
    base, _ = choose_lineup(roster, slots, sim, strategy, opp_totals)
    base_value = objective(base, strategy, has_opp)
    score = STRATEGIES.get(strategy, "projected")

    pool = pool[pool["available"] & ~pool["player_id"].isin(roster["player_id"])]
    if pool.empty:
        return pd.DataFrame()

    # Only players who beat the weakest starter they could replace can help
    starters = base["starters"].dropna(subset=["player_id"])
    weakest = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        eligible = starters[(starters["slot"] == pos)
                            | ((starters["slot"] == "FLEX") & (pos in FLEX_POSITIONS))]
        weakest[pos] = eligible[score].min() if not eligible.empty else -np.inf
    pool = pool[pool[score] > pool["position"].map(weakest)]
    pool = pool.sort_values(score, ascending=False).groupby("position").head(per_position)

    rows = []
    for _, fa in pool.iterrows():
        new_roster = pd.concat([roster, fa.to_frame().T], ignore_index=True)
        new_roster[score] = new_roster[score].astype(float)
        chosen, _ = choose_lineup(new_roster, slots, sim, strategy, opp_totals)
        gain = objective(chosen, strategy, has_opp) - base_value
        if gain <= (0.005 if has_opp and strategy == "Auto" else 0.05):
            continue
        replaced = base["starters"][base["starters"]["player_id"].isin(
            base["ids"] - chosen["ids"])]["player_display_name"].tolist()
        rows.append(fa.to_dict() | {"gain": gain, "replaces": ", ".join(replaced) or "-"})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("gain", ascending=False).head(n)
