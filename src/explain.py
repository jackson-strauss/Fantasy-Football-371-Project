"""Plain-English explanations of player projections.

Explanations are built from the same inputs the model uses (recent form,
usage, opponent, volatility) plus two model-based effects computed in
predict_week.add_projection_details:
  * matchup_effect: projection minus the projection vs. a league-median defense
  * form_effect:    projection minus the projection if recent form matched
                    the player's career average
"""
import math

ROLL3 = "fantasy_points_ppr_roll3"


def ordinal(n):
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _num(row, col):
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _signed(x):
    return f"{x:+.1f}"


def matchup_label(row):
    rank, count = _num(row, "opp_rank"), _num(row, "opp_count")
    if rank is None or count is None:
        return None
    if rank <= count / 3:
        return "favorable"
    if rank > 2 * count / 3:
        return "tough"
    return "neutral"


def form_direction(row):
    """+1 if the player is running hot vs. career average, -1 if cold, 0 otherwise."""
    roll3, career = _num(row, ROLL3), _num(row, "ppr_career_avg")
    if roll3 is None or career is None or (_num(row, "games_played") or 0) < 6:
        return 0
    if roll3 - career >= 2:
        return 1
    if roll3 - career <= -2:
        return -1
    return 0


def consistent_effects(row):
    """Model effects whose direction agrees with the underlying stats.

    Tree models can respond non-monotonically to unusual inputs (e.g. a huge
    recent average); an effect pointing the "wrong" way would read as a
    contradiction, so it is left out of the explanation.
    """
    out = {}
    me = _num(row, "matchup_effect")
    label = matchup_label(row)
    if me is not None and abs(me) >= 0.3 and not (
            (label == "favorable" and me < 0) or (label == "tough" and me > 0)):
        out["matchup"] = me
    fe = _num(row, "form_effect")
    direction = form_direction(row)
    if fe is not None and abs(fe) >= 0.3 and direction != 0 and fe * direction > 0:
        out["form"] = fe
    return out


def matchup_sentence(row):
    rank, count = _num(row, "opp_rank"), _num(row, "opp_count")
    allowed = _num(row, "opp_allowed_avg")
    if rank is None or count is None or allowed is None:
        return None
    pos, opp = row["position"], row["opponent_team"]
    rank, count = int(rank), int(count)
    if rank <= count / 2:
        where = f"the {ordinal(rank)}-most"
    else:
        where = f"the {ordinal(count - rank + 1)}-fewest"
    return (f"{matchup_label(row).capitalize()} matchup: {opp} has allowed {where} PPR points "
            f"to {pos}s of the {count} defenses playing this week "
            f"({allowed:.1f} per {pos} over its last 5 games).")


def player_reasons(row):
    """List of explanation sentences for one projected player (a dict-like row)."""
    reasons = []
    pos = row["position"]
    roll3, career = _num(row, ROLL3), _num(row, "ppr_career_avg")
    games = _num(row, "games_played") or 0

    # Recent form vs. career
    if roll3 is not None:
        s = f"Averaging {roll3:.1f} PPR points over the last 3 games"
        if career is not None and games >= 6 and abs(roll3 - career) >= 1:
            direction = "up from" if roll3 > career else "down from"
            s += f", {direction} a {career:.1f} career average"
        reasons.append(s + ".")

    # Usage
    targets, carries = _num(row, "targets_roll3"), _num(row, "carries_roll3")
    if pos in ("WR", "TE") and targets is not None:
        reasons.append(f"Seeing {targets:.1f} targets per game over the last 3.")
    elif pos == "RB" and carries is not None:
        t = f" and {targets:.1f} targets" if targets else ""
        reasons.append(f"Handling {carries:.1f} carries{t} per game over the last 3.")
    elif pos == "QB" and carries is not None and carries >= 4:
        reasons.append(f"Adds rushing value with {carries:.1f} carries per game.")

    # Matchup
    m = matchup_sentence(row)
    if m:
        reasons.append(m)

    # What actually moves the model's number
    effects = []
    eff = consistent_effects(row)
    if "matchup" in eff:
        me = eff["matchup"]
        effects.append(f"the matchup {'adds' if me > 0 else 'costs'} {abs(me):.1f} pts "
                       "compared with an average defense")
    if "form" in eff:
        fe = eff["form"]
        effects.append(f"recent form {'adds' if fe > 0 else 'costs'} {abs(fe):.1f} pts "
                       "compared with career-average form")
    if effects:
        reasons.append("In the model, " + " and ".join(effects) + ".")
    if roll3 is not None and roll3 - row["projected"] >= 5:
        reasons.append(f"The projection ({row['projected']:.1f}) is well below the recent "
                       f"average: the model expects a hot streak to cool off, which is what "
                       f"usually happens historically.")

    # Range of outcomes
    floor, ceiling = _num(row, "floor"), _num(row, "ceiling")
    if floor is not None and ceiling is not None:
        risk = row.get("risk", "")
        risk_text = {
            "Steady": "a narrower range than similar players, so a safer start",
            "Boom/bust": "a wider range than similar players: more upside, but a lower floor",
            "Moderate": "a typical range for a player with this projection",
        }.get(risk, "")
        s = f"Likely range: {floor:.1f} to {ceiling:.1f} points (80% of outcomes)"
        reasons.append(s + (f", {risk_text}." if risk_text else "."))
    std6 = _num(row, "ppr_std6")
    if std6 is not None:
        reasons.append(f"Weekly scoring has swung by about ±{std6:.1f} points over the last 6 games.")

    if games < 4:
        reasons.append(f"Only {int(games)} career games, so this projection is less certain.")
    status = row.get("injury_status")
    if isinstance(status, str) and status.startswith("Questionable"):
        reasons.append("Listed as Questionable. Check the final injury report before kickoff.")
    return reasons


def headline(row):
    """One short phrase summarizing the biggest reason behind a projection."""
    parts = []
    label = matchup_label(row)
    if label in ("favorable", "tough"):
        parts.append(f"{label} matchup vs {row['opponent_team']}")
    direction = form_direction(row)
    if direction > 0:
        parts.append("hot recent form")
    elif direction < 0:
        parts.append("cold recent form")
    risk = row.get("risk")
    if risk == "Steady":
        parts.append("high floor")
    elif risk == "Boom/bust":
        parts.append("boom/bust")
    if not parts:
        return "Typical week expected"
    text = ", ".join(parts)
    return text[0].upper() + text[1:]


def compare(start, sit, score="projected"):
    """Explain why `start` is recommended over `sit` (dict-like rows)."""
    label = {"projected": "projected", "floor": "floor", "ceiling": "ceiling"}.get(score, score)
    diff = start[score] - sit[score]
    text = (f"**{start['player_display_name']}** over **{sit['player_display_name']}**: "
            f"{_signed(diff)} pts {label}")
    if score != "projected":
        text += f" ({start['projected']:.1f} vs {sit['projected']:.1f} projected)"
    edges = []
    ea, eb = consistent_effects(start), consistent_effects(sit)
    if ea.get("matchup", 0) - eb.get("matchup", 0) >= 0.5:
        edges.append(f"better matchup ({start['opponent_team']} vs {sit['opponent_team']})")
    if ea.get("form", 0) - eb.get("form", 0) >= 0.5:
        edges.append("better recent form")
    if start["floor"] - sit["floor"] >= 1:
        edges.append(f"higher floor ({start['floor']:.1f} vs {sit['floor']:.1f})")
    if start["ceiling"] - sit["ceiling"] >= 1:
        edges.append(f"higher ceiling ({start['ceiling']:.1f} vs {sit['ceiling']:.1f})")
    if edges:
        text += ". Edge: " + ", ".join(edges)
    return text + "."
