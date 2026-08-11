"""Reward. Fully programmatic — no LLM appears anywhere in this module, by design.

Shape:

    reward_t = Φ(s_t) − Φ(s_{t-1})        shaping, policy-invariant (§ below)
             − friction_t                  per-turn cost of the agent's time

    reward_T = realized_client_utility     terminal: hidden weights · realized itinerary
             + budget_efficiency           terminal, GATED on hard needs met
             − disruption_damage           act II: value forfeited + scramble costs
             − patience_forfeit            terminal, if the client walked away

**The potential term and the footgun in it.** Ng, Harada & Russell (ICML 1999) prove that
`F(s,a,s') = γΦ(s') − Φ(s)` is *necessary and sufficient* for the shaping to leave the
optimal policy unchanged. Any shaping term whose sum around a closed state cycle is non-zero
creates a positive-reward loop the agent will farm instead of solving the task — their
canonical case is Randløv's bicycle, which "learned to ride in tiny circles near the start
state."

Less well known, and the part most implementations get wrong: Grześ (AAMAS 2017) shows the
trajectory utility picks up a `γ^N·Φ(s_N)` term that **depends on which terminal you reach
and can flip the optimal policy**. The fix is `Φ(terminal) = 0` — *including at truncation*,
because a step cap manufactures exactly the multiple-terminal-states situation he warns
about. Both are enforced here and asserted in `tests/test_rewards.py`.

**No exploration bonus.** Rewarding exploration tends to buy turns that look exploratory
and do nothing: questions asked for the bonus rather than for the answer. Asking the client
questions is made worthwhile by the hidden weights, not by a bonus for asking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from travel_env.contracts import ClientPersona, PreferenceAxis
from travel_env.personas import normalized_weights

# Per-turn friction. Deliberately small: it should nudge against dithering, not dominate.
# The real time pressure is patience (a hard terminal) and the turn cap (truncation).
FRICTION_PER_TURN = 0.004
FRICTION_REDUNDANT_SEARCH = 0.010
"""Repeating a search you already ran this episode. Cheap, but it accumulates."""

PATIENCE_FORFEIT = 1.0
"""Terminal penalty when the client walks. Comparable to a whole good itinerary, because
losing the client is comparable to not doing the job."""

UTILITY_SCALE = 1.0
EFFICIENCY_SCALE = 0.25
DISRUPTION_SCALE = 0.9


@dataclass
class RewardInputs:
    """Everything the reward reads. Assembled by the env; nothing here is an LLM call."""

    persona: ClientPersona
    realized_features: dict[PreferenceAxis, float]
    spent_cents: int
    budget_cents: int
    hard_needs_met: bool
    """Every declared hard constraint satisfied: complete itinerary, party fits, within
    budget, feasible schedule. The efficiency bonus is gated on this."""
    itinerary_complete: bool
    disruption_damage_cents: int
    forfeited_activity_value: float
    client_walked: bool
    turns_used: int
    redundant_searches: int


def potential(
    *,
    slots_filled: float,
    belief_confidence: float,
    is_terminal: bool,
) -> float:
    """Φ(s) ∈ [0, 1] — progress toward a bookable, well-informed plan.

    `slots_filled` is the fraction of the required itinerary slots that are filled.
    `belief_confidence` is the fraction of preference axes the agent has elicited. The
    product, not the sum: a fully-booked trip against zero knowledge of the client is not
    progress, and perfect knowledge with nothing booked is not either.

    Returns exactly 0.0 at any terminal or truncated state. See the module docstring — this
    is the Grześ condition, and it is not optional.
    """
    if is_terminal:
        return 0.0
    return float(slots_filled * (0.35 + 0.65 * belief_confidence))


def realized_utility(inputs: RewardInputs) -> float:
    """The dot product against the client's hidden weights, in [0, 1].

    This is the whole reason the environment is interesting. The weights never appear in an
    observation, so an agent that books without eliciting them is betting on its prior —
    and the expected value of that bet is low, without any penalty term saying so.
    """
    weights = normalized_weights(inputs.persona)
    feats = dict(inputs.realized_features)

    # COST is the one axis the itinerary cannot supply on its own: "good value" is only
    # meaningful against this client's budget.
    if inputs.budget_cents > 0:
        spend_ratio = inputs.spent_cents / inputs.budget_cents
        # Peaks at ~85% of budget. Underspending is not thrift if the client wanted a nice
        # trip; overspending is a hard gate elsewhere, so this only shapes within-budget.
        feats[PreferenceAxis.COST] = max(0.0, 1.0 - abs(spend_ratio - 0.85) / 0.85)
    else:
        feats[PreferenceAxis.COST] = 0.0

    return sum(weights[axis] * feats.get(axis, 0.0) for axis in PreferenceAxis)


def budget_efficiency(inputs: RewardInputs) -> float:
    """Value delivered per unit spent — the *only* ratio in the whole reward.

    Denominator audit: an agent games a ratio by shrinking its denominator, not by growing
    the numerator. The defence here is the `hard_needs_met` gate — booking less to spend
    less fails the gate and scores zero, so shrinking the denominator cannot win. This is
    the same discipline SlimSearcher arrived at for cost/quality tradeoffs: gate correctness
    first, then rank cost.
    """
    if not inputs.hard_needs_met or inputs.spent_cents <= 0:
        return 0.0
    utility = realized_utility(inputs)
    headroom = max(0.0, 1.0 - inputs.spent_cents / max(1, inputs.budget_cents))
    return utility * headroom


def terminal_reward(inputs: RewardInputs) -> tuple[float, dict[str, float]]:
    """The episode's verdict. Returns (scalar, per-component breakdown).

    The breakdown goes into `info` so the reward-shaping analysis is computable without
    re-running anything — the MuJoCo Ant-v5 pattern. Without it, diagnosing a bad reward
    means a full training run and a guess.
    """
    utility = realized_utility(inputs) if inputs.itinerary_complete else 0.0
    efficiency = budget_efficiency(inputs)

    damage = 0.0
    if inputs.budget_cents > 0:
        damage += inputs.disruption_damage_cents / inputs.budget_cents
    damage += inputs.forfeited_activity_value
    damage = min(damage, 2.0)  # a catastrophic trip is bounded, not unbounded

    walked = PATIENCE_FORFEIT if inputs.client_walked else 0.0

    components = {
        "reward_utility": UTILITY_SCALE * utility,
        "reward_efficiency": EFFICIENCY_SCALE * efficiency,
        "reward_disruption": -DISRUPTION_SCALE * damage,
        "reward_patience": -walked,
    }
    return float(sum(components.values())), components


def step_reward(
    *,
    prev_potential: float,
    new_potential: float,
    redundant_search: bool,
) -> tuple[float, dict[str, float]]:
    """Per-turn reward: the potential difference plus friction.

    Note there is no reward for *doing* anything here — no bonus for searching, asking, or
    booking. The only positive per-step term is the potential difference, which telescopes
    to zero over the episode and therefore cannot be farmed.
    """
    progress = new_potential - prev_potential
    friction = FRICTION_PER_TURN + (FRICTION_REDUNDANT_SEARCH if redundant_search else 0.0)
    components = {"reward_progress": progress, "reward_friction": -friction}
    return float(progress - friction), components


def zero_components() -> dict[str, float]:
    """Every component key, all zero.

    Emitted on every step whether or not the term fired. Under vectorisation, Gymnasium
    fixes an info key's dtype from the first sub-env to emit it and zero-fills absences —
    so a key that appears only sometimes produces silently wrong batched logs.
    """
    return {
        "reward_progress": 0.0,
        "reward_friction": 0.0,
        "reward_utility": 0.0,
        "reward_efficiency": 0.0,
        "reward_disruption": 0.0,
        "reward_patience": 0.0,
    }
