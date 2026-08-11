"""Client persona generation.

The persona is where the episode's difficulty actually lives. Two things make it more
than a random draw:

* **The utility weights are hidden.** They never enter an observation. The reward is a
  dot product against them, so an agent that books without eliciting them is betting on
  its prior. This is what makes asking questions instrumentally necessary rather than
  merely instructed.
* **Demographics are correlated with behaviour, not decorative.** Age drives patience,
  tech-savviness and register; party type shifts the weight priors. A 68-year-old formal
  business traveller and a 24-year-old terse backpacker produce structurally different
  episodes, not the same episode with different prose.

Draw counts: 8 axes x 10 levels, times 4 age bands, 5 party types, 4 registers, and the
continuous parameters — the persona space is combinatorially large by construction, so a
policy cannot memorise the client distribution.
"""

from __future__ import annotations

import numpy as np

from travel_env.contracts import (
    AgeBand,
    ClientPersona,
    CommunicationStyle,
    PartyType,
    PreferenceAxis,
)

# Weight priors per party type, as a mean level on the 0-9 scale. These encode ordinary
# social facts: families optimise pacing and cost, business travellers buy reliability,
# friends chase novelty. Noise is added on top so no two personas of a type are alike.
_PARTY_PRIORS: dict[PartyType, dict[PreferenceAxis, float]] = {
    PartyType.SOLO: {
        PreferenceAxis.COST: 6.0, PreferenceAxis.COMFORT: 3.5, PreferenceAxis.LOCATION: 6.0,
        PreferenceAxis.PACING: 4.0, PreferenceAxis.FOOD: 5.5, PreferenceAxis.CULTURE: 6.0,
        PreferenceAxis.NOVELTY: 7.0, PreferenceAxis.RELIABILITY: 4.0,
    },
    PartyType.COUPLE: {
        PreferenceAxis.COST: 4.5, PreferenceAxis.COMFORT: 6.5, PreferenceAxis.LOCATION: 6.5,
        PreferenceAxis.PACING: 5.0, PreferenceAxis.FOOD: 7.0, PreferenceAxis.CULTURE: 6.0,
        PreferenceAxis.NOVELTY: 5.5, PreferenceAxis.RELIABILITY: 5.5,
    },
    PartyType.FRIENDS: {
        PreferenceAxis.COST: 7.0, PreferenceAxis.COMFORT: 3.0, PreferenceAxis.LOCATION: 7.0,
        PreferenceAxis.PACING: 3.0, PreferenceAxis.FOOD: 6.0, PreferenceAxis.CULTURE: 4.0,
        PreferenceAxis.NOVELTY: 7.5, PreferenceAxis.RELIABILITY: 3.5,
    },
    PartyType.FAMILY_WITH_KIDS: {
        PreferenceAxis.COST: 6.0, PreferenceAxis.COMFORT: 6.0, PreferenceAxis.LOCATION: 5.5,
        PreferenceAxis.PACING: 7.5, PreferenceAxis.FOOD: 4.5, PreferenceAxis.CULTURE: 4.5,
        PreferenceAxis.NOVELTY: 3.0, PreferenceAxis.RELIABILITY: 7.5,
    },
    PartyType.BUSINESS: {
        PreferenceAxis.COST: 2.5, PreferenceAxis.COMFORT: 7.5, PreferenceAxis.LOCATION: 8.0,
        PreferenceAxis.PACING: 2.5, PreferenceAxis.FOOD: 5.0, PreferenceAxis.CULTURE: 2.5,
        PreferenceAxis.NOVELTY: 2.0, PreferenceAxis.RELIABILITY: 9.0,
    },
}

# Age shifts applied on top of the party prior. Older travellers weight comfort and
# reliability up and novelty down; the young trade comfort for cost.
_AGE_SHIFTS: dict[AgeBand, dict[PreferenceAxis, float]] = {
    AgeBand.YOUNG_ADULT: {
        PreferenceAxis.COST: +1.5, PreferenceAxis.COMFORT: -1.5, PreferenceAxis.NOVELTY: +1.5,
        PreferenceAxis.RELIABILITY: -1.0,
    },
    AgeBand.ADULT: {},
    AgeBand.MIDLIFE: {
        PreferenceAxis.COMFORT: +1.0, PreferenceAxis.RELIABILITY: +1.0,
        PreferenceAxis.NOVELTY: -0.5,
    },
    AgeBand.SENIOR: {
        PreferenceAxis.COMFORT: +2.0, PreferenceAxis.RELIABILITY: +2.0,
        PreferenceAxis.NOVELTY: -2.0, PreferenceAxis.PACING: +2.0,
    },
}

_PARTY_SIZE_RANGE: dict[PartyType, tuple[int, int]] = {
    PartyType.SOLO: (1, 1),
    PartyType.COUPLE: (2, 2),
    PartyType.FRIENDS: (2, 5),
    PartyType.FAMILY_WITH_KIDS: (3, 6),
    PartyType.BUSINESS: (1, 2),
}

# Register skews young->casual, old->formal, but never deterministically: a terse senior
# and a chatty 25-year-old both exist.
_STYLE_WEIGHTS: dict[AgeBand, dict[CommunicationStyle, float]] = {
    AgeBand.YOUNG_ADULT: {
        CommunicationStyle.TERSE: 0.40, CommunicationStyle.CASUAL: 0.42,
        CommunicationStyle.FORMAL: 0.06, CommunicationStyle.CHATTY: 0.12,
    },
    AgeBand.ADULT: {
        CommunicationStyle.TERSE: 0.34, CommunicationStyle.CASUAL: 0.32,
        CommunicationStyle.FORMAL: 0.20, CommunicationStyle.CHATTY: 0.14,
    },
    AgeBand.MIDLIFE: {
        CommunicationStyle.TERSE: 0.28, CommunicationStyle.CASUAL: 0.22,
        CommunicationStyle.FORMAL: 0.34, CommunicationStyle.CHATTY: 0.16,
    },
    AgeBand.SENIOR: {
        CommunicationStyle.TERSE: 0.20, CommunicationStyle.CASUAL: 0.12,
        CommunicationStyle.FORMAL: 0.48, CommunicationStyle.CHATTY: 0.20,
    },
}

_BASE_BUDGET_PER_PERSON_CENTS = 180_000
"""~$1,800pp for a week in Japan — mid-market, before party-type and noise adjustment."""


def _clamp_level(value: float) -> int:
    return int(np.clip(round(value), 0, 9))


def draw_persona(rng: np.random.Generator, *, difficulty: float = 0.5) -> ClientPersona:
    """Draw a client.

    `difficulty` in [0, 1] tightens the episode without changing its shape: it shrinks the
    budget toward what the trip actually costs, shortens patience, reduces how much the
    opening request reveals, and shortens replies. It is the curriculum knob — the same
    generator produces the easy and the hard end of the distribution.
    """
    age_band = AgeBand(rng.choice(list(AgeBand)))
    party_type = PartyType(rng.choice(list(PartyType)))
    low, high = _PARTY_SIZE_RANGE[party_type]
    party_size = int(rng.integers(low, high + 1))

    styles = list(_STYLE_WEIGHTS[age_band])
    probabilities = np.array([_STYLE_WEIGHTS[age_band][s] for s in styles], dtype=float)
    communication_style = styles[int(rng.choice(len(styles), p=probabilities / probabilities.sum()))]

    prior = _PARTY_PRIORS[party_type]
    shifts = _AGE_SHIFTS[age_band]
    weights = {
        axis: _clamp_level(prior[axis] + shifts.get(axis, 0.0) + rng.normal(0.0, 1.6))
        for axis in PreferenceAxis
    }

    # What the opening request gives away. Harder episodes reveal less; a client never
    # volunteers everything, which is the whole reason the agent has to interview them.
    revealed = max(1, int(round(3 - 2 * difficulty)))
    top_axes = sorted(weights, key=lambda a: weights[a], reverse=True)[:4]
    stated = frozenset(rng.choice(top_axes, size=min(revealed, len(top_axes)), replace=False))

    budget_noise = float(rng.normal(1.0, 0.22))
    party_multiplier = {  # business travel is expensed; friends split and go cheap
        PartyType.BUSINESS: 1.9, PartyType.COUPLE: 1.15, PartyType.SOLO: 1.0,
        PartyType.FRIENDS: 0.8, PartyType.FAMILY_WITH_KIDS: 0.85,
    }[party_type]
    generosity = 1.35 - 0.55 * difficulty
    budget = int(
        _BASE_BUDGET_PER_PERSON_CENTS
        * party_size
        * party_multiplier
        * max(0.45, budget_noise)
        * generosity
    )

    # Older clients run out of patience sooner (your call, and it matches the way people
    # actually behave with a slow agent). Tech-savviness moves the other way.
    age_patience_offset = {
        AgeBand.YOUNG_ADULT: +6, AgeBand.ADULT: +3, AgeBand.MIDLIFE: 0, AgeBand.SENIOR: -4,
    }[age_band]
    # Calibrated against the shortest competent trajectory (interview ~5, search ~3,
    # select ~3, propose/book/finalize 3 = ~14 turns). A hard episode must still be
    # winnable by a good agent, or the environment sits in the no-signal band.
    initial_patience = max(
        16, int(round(34 + age_patience_offset - 8 * difficulty + rng.normal(0, 2)))
    )

    tech_savvy = _clamp_level(
        {AgeBand.YOUNG_ADULT: 8.0, AgeBand.ADULT: 6.5, AgeBand.MIDLIFE: 4.5, AgeBand.SENIOR: 2.5}[age_band]
        + rng.normal(0, 1.4)
    )

    # Most real clients answer in a few words. Chatty and formal registers say more;
    # difficulty pushes everyone toward monosyllables.
    verbosity_base = {
        CommunicationStyle.TERSE: 1.0, CommunicationStyle.CASUAL: 3.0,
        CommunicationStyle.FORMAL: 4.5, CommunicationStyle.CHATTY: 7.0,
    }[communication_style]
    reply_verbosity = _clamp_level(verbosity_base - 2.5 * difficulty + rng.normal(0, 1.0))

    return ClientPersona(
        key=f"client_{int(rng.integers(0, 1_000_000)):06d}",
        age_band=age_band,
        party_type=party_type,
        party_size=party_size,
        communication_style=communication_style,
        weights=weights,
        stated_axes=stated,
        budget_cents=budget,
        budget_flexibility=float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0)),
        initial_patience=initial_patience,
        patience_decay_per_turn=float(np.clip(1.0 + rng.normal(0.0, 0.25) + 0.6 * difficulty, 0.4, 3.0)),
        tech_savvy=tech_savvy,
        risk_tolerance=_clamp_level(rng.normal(4.5, 2.0)),
        reply_verbosity=reply_verbosity,
    )


def normalized_weights(persona: ClientPersona) -> dict[PreferenceAxis, float]:
    """Latent weights as a simplex. The reward is a dot product against this, so it is
    scale-free: a client who cares intensely about everything is not worth more points
    than one with a single strong preference."""
    total = sum(persona.weights.values()) or 1
    return {axis: level / total for axis, level in persona.weights.items()}


def opening_request(persona: ClientPersona, nights: int) -> str:
    """The first message. Deliberately underspecified — it names the destination, the
    party, the dates and at most a couple of preference axes, because that is what a real
    inbound client email contains."""
    axis_phrases = {
        PreferenceAxis.COST: "keep it affordable",
        PreferenceAxis.COMFORT: "somewhere comfortable",
        PreferenceAxis.LOCATION: "central, easy to get around",
        PreferenceAxis.PACING: "not too packed a schedule",
        PreferenceAxis.FOOD: "we're into food",
        PreferenceAxis.CULTURE: "historical sites matter to us",
        PreferenceAxis.NOVELTY: "want to see something off the beaten path",
        PreferenceAxis.RELIABILITY: "nothing that risks going wrong",
    }
    party = {
        PartyType.SOLO: "just me", PartyType.COUPLE: "two of us",
        PartyType.FRIENDS: f"{persona.party_size} of us",
        PartyType.FAMILY_WITH_KIDS: f"family of {persona.party_size}, kids included",
        PartyType.BUSINESS: "business trip",
    }[persona.party_type]
    wants = ", ".join(axis_phrases[a] for a in sorted(persona.stated_axes))
    budget = f"~${persona.budget_cents // 100:,}"
    return f"Japan, {nights} nights, {party}. Budget {budget}. {wants.capitalize()}."
