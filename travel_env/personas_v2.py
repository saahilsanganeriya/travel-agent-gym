"""v2 clients: a manifest of people, not a headcount.

v1's persona had `party_size: int`. That cannot express "four travellers, one of them is
seven and one uses a wheelchair", and those are exactly the facts that decide whether a plan
is bookable at all. So the party becomes a tuple of `Traveller`, each with an age, a
passport, and their own access and dietary needs.

**Why this makes the environment harder in the right way.** Every hard gate in
`constraints.py` reads the manifest, and the manifest is *hidden* — the opening request says
"family of four", not "one of us needs step-free access". An agent that does not ask books a
hotel the party cannot use, and finds out at check-in. That is real travel-agent work, it
rewards interviewing rather than searching, and no penalty term is involved: the trip simply
fails.

**Budget is calibrated against the world, not drawn in a vacuum.** Three regions differ in
price level by a factor of two and a half, so a fixed dollar budget would make Thailand
trivial and Japan impossible. Instead the floor cost of a *minimum viable trip in this
particular world* is measured, and the budget is a multiple of it. Difficulty then squeezes
that multiple. This is the feasibility gate the design doc called for, in its cheapest
useful form: every episode is winnable, and a hard one is only just.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np

from travel_env.contracts import (
    AccessibilityNeed,
    AgeBand,
    CityCode,
    ClientPersona,
    CommunicationStyle,
    DietaryNeed,
    FixedCommitment,
    Nationality,
    PartyType,
    PreferenceAxis,
    Traveller,
    World,
)
from travel_env.personas import _AGE_SHIFTS, _PARTY_PRIORS, _STYLE_WEIGHTS
from travel_env.regions import NATIONALITY_WEIGHTS

# (adults low, adults high, children low, children high) per party type.
_COMPOSITION: dict[PartyType, tuple[int, int, int, int]] = {
    PartyType.SOLO: (1, 1, 0, 0),
    PartyType.COUPLE: (2, 2, 0, 0),
    PartyType.FRIENDS: (2, 4, 0, 0),
    PartyType.FAMILY_WITH_KIDS: (2, 2, 1, 3),
    PartyType.BUSINESS: (1, 2, 0, 0),
}

_ADULT_AGE_RANGE: dict[AgeBand, tuple[int, int]] = {
    AgeBand.YOUNG_ADULT: (18, 29),
    AgeBand.ADULT: (30, 44),
    AgeBand.MIDLIFE: (45, 59),
    AgeBand.SENIOR: (60, 82),
}

# Base rate that a *party* contains at least one person with an access need. Roughly one in
# four, which is not far off the real population share once you count temporary and sensory
# needs. High enough that an agent which never asks is reliably punished, low enough that
# asking is a judgement call rather than a rote opener.
_ACCESS_PARTY_RATE = 0.26
_DIET_PARTY_RATE = 0.38

_COMMITMENT_LABELS = (
    "a wedding they are attending",
    "a conference keynote",
    "a family reunion lunch",
    "a client meeting",
    "a concert with booked tickets",
)


def _clamp_level(value: float) -> int:
    return int(min(9, max(0, round(value))))


def estimate_floor_cost(world: World, *, nights: int, party_size: int) -> int:
    """Roughly what the cheapest viable trip in THIS world costs, in cents.

    Cheapest return flight for everyone, plus enough of the cheapest beds for everyone, for
    every night. Deliberately an underestimate of a *good* trip and an overestimate of
    nothing — it is a scale, not a target, and the budget is a multiple of it.

    Without this the budget would be a fixed dollar figure against a world whose price level
    varies by 2.5x across regions, which would make one region trivial and another unwinnable.
    """
    if not world.flights or not world.room_offers:
        return 100_000

    origin = world.origin_airport
    out_legs = [f.price for f in world.flights if f.origin == origin and f.seats_remaining > 0]
    home_legs = [f.price for f in world.flights if f.destination == origin and f.seats_remaining > 0]
    air = (min(out_legs) if out_legs else 60_000) + (min(home_legs) if home_legs else 60_000)

    # Cheapest way to sleep `party_size` heads: the best price-per-head across all offers,
    # which is how an apartment beats three hotel rooms.
    per_head = min(
        (o.price_per_night / max(1, o.max_occupancy) for o in world.room_offers if o.rooms_remaining > 0),
        default=4_000,
    )
    beds = int(per_head * party_size * nights)
    return int(air * party_size + beds)


def draw_persona(
    rng: np.random.Generator,
    world: World,
    *,
    nights: int,
    trip_start: datetime,
    difficulty: float = 0.5,
) -> ClientPersona:
    """Draw a v2 client: hidden weights, a hidden manifest, and a world-calibrated budget."""
    age_band = AgeBand(rng.choice(list(AgeBand)))
    party_type = PartyType(rng.choice(list(PartyType)))
    travellers = _draw_manifest(rng, party_type, age_band, trip_start, nights, difficulty)
    party_size = len(travellers)

    styles = list(_STYLE_WEIGHTS[age_band])
    probabilities = np.array([_STYLE_WEIGHTS[age_band][s] for s in styles], dtype=float)
    communication_style = styles[int(rng.choice(len(styles), p=probabilities / probabilities.sum()))]

    prior = _PARTY_PRIORS[party_type]
    shifts = _AGE_SHIFTS[age_band]
    weights = {
        axis: _clamp_level(prior[axis] + shifts.get(axis, 0.0) + rng.normal(0.0, 1.6))
        for axis in PreferenceAxis
    }

    revealed = max(1, int(round(3 - 2 * difficulty)))
    top_axes = sorted(weights, key=lambda a: weights[a], reverse=True)[:4]
    stated = frozenset(rng.choice(top_axes, size=min(revealed, len(top_axes)), replace=False))

    # Budget as a multiple of what this world actually charges. 1.9x the bare floor at
    # difficulty 0 leaves room for a comfortable trip; 1.15x at difficulty 1 barely covers a
    # viable one, so every choice has to earn its place.
    floor = estimate_floor_cost(world, nights=nights, party_size=party_size)
    generosity = (1.90 - 0.75 * difficulty) * float(max(0.72, rng.normal(1.0, 0.16)))
    party_multiplier = {
        PartyType.BUSINESS: 1.55, PartyType.COUPLE: 1.10, PartyType.SOLO: 1.0,
        PartyType.FRIENDS: 0.92, PartyType.FAMILY_WITH_KIDS: 0.95,
    }[party_type]
    budget = int(floor * generosity * party_multiplier)

    age_patience_offset = {
        AgeBand.YOUNG_ADULT: +6, AgeBand.ADULT: +3, AgeBand.MIDLIFE: 0, AgeBand.SENIOR: -4,
    }[age_band]
    # MEASURED, not guessed. The first v2 calibration used 56 and every policy walked out:
    # random 97%, heuristic 90%. A v2 trajectory is ~46 turns (route, four constraint
    # questions, two flights, ground hops, lodging per city, activities, propose/book/close)
    # against v1's ~14, so patience has to cover it or the environment sits in the
    # no-signal band for a reason that has nothing to do with skill.
    initial_patience = max(
        60, int(round(112 + 2 * age_patience_offset - 26 * difficulty + rng.normal(0, 5)))
    )

    tech_savvy = _clamp_level(
        {AgeBand.YOUNG_ADULT: 8.0, AgeBand.ADULT: 6.5, AgeBand.MIDLIFE: 4.5, AgeBand.SENIOR: 2.5}[age_band]
        + rng.normal(0, 1.4)
    )
    verbosity_base = {
        CommunicationStyle.TERSE: 1.0, CommunicationStyle.CASUAL: 3.0,
        CommunicationStyle.FORMAL: 4.5, CommunicationStyle.CHATTY: 7.0,
    }[communication_style]

    return ClientPersona(
        key=f"client_{int(rng.integers(0, 1_000_000)):06d}",
        age_band=age_band,
        party_type=party_type,
        party_size=party_size,
        communication_style=communication_style,
        weights=weights,
        stated_axes=stated,
        budget_cents=budget,
        budget_flexibility=float(min(1.0, max(0.0, rng.beta(2.0, 3.0)))),
        initial_patience=initial_patience,
        patience_decay_per_turn=float(min(2.2, max(0.5, 1.0 + rng.normal(0.0, 0.2) + 0.45 * difficulty))),
        tech_savvy=tech_savvy,
        risk_tolerance=_clamp_level(rng.normal(4.5, 2.0)),
        reply_verbosity=_clamp_level(verbosity_base - 2.5 * difficulty + rng.normal(0, 1.0)),
        travellers=travellers,
        fixed_commitments=_draw_commitments(rng, world, party_type, trip_start, nights, difficulty),
        home_region="US",
    )


def _draw_manifest(
    rng: np.random.Generator,
    party_type: PartyType,
    age_band: AgeBand,
    trip_start: datetime,
    nights: int,
    difficulty: float,
) -> tuple[Traveller, ...]:
    """Build the actual people.

    Nationality is mostly shared — a family carries one country's passports — but not always,
    which is what makes a mixed-nationality party a real (and realistic) visa problem.
    """
    adults_lo, adults_hi, kids_lo, kids_hi = _COMPOSITION[party_type]
    n_adults = int(rng.integers(adults_lo, adults_hi + 1))
    n_kids = int(rng.integers(kids_lo, kids_hi + 1)) if kids_hi else 0

    nations = list(NATIONALITY_WEIGHTS)
    nation_p = np.array([NATIONALITY_WEIGHTS[n] for n in nations], dtype=float)
    nation_p /= nation_p.sum()
    primary = nations[int(rng.choice(len(nations), p=nation_p))]

    return_date = (trip_start + timedelta(days=nights)).date()
    party_has_access_need = rng.random() < (_ACCESS_PARTY_RATE + 0.14 * difficulty)
    party_has_diet = rng.random() < _DIET_PARTY_RATE

    people: list[Traveller] = []
    for i in range(n_adults + n_kids):
        is_kid = i >= n_adults
        if is_kid:
            age = int(rng.integers(1, 16))
        else:
            low, high = _ADULT_AGE_RANGE[age_band] if i == 0 else _ADULT_AGE_RANGE[
                AgeBand(rng.choice(list(AgeBand)))
            ]
            age = int(rng.integers(low, high + 1))

        # Mostly one passport country per party; occasionally not, which is exactly the case
        # where one traveller needs a visa and the rest do not.
        nationality = primary if rng.random() < 0.86 else nations[int(rng.choice(len(nations), p=nation_p))]

        # The six-month rule. Most passports are comfortably valid; a minority expire just
        # inside the window, which is a document failure an agent can only catch by asking.
        roll = rng.random()
        if roll < 0.08 + 0.06 * difficulty:
            months = float(rng.uniform(0.5, 5.5))  # short — this trip is not legal on it
        elif roll < 0.22:
            months = float(rng.uniform(6.5, 14.0))
        else:
            months = float(rng.uniform(18.0, 108.0))
        expiry = return_date + timedelta(days=int(months * 30.4))

        access: set[AccessibilityNeed] = set()
        if party_has_access_need and (i == 0 or rng.random() < 0.3):
            senior = age >= 68
            pool = [
                (AccessibilityNeed.STEP_FREE, 0.55 if senior else 0.34),
                (AccessibilityNeed.WHEELCHAIR, 0.30 if senior else 0.20),
                (AccessibilityNeed.ELEVATOR, 0.42),
                (AccessibilityNeed.SERVICE_ANIMAL, 0.10),
                (AccessibilityNeed.LOW_STIMULUS, 0.22 if age < 16 else 0.12),
            ]
            for need, p in pool:
                if rng.random() < p:
                    access.add(need)
            # A wheelchair user needs step-free access by definition. Letting the draw emit
            # one without the other would make the constraint incoherent.
            if AccessibilityNeed.WHEELCHAIR in access:
                access.add(AccessibilityNeed.STEP_FREE)
            if not access:
                access.add(AccessibilityNeed.STEP_FREE)

        dietary: set[DietaryNeed] = set()
        if party_has_diet and rng.random() < 0.6:
            dietary.add(DietaryNeed(rng.choice(list(DietaryNeed))))

        people.append(
            Traveller(
                key=f"t{i}",
                age=age,
                nationality=nationality,
                passport_expiry=expiry,
                accessibility=frozenset(access),
                dietary=frozenset(dietary),
            )
        )
    return tuple(people)


def _draw_commitments(
    rng: np.random.Generator,
    world: World,
    party_type: PartyType,
    trip_start: datetime,
    nights: int,
    difficulty: float,
) -> tuple[FixedCommitment, ...]:
    """An immovable point, sometimes.

    A commitment turns an open planning problem into a constrained one: the party must be in
    a named city inside a named window, so the route, the intercity legs and the day's
    activities all have to be arranged around it. It fires more often for business travel,
    because that is when it is true.
    """
    rate = 0.22 + 0.30 * difficulty + (0.30 if party_type is PartyType.BUSINESS else 0.0)
    if rng.random() > min(0.75, rate) or not world.cities:
        return ()

    city: CityCode = world.cities[int(rng.integers(0, len(world.cities)))].code
    offset = next(c.utc_offset_minutes for c in world.cities if c.code is city)
    # Never on the first or last day: a commitment on the arrival day is decided by the
    # long-haul schedule rather than by planning, which is luck rather than skill.
    day = int(rng.integers(1, max(2, nights - 1)))
    hour = int(rng.choice([10, 11, 14, 15, 16, 19]))
    local = (trip_start.replace(tzinfo=None) + timedelta(days=day)).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(hours=hour)
    start = (local - timedelta(minutes=offset)).replace(tzinfo=UTC)
    return (
        FixedCommitment(
            key=f"fix_{day}",
            label=str(rng.choice(_COMMITMENT_LABELS)),
            city=city,
            start_utc=start,
            end_utc=start + timedelta(hours=int(rng.integers(2, 5))),
        ),
    )


def opening_request(persona: ClientPersona, world: World, nights: int) -> str:
    """The first message. Underspecified on purpose — it names the destination, the party,
    the dates and at most a couple of preference axes, because that is what a real inbound
    client email contains. It never mentions access needs, passports or the manifest."""
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
        PartyType.SOLO: "just me",
        PartyType.COUPLE: "two of us",
        PartyType.FRIENDS: f"{persona.party_size} of us",
        PartyType.FAMILY_WITH_KIDS: f"family of {persona.party_size}, kids included",
        PartyType.BUSINESS: "business trip",
    }[persona.party_type]
    where = " and ".join(r.value.title() for r in world.regions)
    wants = ", ".join(axis_phrases[a] for a in sorted(persona.stated_axes))
    budget = f"~${persona.budget_cents // 100:,}"
    return f"{where}, {nights} nights, {party}. Budget {budget}. {wants.capitalize()}."


def describe_manifest(persona: ClientPersona) -> str:
    """Human-readable manifest, for `render()` and for the LLM client's private context.

    Never enters an observation — the whole point is that the agent has to ask.
    """
    parts = []
    for traveller in persona.travellers:
        bits = [f"{traveller.age}y {traveller.nationality.value}"]
        if traveller.accessibility:
            bits.append("/".join(sorted(n.value for n in traveller.accessibility)))
        if traveller.dietary:
            bits.append("/".join(sorted(d.value for d in traveller.dietary)))
        parts.append(" ".join(bits))
    return "; ".join(parts)
