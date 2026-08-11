"""Hard constraints: the things a plan either satisfies or fails.

Four classes, all reading the hidden traveller manifest:

1. **Accessibility** — step-free, wheelchair, lift, service animal, low stimulus.
2. **Capacity and grouping** — enough beds, enough seats, a child seated beside an adult,
   a licensed driver for a hire car, luggage that fits.
3. **Documents** — the six-month passport rule, and visas that need more consular lead time
   than the trip allows.
4. **Fixed commitments** — handled in `timeline.py`, because being somewhere at a time is a
   clock fact rather than an eligibility fact.

**Why these are gates and not preferences.** A hotel with stairs is not a slightly worse
hotel for a wheelchair user; it is a booking that fails at the door. Modelled as utility, an
optimiser would learn to trade it against price — which is both wrong and the exact failure
mode the reward is supposed to avoid.

**The mechanic that makes elicitation matter.** A constraint the agent has *asked about* is
enforced by the action mask: violating options are not offered, so a well-informed agent
cannot fumble. A constraint it has *not* asked about is not masked — the agent can happily
book the inaccessible hotel — and it fails at travel time, in act II, with the value of the
booking lost. So the cost of not interviewing the client is not a penalty term; it is a trip
that does not work. That is the whole design in one sentence, and it is why the hand-written
heuristic (which asks only about preference axes) should fall a long way in v2.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import StrEnum

from travel_env.contracts import (
    AccessibilityNeed,
    Activity,
    ActivityCategory,
    CityCode,
    ClientPersona,
    DietaryNeed,
    DisruptionKind,
    Flight,
    GroundLeg,
    Hotel,
    RoomOffer,
    TransportMode,
    World,
)
from travel_env.timeline import Timeline, Violation

MIN_DRIVER_AGE = 21
BAGS_PER_TRAVELLER = 1


class ConstraintClass(StrEnum):
    """What the agent can ask about, and therefore what the mask can enforce."""

    ACCESSIBILITY = "accessibility"
    CAPACITY = "capacity"
    DOCUMENTS = "documents"
    COMMITMENTS = "commitments"


ASKABLE: tuple[ConstraintClass, ...] = tuple(ConstraintClass)


# ---- Party-level summaries (computed once per episode) --------------------------------


class PartyNeeds:
    """Everything the manifest implies, precomputed.

    Built once at `reset` because the mask asks these questions on every step and walking the
    manifest each time is pure overhead on the hot path.
    """

    __slots__ = (
        "access", "dietary", "size", "has_minor", "needs_adjacent", "oldest",
        "youngest", "has_driver", "nationalities", "earliest_expiry",
    )

    def __init__(self, persona: ClientPersona) -> None:
        travellers = persona.travellers
        self.size = persona.party_size
        if not travellers:
            # v1-shaped persona: no manifest, so no manifest-derived constraint can fire.
            self.access: frozenset[AccessibilityNeed] = frozenset()
            self.dietary: frozenset[DietaryNeed] = frozenset()
            self.has_minor = False
            self.needs_adjacent = False
            self.oldest = 40
            self.youngest = 40
            self.has_driver = True
            self.nationalities: frozenset = frozenset()
            self.earliest_expiry: date | None = None
            return

        self.access = frozenset(n for t in travellers for n in t.accessibility)
        self.dietary = frozenset(d for t in travellers for d in t.dietary)
        self.has_minor = any(t.is_minor for t in travellers)
        self.needs_adjacent = any(t.needs_adjacent_seat for t in travellers)
        self.oldest = max(t.age for t in travellers)
        self.youngest = min(t.age for t in travellers)
        self.has_driver = any(t.age >= MIN_DRIVER_AGE for t in travellers)
        self.nationalities = frozenset(t.nationality for t in travellers)
        self.earliest_expiry = min(t.passport_expiry for t in travellers)

    @property
    def mobility(self) -> bool:
        return bool(self.access & {AccessibilityNeed.STEP_FREE, AccessibilityNeed.WHEELCHAIR})


# ---- Per-item predicates: what the mask calls -----------------------------------------
# Each returns None when the item is fine, or a short human-readable reason. The reason is
# for diagnostics and test failures; the agent only ever sees the mask bit.


def lodging_blocked(needs: PartyNeeds, hotel: Hotel, offer: RoomOffer, units: int) -> str | None:
    required = needs.access & {
        AccessibilityNeed.STEP_FREE, AccessibilityNeed.WHEELCHAIR, AccessibilityNeed.ELEVATOR
    }
    missing = required - hotel.accessibility
    if missing:
        return f"{hotel.name} lacks {', '.join(sorted(n.value for n in missing))}"
    if AccessibilityNeed.SERVICE_ANIMAL in needs.access and not hotel.allows_service_animal:
        return f"{hotel.name} does not accept service animals"

    sleeps = units * (offer.max_occupancy + (1 if offer.cot_available and needs.has_minor else 0))
    if sleeps < needs.size:
        return f"sleeps {sleeps}, party is {needs.size}"
    if units > offer.rooms_remaining:
        return f"only {offer.rooms_remaining} units left, needs {units}"
    return None


def units_needed(needs: PartyNeeds, offer: RoomOffer) -> int:
    """How many units of this offer the party requires.

    A cot adds one head and only helps if there is a child to put in it, which is what makes
    an apartment sleeping six genuinely cheaper than three hotel rooms — and only for the
    parties it actually fits.
    """
    per_unit = offer.max_occupancy + (1 if offer.cot_available and needs.has_minor else 0)
    return max(1, -(-needs.size // max(1, per_unit)))


def activity_blocked(needs: PartyNeeds, activity: Activity, on: date) -> str | None:
    if needs.mobility and not activity.accessible:
        return f"{activity.name} is not step-free"
    if AccessibilityNeed.LOW_STIMULUS in needs.access and not activity.low_stimulus:
        return f"{activity.name} is loud or crowded"
    if needs.youngest < activity.min_age:
        return f"{activity.name} has a minimum age of {activity.min_age}"
    if needs.oldest > activity.max_age:
        return f"{activity.name} has a maximum age of {activity.max_age}"
    if needs.size > activity.max_group_size:
        return f"{activity.name} takes at most {activity.max_group_size}"
    if on.month not in activity.season_months:
        return f"{activity.name} does not run in month {on.month}"
    if on.isoweekday() in activity.closed_weekdays:
        return f"{activity.name} is closed that weekday"
    return None


def flight_blocked(needs: PartyNeeds, flight: Flight) -> str | None:
    if flight.seats_remaining < needs.size:
        return f"{flight.seats_remaining} seats left, party is {needs.size}"
    if AccessibilityNeed.WHEELCHAIR in needs.access and not flight.wheelchair_service:
        return f"{flight.carrier}{flight.flight_number} has no wheelchair service"
    # An unaccompanied under-12 is not a seating preference; carriers refuse it.
    if needs.needs_adjacent and flight.adjacent_seats_available < needs.size:
        return (
            f"{flight.adjacent_seats_available} adjacent seats for a party of {needs.size} "
            "travelling with a young child"
        )
    return None


def ground_blocked(needs: PartyNeeds, leg: GroundLeg) -> str | None:
    if leg.seats_remaining < needs.size:
        return f"{leg.seats_remaining} seats left, party is {needs.size}"
    if needs.mobility and not leg.step_free:
        return f"{leg.operator} {leg.mode.value} is not step-free"
    if leg.requires_driver and not needs.has_driver:
        return "nobody in the party is old enough to drive"
    if leg.luggage_allowance < BAGS_PER_TRAVELLER:
        return f"{leg.operator} carries no luggage"
    return None


def documents_blocked(
    needs: PartyNeeds, world: World, *, departure: datetime, return_date: date
) -> str | None:
    """Passport validity and visa lead time, for every country in this world.

    A whole-destination gate: unlike a hotel it cannot be worked around by choosing something
    else, so an agent that discovers it late has already wasted the episode. That is the point
    — real agencies ask about passports in the first exchange.
    """
    if not needs.nationalities or needs.earliest_expiry is None:
        return None

    countries = {city.country for city in world.cities}
    rules = {(r.nationality, r.country): r for r in world.visa_rules}
    days_to_departure = (departure.date() - _today_of(world)).days

    for country in sorted(countries):
        for nationality in sorted(needs.nationalities):
            rule = rules.get((nationality, country))
            if rule is None:
                continue
            if needs.earliest_expiry < return_date + timedelta(days=30 * rule.passport_validity_months):
                return (
                    f"a passport expires {needs.earliest_expiry:%Y-%m-%d}, inside {country}'s "
                    f"{rule.passport_validity_months}-month rule after the return"
                )
            if rule.visa_required and rule.lead_time_days > days_to_departure:
                return (
                    f"{nationality.value} needs a {country} visa with {rule.lead_time_days} days "
                    f"of lead time and departure is in {days_to_departure}"
                )
    return None


def _today_of(world: World) -> date:
    """The world's "now" — when the booking is being made.

    Derived from the clock window rather than from `datetime.now()`, so the same seed prices
    and gates identically forever. A wall-clock read here would make visa lead times drift
    from run to run, which is the determinism bug v1 shipped in `generated_at`.
    """
    return (world.window_start_utc or datetime(2027, 1, 1)).date()


def region_visa_summary(needs: PartyNeeds, world: World) -> tuple[int, int]:
    """(countries requiring a visa, worst lead time in days) for this party. Observation-safe."""
    rules = {(r.nationality, r.country): r for r in world.visa_rules}
    countries = {city.country for city in world.cities}
    required, worst = 0, 0
    for country in countries:
        needed = False
        for nationality in needs.nationalities:
            rule = rules.get((nationality, country))
            if rule is not None and rule.visa_required:
                needed = True
                worst = max(worst, rule.lead_time_days)
        required += int(needed)
    return required, worst


# ---- Whole-plan audit: what the reward and act II read ---------------------------------


def audit(
    persona: ClientPersona,
    needs: PartyNeeds,
    timeline: Timeline,
    world: World,
    *,
    departure: datetime,
    return_date: date,
) -> list[Violation]:
    """Every hard constraint this booked plan actually violates.

    Called at booking and again at travel time. Returns; never raises. A violation here means
    the trip fails on that component — the party could not use the room, could not board the
    plane, could not enter the country — so the value is lost and the money often is too.
    """
    found: list[Violation] = []

    document_fault = documents_blocked(needs, world, departure=departure, return_date=return_date)
    if document_fault:
        found.append(
            Violation(kind=DisruptionKind.DENIED_BOARDING, segment_key="documents", detail=document_fault)
        )

    for flight, label in ((timeline.outbound, "outbound"), (timeline.inbound, "return")):
        if flight is None:
            continue
        fault = flight_blocked(needs, flight)
        if fault:
            kind = (
                DisruptionKind.ACCESS_FAILURE
                if "wheelchair" in fault
                else DisruptionKind.CAPACITY_FAILURE
            )
            found.append(Violation(kind=kind, segment_key=f"flight_{label}", detail=fault))

    for n, leg in enumerate(timeline.ground):
        fault = ground_blocked(needs, leg)
        if fault:
            kind = (
                DisruptionKind.ACCESS_FAILURE
                if "step-free" in fault
                else DisruptionKind.CAPACITY_FAILURE
            )
            found.append(Violation(kind=kind, segment_key=f"seg_ground_{n}", detail=fault))

    for stay in timeline.stays:
        fault = lodging_blocked(needs, stay.hotel, stay.offer, stay.units)
        if fault:
            kind = (
                DisruptionKind.CAPACITY_FAILURE if "sleeps" in fault or "units" in fault
                else DisruptionKind.ACCESS_FAILURE
            )
            found.append(Violation(kind=kind, segment_key=f"stay_{stay.hotel.key}", detail=fault))

    for n, (activity, start) in enumerate(timeline.activities):
        offset = timeline.resolver.offset(activity.city)
        local_day = (start + timedelta(minutes=offset)).date()
        fault = activity_blocked(needs, activity, local_day)
        if fault:
            found.append(
                Violation(kind=DisruptionKind.ACCESS_FAILURE, segment_key=f"seg_act_{n}", detail=fault)
            )

    return found


def dietary_shortfall(needs: PartyNeeds, timeline: Timeline) -> float:
    """Share of booked food experiences the party cannot actually eat, in [0, 1].

    Deliberately NOT a hard gate. A restaurant that cannot do gluten-free is a disappointing
    evening, not a trip that fails, so this reduces realized food utility instead of blocking
    the booking. Getting that line right matters: gating everything would make the
    environment brittle rather than hard.
    """
    if not needs.dietary:
        return 0.0
    food = [a for a, _ in timeline.activities if a.category is ActivityCategory.FOOD]
    if not food:
        return 0.0
    unmet = sum(1 for a in food if needs.dietary - a.dietary_supported)
    return unmet / len(food)


def party_cities(timeline: Timeline) -> set[CityCode]:
    return {stay.hotel.city for stay in timeline.stays}


def ancillary_cost(needs: PartyNeeds, timeline: Timeline, world: World) -> int:
    """Money the headline prices do not include: cots, checked bags, visa fees.

    Real, and routinely ignored — which is why an agent that plans to 100% of budget on
    headline fares comes in over. The ledger charges these at booking.
    """
    total = 0
    for stay in timeline.stays:
        if stay.with_cot and stay.offer.cot_available:
            total += stay.offer.cot_price_per_night * stay.nights * stay.units
    for flight in (timeline.outbound, timeline.inbound):
        if flight is not None and not flight.checked_bag_included:
            total += flight.checked_bag_fee * needs.size
    rules = {(r.nationality, r.country): r for r in world.visa_rules}
    for country in {city.country for city in world.cities}:
        for nationality in needs.nationalities:
            rule = rules.get((nationality, country))
            if rule is not None and rule.visa_required:
                total += rule.fee_cents * sum(
                    1 for t in timeline_travellers(needs) if t == nationality
                )
    return total


def timeline_travellers(needs: PartyNeeds) -> list:
    """Nationalities, one entry per traveller, for per-head fee arithmetic."""
    return list(needs.nationalities)


def mode_reliability_bonus(timeline: Timeline) -> float:
    """Rail and through-ticketing are robustness purchases; self-transfer is the opposite.

    Read by the reliability axis of the realized-feature vector, so an agent serving a
    reliability-weighted client has a concrete lever beyond picking a punctual carrier.
    """
    bonus = 0.0
    for leg in timeline.ground:
        if leg.mode is TransportMode.RAIL:
            bonus += 0.08
        elif leg.mode in (TransportMode.FERRY, TransportMode.BUS):
            bonus -= 0.05
    for flight in (timeline.outbound, timeline.inbound):
        if flight is not None and flight.is_self_transfer:
            bonus -= 0.18
    return bonus
