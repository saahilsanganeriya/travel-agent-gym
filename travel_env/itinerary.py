"""The itinerary under construction, and the invariants that make incoherence impossible.

Everything in this module exists to answer one question: *can this item be added to this
plan?* If the answer is no, the corresponding action is masked out — it is not penalised,
it is not available. That is the difference between an environment that teaches the shape
of the work and one that teaches an optimiser to price a penalty.

The temporal and spatial invariants (T4-T6, S1-S2 in the design doc) all live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from travel_env.contracts import (
    Activity,
    CityCode,
    District,
    Flight,
    Hotel,
    Key,
    PreferenceAxis,
    RailLeg,
    RoomOffer,
)
from travel_env.world import transfer_minutes

MAX_ACTIVITIES = 6


@dataclass
class ActivitySlot:
    activity: Activity
    trip_day: int
    start_utc: datetime

    @property
    def end_utc(self) -> datetime:
        return self.start_utc + timedelta(minutes=self.activity.duration_minutes)


@dataclass
class Itinerary:
    """Mutable plan. The env owns exactly one; the world it draws from is frozen."""

    trip_start: datetime
    nights: int
    party_size: int
    districts: dict[Key, District]
    """Coordinate lookup for transfer times. Passed in rather than reached for, so an
    Itinerary can be constructed and tested without a whole World."""

    outbound: Flight | None = None
    inbound: Flight | None = None
    rail: list[RailLeg] = field(default_factory=list)
    lodging: dict[CityCode, tuple[Hotel, RoomOffer, int]] = field(default_factory=dict)
    activities: list[ActivitySlot] = field(default_factory=list)

    # ---- shape ---------------------------------------------------------------------

    @property
    def trip_end(self) -> datetime:
        return self.trip_start + timedelta(days=self.nights)

    @property
    def nights_covered(self) -> int:
        return sum(nights for _, _, nights in self.lodging.values())

    def is_complete(self) -> bool:
        """The minimum viable trip: you can get there, sleep every night, and get home.
        Activities are upside, not a requirement — a business traveller may want none."""
        return (
            self.outbound is not None
            and self.inbound is not None
            and self.nights_covered >= self.nights
        )

    def cities_visited(self) -> list[CityCode]:
        return list(self.lodging.keys())

    def city_on_day(self, trip_day: int) -> CityCode | None:
        """Which city the party sleeps in on a given night. Lodging is allocated in the
        order it was booked, which is how multi-city sequencing is expressed."""
        cursor = 0
        for city, (_, _, nights) in self.lodging.items():
            if cursor <= trip_day < cursor + nights:
                return city
            cursor += nights
        return None

    # ---- the feasibility predicates (T4-T6, S1-S2) ----------------------------------

    def can_add_lodging(self, hotel: Hotel, offer: RoomOffer, nights: int) -> str | None:
        """Returns None if legal, else a human-readable reason. The reason is for
        diagnostics and tests; the env never shows it to the agent as a penalty."""
        if hotel.city in self.lodging:
            return f"already have lodging in {hotel.city.value}"
        if nights <= 0:
            return "nights must be positive"
        if self.nights_covered + nights > self.nights:
            return (
                f"{self.nights_covered} + {nights} nights exceeds the {self.nights}-night trip"
            )
        if offer.max_occupancy < self.party_size:
            return f"room sleeps {offer.max_occupancy}, party is {self.party_size}"
        if offer.rooms_remaining <= 0:
            return "sold out"
        return None

    def can_add_activity(self, activity: Activity, trip_day: int) -> str | None:
        """The tightest constraint in the environment, and the one that makes an
        over-packed plan impossible rather than merely bad."""
        if len(self.activities) >= MAX_ACTIVITIES:
            return "itinerary full"
        if trip_day < 0 or trip_day >= self.nights:
            return "day outside the trip"

        # S2 — you must be in the city that night.
        if self.city_on_day(trip_day) is not activity.city:
            return f"not in {activity.city.value} on day {trip_day}"

        day_date = (self.trip_start + timedelta(days=trip_day)).date()
        # T4 — most museums close one day a week. This is a gate, not a preference.
        if day_date.isoweekday() in activity.closed_weekdays:
            return f"{activity.name} is closed on that weekday"

        start = self._earliest_start(activity, trip_day)
        if start is None:
            return "no feasible slot left in the day"
        if activity.daily_capacity <= 0:
            return "no capacity"
        return None

    def _earliest_start(self, activity: Activity, trip_day: int) -> datetime | None:
        """First start time on `trip_day` that satisfies opening hours (T4), no overlap
        with what is already booked (T5), and the transfer from the previous item (T6)."""
        day_base = (self.trip_start + timedelta(days=trip_day)).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
        )
        opens = day_base + timedelta(hours=activity.opens_hour)
        closes = day_base + timedelta(hours=activity.closes_hour)
        duration = timedelta(minutes=activity.duration_minutes)

        same_day = sorted(
            (s for s in self.activities if s.trip_day == trip_day),
            key=lambda s: s.start_utc,
        )

        candidate = opens
        for slot in same_day:
            # T6 — leave room for the journey between districts.
            travel = timedelta(
                minutes=self.transfer(slot.activity.district, activity.district)
            )
            if candidate + duration + travel <= slot.start_utc:
                return candidate
            candidate = max(candidate, slot.end_utc + travel)

        return candidate if candidate + duration <= closes else None

    def add_activity(self, activity: Activity, trip_day: int) -> ActivitySlot:
        start = self._earliest_start(activity, trip_day)
        if start is None:
            raise ValueError(f"no feasible slot for {activity.key} on day {trip_day}")
        slot = ActivitySlot(activity=activity, trip_day=trip_day, start_utc=start)
        self.activities.append(slot)
        return slot

    def committed_cost(self, party_size: int) -> int:
        """Total cents this plan will charge when booked.

        The mask reasons about this, not about item prices: selecting four individually
        affordable things and then being unable to pay for them is exactly the incoherence
        the environment is supposed to make unrepresentable.
        """
        total = 0
        for flight in (self.outbound, self.inbound):
            if flight is not None:
                total += flight.price * party_size
        for _, offer, nights in self.lodging.values():
            total += offer.price_per_night * nights
        for slot in self.activities:
            total += slot.activity.price_per_person * party_size
        return total

    def transfer(self, a: Key, b: Key) -> int:
        """Minutes between two districts. Same district still costs a short walk between
        venues, which is why this is not zero."""
        if a == b:
            return 10
        da, db = self.districts.get(a), self.districts.get(b)
        if da is None or db is None:
            return 45
        return transfer_minutes(da.coords, db.coords)

    # ---- what the client actually gets ----------------------------------------------

    def realized_features(self) -> dict[PreferenceAxis, float]:
        """Project the booked plan onto the eight preference axes, each in [0, 1].

        This is the vector the hidden client weights are dotted against, so it is the
        single most consequential function in the reward path. Every term reads only from
        what was actually booked.
        """
        feats = dict.fromkeys(PreferenceAxis, 0.0)

        if not self.lodging and self.outbound is None:
            return feats

        stays = list(self.lodging.values())
        if stays:
            weight = sum(n for _, _, n in stays) or 1
            feats[PreferenceAxis.COMFORT] = sum(
                (h.star_rating / 5.0) * n for h, _, n in stays
            ) / weight
            feats[PreferenceAxis.LOCATION] = sum(
                (self.districts[h.district].walkability / 9.0) * n for h, _, n in stays
            ) / weight

        cats = [s.activity.category.value for s in self.activities]
        feats[PreferenceAxis.FOOD] = min(1.0, cats.count("food") / 2.0)
        feats[PreferenceAxis.CULTURE] = min(
            1.0, (cats.count("history") + cats.count("museum")) / 3.0
        )
        # Novelty rewards breadth of district, not count of activities — visiting six
        # things in Shinjuku is not novel.
        feats[PreferenceAxis.NOVELTY] = min(
            1.0, len({s.activity.district for s in self.activities}) / 4.0
        )
        # Pacing peaks at ~1.5 activities/day and falls off in both directions: an empty
        # trip is not relaxing, and a packed one is not either.
        per_day = len(self.activities) / max(1, self.nights)
        feats[PreferenceAxis.PACING] = max(0.0, 1.0 - abs(per_day - 1.5) / 1.5)

        legs = [f for f in (self.outbound, self.inbound) if f is not None]
        if legs:
            feats[PreferenceAxis.RELIABILITY] = sum(f.on_time_rate for f in legs) / len(legs)
            if self.rail:
                # Choosing rail over a domestic hop is a reliability purchase.
                feats[PreferenceAxis.RELIABILITY] = min(
                    1.0, feats[PreferenceAxis.RELIABILITY] + 0.12 * len(self.rail)
                )
        # COST is filled by the caller, which alone knows the budget.
        return feats
