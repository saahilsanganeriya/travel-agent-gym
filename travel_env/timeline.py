"""The world graph: where the party is, at every instant of the trip.

This is the single most important module in v2, and it replaces a pile of ad-hoc checks with
one idea.

**The idea.** A trip is a chain of things that occupy time at a place. Between any two
consecutive things, the party has to physically get from the end of the first to the start of
the second. Written down:

    for consecutive segments (a, b):
        a.end_utc + transfer(a.place_out, b.place_in) <= b.start_utc

That one inequality subsumes almost every temporal rule the design doc used to list
separately:

* An over-packed day is infeasible, because the transfers do not fit.
* An activity in Kyoto cannot follow an activity in Tokyo, because the cross-city transfer is
  infinite — *unless* a booked rail leg sits between them, which is itself a segment whose
  `place_in` is Tokyo and whose `place_out` is Kyoto.
* A 09:00 tour cannot follow a flight that lands at 11:00.
* An overnight bus and a hotel night are alternatives, not additions, because the sleep block
  and the bus both occupy the same hours.
* **Act II needs no separate cascade logic.** A delay pushes one segment's `end_utc` forward;
  everything downstream that no longer satisfies the inequality is simply missed. The cascade
  is derived from the clock rather than sampled, so it is arbitrarily deep and always
  consistent with what was actually booked.

**Violations are returned, never raised.** A plan that does not fit in time is a *worse trip*,
not a programming error: the party could not be there, so the thing does not happen and its
value evaporates. The environment prices that through forfeiture, exactly as it prices a
cancelled flight — there is no penalty term. Generator faults are the opposite case and do
raise; see `clock.assert_world_coherent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

from travel_env.clock import from_local, local_date, to_local
from travel_env.contracts import (
    Activity,
    Airport,
    City,
    CityCode,
    District,
    DisruptionKind,
    FixedCommitment,
    Flight,
    GroundLeg,
    Hotel,
    Key,
    RoomOffer,
)
from travel_env.geo import airport_transfer_minutes, transfer_minutes

UNREACHABLE_MINUTES = 100_000
"""The transfer cost between two places with no local connection. Large enough that any
inequality involving it fails, finite so arithmetic never overflows or needs special cases."""

SLEEP_FROM_HOUR = 0
SLEEP_FROM_MINUTE = 30
SLEEP_TO_HOUR = 7
"""The party must be at its lodging between 00:30 and 07:00 local.

Deliberately generous. The purpose is to make "you cannot sleep in Kyoto and be at a bar in
Tokyo" true, not to impose a curfew — a tight bedtime would add difficulty that reflects
nothing real, and the whole point of v2 is that hardness comes from faithfulness.
"""


class SegmentKind(StrEnum):
    TRANSPORT = "transport"
    ACTIVITY = "activity"
    COMMITMENT = "commitment"
    SLEEP = "sleep"
    """Derived from a stay, not booked directly. Occupies the night at the lodging."""


@dataclass(frozen=True)
class Place:
    """Somewhere the party can be. A district, or an airport.

    An airport is not a district: it is an hour outside the city it serves, and treating it
    as "the city" is how a plan comes to assume you can land and be at dinner immediately.
    """

    key: Key
    city: CityCode | None
    is_airport: bool = False

    def __str__(self) -> str:  # for render() and test failures
        return f"{self.key}{'✈' if self.is_airport else ''}"


@dataclass
class Segment:
    """One thing that occupies time at, or between, places."""

    key: str
    kind: SegmentKind
    start_utc: datetime
    end_utc: datetime
    place_in: Place
    place_out: Place
    label: str
    ref: Any = None
    """The underlying record — Flight, GroundLeg, Activity, FixedCommitment or Hotel."""
    delay_minutes: int = 0
    """Applied in act II. Shifts `effective_end` only; the scheduled start is what was sold."""

    @property
    def effective_end(self) -> datetime:
        return self.end_utc + timedelta(minutes=self.delay_minutes)


@dataclass
class Stay:
    """A booked block of nights at one property."""

    hotel: Hotel
    offer: RoomOffer
    units: int
    first_night: int
    """Trip-day index of the first night. Night `d` runs from local day `d` into day `d+1`."""
    nights: int
    with_cot: bool = False

    @property
    def sleeps(self) -> int:
        return self.units * (self.offer.max_occupancy + (1 if self.with_cot else 0))

    def covers(self, night: int) -> bool:
        return self.first_night <= night < self.first_night + self.nights


@dataclass(frozen=True)
class Violation:
    """Something in the plan that cannot happen. Returned, never raised."""

    kind: DisruptionKind
    segment_key: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.detail}"


class PlaceResolver:
    """Turns world records into `Place`s and answers how long it takes to move between them.

    Holds the district and airport indices so a `Timeline` can be built and tested without a
    whole `World` — the same reason `Itinerary` takes its districts by argument.
    """

    def __init__(self, districts: dict[Key, District], airports: dict[Key, Airport], cities: dict[CityCode, City]):
        self.districts = districts
        self.airports = airports
        self.cities = cities
        # The station district: the most walkable district in each city, which is in practice
        # where the intercity terminal is. Precomputed because it is asked for on every leg.
        self._station: dict[CityCode, Place] = {}
        for code, city in cities.items():
            if city.districts:
                best = max(city.districts, key=lambda d: (d.walkability, d.key))
                self._station[code] = Place(key=best.key, city=code)

    def station(self, city: CityCode) -> Place:
        return self._station.get(city, Place(key=f"{city.value}_centre", city=city))

    def airport(self, key: Key) -> Place:
        port = self.airports.get(key)
        return Place(key=key, city=port.city if port else None, is_airport=True)

    def district(self, key: Key) -> Place:
        district = self.districts.get(key)
        return Place(key=key, city=district.city if district else None)

    def offset(self, city: CityCode | None) -> int:
        city_record = self.cities.get(city) if city else None
        return city_record.utc_offset_minutes if city_record else 0

    def transfer(self, a: Place, b: Place) -> int:
        """Minutes to get from `a` to `b` locally. `UNREACHABLE_MINUTES` if you cannot.

        Crossing cities is deliberately unreachable: intercity movement is a *booked leg*
        with a departure time, not a transfer you can assume. That is what forces the agent
        to actually buy the train.
        """
        if a == b:
            return 10
        if a.city is None or b.city is None or a.city is not b.city:
            return UNREACHABLE_MINUTES
        if a.is_airport or b.is_airport:
            port = self.airports.get(a.key if a.is_airport else b.key)
            other = self.districts.get(b.key if a.is_airport else a.key)
            if port is None or other is None:
                return UNREACHABLE_MINUTES
            return airport_transfer_minutes(port.minutes_to_centre, other)
        da, db = self.districts.get(a.key), self.districts.get(b.key)
        if da is None or db is None:
            return UNREACHABLE_MINUTES
        return transfer_minutes(da, db)


@dataclass
class Timeline:
    """The plan as a chain of occupied intervals, plus the nights that must be covered."""

    resolver: PlaceResolver
    trip_nights: int
    origin_airport: Key
    depart_home_after_utc: datetime
    """The earliest the party can leave home. Nothing may be scheduled before it."""

    outbound: Flight | None = None
    inbound: Flight | None = None
    ground: list[GroundLeg] = field(default_factory=list)
    stays: list[Stay] = field(default_factory=list)
    activities: list[tuple[Activity, datetime]] = field(default_factory=list)
    commitments: tuple[FixedCommitment, ...] = ()
    delays: dict[str, int] = field(default_factory=dict)
    """Act II arrival slippage, per segment key. Applied in `segments()` so every
    downstream consequence falls out of `violations()` instead of being enumerated."""

    # ---- day zero ------------------------------------------------------------------

    def arrival_place(self) -> Place | None:
        if self.outbound is None:
            return None
        return self.resolver.airport(self.outbound.destination)

    def day_zero(self) -> date | None:
        """The local calendar date of the first night, in the city the party lands in.

        Everything day-indexed hangs off this. It is derived from the outbound arrival rather
        than from `trip_start`, because a flight that leaves on the 3rd and crosses the date
        line lands on the 4th, and a trip whose day zero is wrong has every subsequent rule
        firing a day early.
        """
        place = self.arrival_place()
        if place is None or self.outbound is None:
            return None
        return local_date(self.outbound.arrival_utc, self.resolver.offset(place.city))

    def night_window(self, night: int, city: CityCode) -> tuple[datetime, datetime] | None:
        """The UTC interval the party must spend at its lodging on a given night."""
        zero = self.day_zero()
        if zero is None:
            return None
        offset = self.resolver.offset(city)
        day = zero + timedelta(days=night)
        start = from_local(
            datetime(day.year, day.month, day.day) + timedelta(days=1, hours=SLEEP_FROM_HOUR, minutes=SLEEP_FROM_MINUTE),
            offset,
        )
        end = from_local(
            datetime(day.year, day.month, day.day) + timedelta(days=1, hours=SLEEP_TO_HOUR), offset
        )
        return start, end

    # ---- the chain -------------------------------------------------------------------

    def segments(self) -> list[Segment]:
        """Every occupying interval, in clock order.

        Sorted by start, which for a coherent plan is also causal order. Where the two
        disagree the plan is wrong, and `violations` is what says so.
        """
        out: list[Segment] = []

        if self.outbound is not None:
            out.append(
                Segment(
                    key="seg_out",
                    kind=SegmentKind.TRANSPORT,
                    start_utc=self.outbound.departure_utc,
                    end_utc=self.outbound.arrival_utc,
                    place_in=self.resolver.airport(self.outbound.origin),
                    place_out=self.resolver.airport(self.outbound.destination),
                    label=f"{self.outbound.carrier}{self.outbound.flight_number} "
                    f"{self.outbound.origin}->{self.outbound.destination}",
                    ref=self.outbound,
                )
            )
        for n, leg in enumerate(self.ground):
            out.append(
                Segment(
                    key=f"seg_ground_{n}",
                    kind=SegmentKind.TRANSPORT,
                    start_utc=leg.departure_utc,
                    end_utc=leg.arrival_utc,
                    place_in=self.resolver.station(leg.origin_city),
                    place_out=self.resolver.station(leg.destination_city),
                    label=f"{leg.operator} {leg.mode.value} "
                    f"{leg.origin_city.value}->{leg.destination_city.value}",
                    ref=leg,
                )
            )
        for n, (activity, start) in enumerate(self.activities):
            out.append(
                Segment(
                    key=f"seg_act_{n}",
                    kind=SegmentKind.ACTIVITY,
                    start_utc=start,
                    end_utc=start + timedelta(minutes=activity.duration_minutes),
                    place_in=self.resolver.district(activity.district),
                    place_out=self.resolver.district(activity.district),
                    label=activity.name,
                    ref=activity,
                )
            )
        for commitment in self.commitments:
            out.append(
                Segment(
                    key=f"seg_fix_{commitment.key}",
                    kind=SegmentKind.COMMITMENT,
                    start_utc=commitment.start_utc,
                    end_utc=commitment.end_utc,
                    place_in=self.resolver.station(commitment.city),
                    place_out=self.resolver.station(commitment.city),
                    label=commitment.label,
                    ref=commitment,
                )
            )
        out.extend(self._sleep_segments())

        if self.inbound is not None:
            out.append(
                Segment(
                    key="seg_ret",
                    kind=SegmentKind.TRANSPORT,
                    start_utc=self.inbound.departure_utc,
                    end_utc=self.inbound.arrival_utc,
                    place_in=self.resolver.airport(self.inbound.origin),
                    place_out=self.resolver.airport(self.inbound.destination),
                    label=f"{self.inbound.carrier}{self.inbound.flight_number} "
                    f"{self.inbound.origin}->{self.inbound.destination}",
                    ref=self.inbound,
                )
            )

        for segment in out:
            segment.delay_minutes = self.delays.get(segment.key, 0)
        out.sort(key=lambda s: (s.start_utc, s.key))
        return out

    def _sleep_segments(self) -> list[Segment]:
        out: list[Segment] = []
        for stay in self.stays:
            place = self.resolver.district(stay.hotel.district)
            for night in range(stay.first_night, stay.first_night + stay.nights):
                window = self.night_window(night, stay.hotel.city)
                if window is None:
                    continue
                start, end = window
                out.append(
                    Segment(
                        key=f"seg_sleep_{night}",
                        kind=SegmentKind.SLEEP,
                        start_utc=start,
                        end_utc=end,
                        place_in=place,
                        place_out=place,
                        label=f"night {night} at {stay.hotel.name}",
                        ref=stay.hotel,
                    )
                )
        return out

    # ---- the verdict -------------------------------------------------------------------

    def violations(self) -> list[Violation]:
        """Everything about this plan that cannot physically happen. Never raises.

        Returned in clock order so the first entry is the root cause and the rest are its
        consequences — which is what makes an act II report readable rather than a list.
        """
        found: list[Violation] = []
        chain = self.segments()

        for earlier, later in zip(chain, chain[1:], strict=False):
            gap = (later.start_utc - earlier.effective_end).total_seconds() / 60.0
            needed = self.resolver.transfer(earlier.place_out, later.place_in)
            if gap >= needed:
                continue
            if needed >= UNREACHABLE_MINUTES:
                detail = (
                    f"{later.label} is in {later.place_in.city.value if later.place_in.city else '?'} "
                    f"but the party is in {earlier.place_out.city.value if earlier.place_out.city else '?'} "
                    f"after {earlier.label}, with nothing booked between them"
                )
            else:
                detail = (
                    f"{later.label} starts {gap:.0f}m after {earlier.label} ends, "
                    f"but the transfer takes {needed}m"
                )
            found.append(
                Violation(kind=DisruptionKind.UNREACHABLE_IN_TIME, segment_key=later.key, detail=detail)
            )

        found.extend(self._check_beds())
        found.extend(self._check_check_in())
        found.extend(self._check_commitments(chain))
        return found

    def _check_beds(self) -> list[Violation]:
        """Every night needs a bed, or an overnight leg that replaces one."""
        out: list[Violation] = []
        for night in range(self.trip_nights):
            if any(stay.covers(night) for stay in self.stays):
                continue
            if self._overnight_leg_on(night):
                continue
            out.append(
                Violation(
                    kind=DisruptionKind.CAPACITY_FAILURE,
                    segment_key=f"night_{night}",
                    detail=f"night {night} has no lodging and no overnight leg",
                )
            )
        return out

    def _overnight_leg_on(self, night: int) -> bool:
        """An overnight bus or sleeper is a real substitute for a hotel — cheaper, and it
        costs a night of rest. Modelled as covering the night if it spans the sleep window."""
        for leg in self.ground:
            window = self.night_window(night, leg.destination_city)
            if window is None:
                continue
            start, end = window
            if leg.departure_utc <= start and leg.arrival_utc >= end:
                return True
        return False

    def _check_check_in(self) -> list[Violation]:
        """Reception has to still be open when the party gets there.

        This is the "land at 23:40 against a 22:00 desk" failure, and it is entirely caused by
        the flight the agent chose — which is what makes it a consequence rather than a tax.
        """
        out: list[Violation] = []
        chain = [s for s in self.segments() if s.kind is SegmentKind.TRANSPORT]
        for stay in self.stays:
            arrival = self._arrival_into(stay.hotel.city, before_night=stay.first_night, chain=chain)
            if arrival is None:
                continue
            offset = self.resolver.offset(stay.hotel.city)
            place_from = (
                self.resolver.airport(self.outbound.destination)
                if self.outbound is not None and arrival == self.outbound.arrival_utc
                else self.resolver.station(stay.hotel.city)
            )
            at_desk = arrival + timedelta(
                minutes=self.resolver.transfer(place_from, self.resolver.district(stay.hotel.district))
            )
            local = to_local(at_desk, offset)
            cutoff = stay.hotel.check_in_until_hour
            if cutoff >= 24:
                continue
            if local.hour >= cutoff or local.hour < 5:
                out.append(
                    Violation(
                        kind=DisruptionKind.HOTEL_OVERBOOKED,
                        segment_key=f"stay_{stay.hotel.key}",
                        detail=(
                            f"{stay.hotel.name} closes reception at {cutoff:02d}:00 local; the party "
                            f"reaches the desk at {local:%H:%M}"
                        ),
                    )
                )
        return out

    def _arrival_into(self, city: CityCode, *, before_night: int, chain: list[Segment]) -> datetime | None:
        """When the party most recently arrived in `city` before that night began."""
        window = self.night_window(before_night, city)
        latest: datetime | None = None
        for segment in chain:
            if segment.place_out.city is not city:
                continue
            end = segment.effective_end
            if window is not None and end > window[1]:
                continue
            if latest is None or end > latest:
                latest = end
        return latest

    def _check_commitments(self, chain: list[Segment]) -> list[Violation]:
        """The client's own immovable points. Missing one is the worst thing on this list —
        the trip can be flawless and still have failed at the only thing that had to happen."""
        out: list[Violation] = []
        keys = {f"seg_fix_{c.key}" for c in self.commitments}
        broken = {v.segment_key for v in self._reachability_failures(chain)}
        for commitment in self.commitments:
            key = f"seg_fix_{commitment.key}"
            if key in keys and key in broken:
                out.append(
                    Violation(
                        kind=DisruptionKind.COMMITMENT_MISSED,
                        segment_key=key,
                        detail=f"{commitment.label} in {commitment.city.value} could not be attended",
                    )
                )
        return out

    def _reachability_failures(self, chain: list[Segment]) -> list[Violation]:
        out: list[Violation] = []
        for earlier, later in zip(chain, chain[1:], strict=False):
            gap = (later.start_utc - earlier.effective_end).total_seconds() / 60.0
            if gap < self.resolver.transfer(earlier.place_out, later.place_in):
                out.append(
                    Violation(
                        kind=DisruptionKind.UNREACHABLE_IN_TIME,
                        segment_key=later.key,
                        detail=later.label,
                    )
                )
        return out

    # ---- the predicate the mask calls ----------------------------------------------------

    def would_break(self, candidate: Segment) -> str | None:
        """Would adding this segment make the plan infeasible? None means it is safe to add.

        This is the *only* thing the action mask needs to know about time. An option that
        would break the plan is not offered, so the agent never has to learn to avoid it —
        which is the masking-over-penalties discipline applied to scheduling.
        """
        chain = sorted([*self.segments(), candidate], key=lambda s: (s.start_utc, s.key))
        index = next((i for i, s in enumerate(chain) if s.key == candidate.key), None)
        if index is None:  # pragma: no cover — the candidate was just inserted
            return "candidate vanished from the chain"

        pairs: list[tuple[Segment, Segment]] = []
        if index > 0:
            pairs.append((chain[index - 1], candidate))
        if index + 1 < len(chain):
            pairs.append((candidate, chain[index + 1]))
        for a, b in pairs:
            gap = (b.start_utc - a.effective_end).total_seconds() / 60.0
            needed = self.resolver.transfer(a.place_out, b.place_in)
            if gap < needed:
                if needed >= UNREACHABLE_MINUTES:
                    return f"cannot get from {a.place_out} to {b.place_in}"
                return f"{needed}m transfer does not fit in a {gap:.0f}m gap"
        return None

    def earliest_start_for(self, activity: Activity, day: int, opens_utc: datetime, closes_utc: datetime) -> datetime | None:
        """First feasible start for `activity` on `day`, honouring every transfer.

        Walks the day's existing segments in order and tries the first gap the activity plus
        its transfers actually fit into — rather than assuming the day is empty, which is how
        an itinerary comes to contain two things at once.
        """
        duration = timedelta(minutes=activity.duration_minutes)
        place = self.resolver.district(activity.district)
        same_day = [
            s for s in self.segments()
            if s.start_utc < closes_utc and s.effective_end > opens_utc
        ]
        same_day.sort(key=lambda s: s.start_utc)

        candidate = opens_utc
        for segment in same_day:
            travel_in = timedelta(minutes=self.resolver.transfer(segment.place_out, place))
            travel_out = timedelta(minutes=self.resolver.transfer(place, segment.place_in))
            if candidate + duration + travel_out <= segment.start_utc:
                return candidate
            candidate = max(candidate, segment.effective_end + travel_in)

        return candidate if candidate + duration <= closes_utc else None

    # ---- summaries the env and the reward read ----------------------------------------

    def nights_covered(self) -> int:
        return sum(stay.nights for stay in self.stays)

    def cities_visited(self) -> list[CityCode]:
        seen: list[CityCode] = []
        for stay in sorted(self.stays, key=lambda s: s.first_night):
            if stay.hotel.city not in seen:
                seen.append(stay.hotel.city)
        return seen

    def is_complete(self) -> bool:
        """The minimum viable trip: you can get there, sleep every night, and get home."""
        return (
            self.outbound is not None
            and self.inbound is not None
            and self.nights_covered() >= self.trip_nights
        )

    def apply_delay(self, segment_key: str, minutes: int) -> None:
        """Act II. Push one segment's arrival later.

        Downstream failures are NOT enumerated here — they fall out of `violations()`, because
        the party simply cannot be where the plan says it should be. That is what makes the
        cascade arbitrarily deep and always consistent with what was actually booked.
        """
        self.delays[segment_key] = self.delays.get(segment_key, 0) + minutes
