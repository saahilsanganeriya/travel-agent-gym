"""The world's clock, and the proof that the world obeys it.

Everything in this environment happens at an instant, and almost every interesting failure
is a *timing* failure — a room released because the flight landed at 23:40, a tour missed
because the ferry runs twice a day, a connection lost to a delay. That only works if there
is exactly one time axis and every record sits on it correctly. v1 did not have that, and
the bugs were invisible by inspection:

* `local_departure_hour = (dep.hour + 9) % 24` applied *Tokyo's* offset to a *San Francisco*
  departure, so every long-haul leg reported the wrong local time by 16-17 hours.
* Activity opening hours are local, but the scheduler compared them against a UTC day base,
  a silent nine-hour shift that made evening venues look like morning ones.
* Arrival local hour was departure hour plus duration, which ignores crossing the date line.

The three disciplines here are lifted from a generator pipeline that had already paid for
them:

1. **One anchor, composed forward, inside a reserved margin.** Nothing is drawn at an
   absolute instant; everything is built forward from the world's start with bounded hops,
   so a chain cannot walk off the end of the world.
2. **Monotone in causal position, not in index.** If A causes B then A's instant precedes
   B's. The graph order and the clock order are the same order or the world is wrong.
3. **One instant, many renderings.** A moment is stored once in UTC. Local hour, local date
   and "which trip day is this" are *derived*, never drawn a second time. Two draws of one
   moment is how two parts of a world come to disagree about when something happened.

**Raising versus scoring.** `assert_world_coherent` raises. It grades the *generator*, not
the agent: a flight that arrives before it departs is a bug in this repository, and an agent
graded down for it would be graded for someone else's mistake. Everything the *agent* can do
wrong — an itinerary that does not fit in real time — is handled in `timeline.py`, which
returns violations and never raises. See `timeline.Timeline.violations`.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from travel_env.contracts import Coordinates, TransportMode, World

WORLD_MARGIN = timedelta(days=21)
"""Headroom reserved before the trip and after it.

Schedules are composed forward from `window_start = trip_start - MARGIN`, and every hop is
far shorter than the margin, so no chain can reach `window_end`. The check is an assertion
rather than a clamp because a moment outside the window means a generator walked further
than it declared it would, and clamping would hide that.
"""

_EARTH_RADIUS_KM = 6371.0

# Effective door-to-door speed bands per mode, in km/h, as a *coherence* check rather than a
# pricing input. A leg outside its band is a teleport or a crawl, and both are generator
# faults. Bands are wide on purpose: block speed includes taxi, boarding and dwell time, so
# a short hop is legitimately far slower than the vehicle's cruise speed.
_SPEED_BAND_KMH: dict[TransportMode, tuple[float, float]] = {
    TransportMode.FLIGHT: (110.0, 1000.0),
    TransportMode.RAIL: (30.0, 300.0),
    TransportMode.BUS: (15.0, 110.0),
    TransportMode.FERRY: (8.0, 90.0),
    TransportMode.CAR_HIRE: (12.0, 120.0),
}


class WorldCoherenceError(AssertionError):
    """The generator produced a world that does not obey its own clock or geography.

    Deliberately an `AssertionError`: this is never an agent-reachable condition and never a
    scored outcome. If one of these escapes into a training run, the run is invalid.
    """


# ---- One instant, many renderings ---------------------------------------------------


def resolve_offset(timezone_name: str, on: datetime) -> int:
    """UTC offset in minutes for a zone on a given instant, resolved once at build time.

    Uses real `zoneinfo` data so summer time is correct — Italy is +60 in February and +120
    in July, and a world generated across that boundary would otherwise mis-time every
    Italian opening hour by an hour. The result is then frozen onto the `City`, because a
    world spans days rather than months and re-deriving it per call invites two call sites
    to disagree about what "09:00 local" meant.
    """
    offset = on.astimezone(ZoneInfo(timezone_name)).utcoffset()
    if offset is None:  # pragma: no cover — zoneinfo always supplies one for a real zone
        raise WorldCoherenceError(f"{timezone_name} yielded no UTC offset at {on.isoformat()}")
    return int(offset.total_seconds() // 60)


def to_local(instant: datetime, offset_minutes: int) -> datetime:
    """The same instant rendered as naive local wall-clock. Never a second draw."""
    return instant.astimezone(UTC).replace(tzinfo=None) + timedelta(minutes=offset_minutes)


def from_local(local_naive: datetime, offset_minutes: int) -> datetime:
    """Local wall-clock back to the UTC instant. The inverse of `to_local`, exactly."""
    return (local_naive - timedelta(minutes=offset_minutes)).replace(tzinfo=UTC)


def local_hour(instant: datetime, offset_minutes: int) -> int:
    return to_local(instant, offset_minutes).hour


def local_date(instant: datetime, offset_minutes: int) -> date:
    """Which calendar day this is *where it happens*.

    A booking's service date is a local fact. Recording every night of a trip against
    `trip_start.date()` — which is what v1 did — makes a day-five hotel night look like a
    day-zero one, and every date-keyed rule downstream then fires on the wrong day.
    """
    return to_local(instant, offset_minutes).date()


def local_window(
    day_local: date, from_hour: int, to_hour: int, offset_minutes: int
) -> tuple[datetime, datetime]:
    """A local opening window as a pair of UTC instants.

    `to_hour == 24` means midnight at the end of the day, which is a real and common value
    for a 24-hour reception and cannot be written as hour 24 of a `datetime`.
    """
    base = datetime(day_local.year, day_local.month, day_local.day)
    start = from_local(base + timedelta(hours=from_hour), offset_minutes)
    end = from_local(base + timedelta(hours=to_hour), offset_minutes)
    return start, end


def haversine_km(a: Coordinates, b: Coordinates) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


# ---- The generator self-check --------------------------------------------------------


def assert_world_coherent(world: World) -> None:
    """Prove the world obeys its own clock and geography. Raises on the first fault.

    Run in the test suite and by `scripts/dump_data.py`, not on every `reset()` — it is a
    proof about the generator, and the generator is deterministic in its seed, so proving it
    over a sample of seeds proves it everywhere the sampler can reach.
    """
    faults: list[str] = []
    start, end = world.window_start_utc, world.window_end_utc
    if start is None or end is None:
        raise WorldCoherenceError("world declares no clock window")
    if start >= end:
        raise WorldCoherenceError(f"window is empty or inverted: {start} .. {end}")

    cities = world.city_index()
    airports = world.airport_index()
    districts = world.district_index()

    def check_instant(label: str, moment: datetime) -> None:
        if moment.tzinfo is None:
            faults.append(f"{label}: naive datetime {moment!r} — every instant must be UTC-aware")
        elif not (start <= moment <= end):
            faults.append(f"{label}: {moment.isoformat()} lies outside the world window")

    # ---- Cities carry the offset their own zone actually had -------------------------
    for city in world.cities:
        expected = resolve_offset(city.timezone, start)
        if city.utc_offset_minutes != expected:
            faults.append(
                f"{city.code.value}: frozen offset {city.utc_offset_minutes} != "
                f"{expected} for {city.timezone} at the window start"
            )
        for district in city.districts:
            if district.city is not city.code:
                faults.append(f"district {district.key} claims {district.city} but sits in {city.code}")

    # ---- Every scheduled leg: forward in time, and physically possible ----------------
    for flight in world.flights:
        label = f"flight {flight.key}"
        check_instant(f"{label} departure", flight.departure_utc)
        check_instant(f"{label} arrival", flight.arrival_utc)
        _check_leg_duration(faults, label, flight.departure_utc, flight.arrival_utc, flight.duration_minutes)
        origin, dest = airports.get(flight.origin), airports.get(flight.destination)
        if origin is None or dest is None:
            faults.append(f"{label}: unknown endpoint {flight.origin} -> {flight.destination}")
        elif origin.key == dest.key:
            faults.append(f"{label}: departs and arrives at {origin.key}")
        else:
            _check_speed(faults, label, TransportMode.FLIGHT, origin.coords, dest.coords, flight.duration_minutes)

    for leg in world.ground:
        label = f"ground {leg.key} ({leg.mode.value})"
        check_instant(f"{label} departure", leg.departure_utc)
        check_instant(f"{label} arrival", leg.arrival_utc)
        _check_leg_duration(faults, label, leg.departure_utc, leg.arrival_utc, leg.duration_minutes)
        origin, dest = cities.get(leg.origin_city), cities.get(leg.destination_city)
        if origin is None or dest is None:
            faults.append(f"{label}: unknown endpoint {leg.origin_city} -> {leg.destination_city}")
        elif origin.code is dest.code:
            faults.append(f"{label}: departs and arrives in {origin.code.value}")
        else:
            _check_speed(faults, label, leg.mode, origin.centre, dest.centre, leg.duration_minutes)
            if origin.region is not dest.region:
                faults.append(f"{label}: surface leg crosses regions {origin.region} -> {dest.region}")

    for leg in world.rail:  # the v1 corridor model, still validated if present
        label = f"rail {leg.key}"
        check_instant(f"{label} departure", leg.departure_utc)
        _check_leg_duration(faults, label, leg.departure_utc, leg.arrival_utc, leg.duration_minutes)

    # ---- Lodging and activities sit in places that exist ------------------------------
    hotel_keys = {h.key for h in world.hotels}
    for hotel in world.hotels:
        if hotel.district not in districts:
            faults.append(f"hotel {hotel.key}: district {hotel.district!r} does not exist")
        elif districts[hotel.district].city is not hotel.city:
            faults.append(f"hotel {hotel.key}: district {hotel.district} is not in {hotel.city.value}")
        if hotel.check_in_until_hour <= hotel.check_in_from_hour and hotel.check_in_until_hour != 24:
            faults.append(
                f"hotel {hotel.key}: reception opens {hotel.check_in_from_hour} and shuts "
                f"{hotel.check_in_until_hour} — the window is empty"
            )

    for offer in world.room_offers:
        if offer.hotel not in hotel_keys:
            faults.append(f"offer {offer.key}: no such hotel {offer.hotel!r}")
        if offer.cot_price_per_night and not offer.cot_available:
            faults.append(f"offer {offer.key}: prices a cot it does not have")

    for activity in world.activities:
        if activity.district not in districts:
            faults.append(f"activity {activity.key}: district {activity.district!r} does not exist")
        elif districts[activity.district].city is not activity.city:
            faults.append(
                f"activity {activity.key}: district {activity.district} is not in {activity.city.value}"
            )
        if activity.closes_hour <= activity.opens_hour:
            faults.append(
                f"activity {activity.key}: opens {activity.opens_hour} closes {activity.closes_hour}"
            )
        elif activity.duration_minutes > (activity.closes_hour - activity.opens_hour) * 60:
            faults.append(
                f"activity {activity.key}: runs {activity.duration_minutes}m but its window is "
                f"only {(activity.closes_hour - activity.opens_hour) * 60}m — it can never be booked"
            )
        if activity.min_age >= activity.max_age:
            faults.append(f"activity {activity.key}: age band {activity.min_age}..{activity.max_age} is empty")
        if not activity.season_months:
            faults.append(f"activity {activity.key}: runs in no month of the year")

    if faults:
        head = "\n  ".join(faults[:12])
        more = f"\n  ... and {len(faults) - 12} more" if len(faults) > 12 else ""
        raise WorldCoherenceError(f"world seed={world.seed} is incoherent:\n  {head}{more}")


def _check_leg_duration(
    faults: list[str], label: str, departure: datetime, arrival: datetime, declared_minutes: int
) -> None:
    """Arrival is departure plus duration. One instant, one arithmetic — not two draws.

    Exact equality rather than a tolerance: both sides are constructed from the same
    `timedelta`, so any drift means somebody drew the arrival independently, which is the
    failure mode this whole module exists to prevent.
    """
    if arrival <= departure:
        faults.append(f"{label}: arrives {arrival.isoformat()} at or before it departs")
        return
    actual = int((arrival - departure).total_seconds() // 60)
    if actual != declared_minutes:
        faults.append(f"{label}: declares {declared_minutes}m but the clock says {actual}m")


def _check_speed(
    faults: list[str],
    label: str,
    mode: TransportMode,
    a: Coordinates,
    b: Coordinates,
    minutes: int,
) -> None:
    low, high = _SPEED_BAND_KMH[mode]
    km = haversine_km(a, b)
    if km < 1.0 or minutes <= 0:
        return
    kmh = km / (minutes / 60.0)
    if kmh > high:
        faults.append(f"{label}: {km:.0f}km in {minutes}m is {kmh:.0f}km/h — faster than {mode.value} goes")
    elif kmh < low:
        faults.append(f"{label}: {km:.0f}km in {minutes}m is {kmh:.0f}km/h — slower than {mode.value} goes")
