"""The v2 world generator: multi-region, timezone-correct, physically coherent.

v1 generated a Tokyo-Kyoto-Osaka corridor in which a uniformly random policy solved a quarter
of the episodes. That is a measurement of the *world*, not of the policy: if stumbling into a
valid itinerary is likely, the space of legal plans is too small.

This generator widens it along the axes that a better agent can exploit and a worse one
cannot — which is the test every addition here had to pass. More price noise would widen the
outcome distribution for everybody and discriminate less; more *places, modes and rules*
makes the right answer harder to find and still definitely right.

What changed, and what each change is for:

* **One or two regions per episode, drawn from three.** Two regions means an open-jaw:
  fly into one, cross, fly home from the other. Random cannot stumble into that.
* **Five transport modes.** Rail, bus, ferry, car hire and flights, each with its own
  reliability, luggage rule, step-free status and weather exposure. Choosing *between* modes
  is where most real travel-agent skill lives.
* **Seven lodging types.** A capsule sleeps one. An apartment sleeps six in one unit for less
  than three hotel rooms. A ryokan is half-board and has no lift. Occupancy is a puzzle.
* **Self-transfer fares.** Two tickets sold as one journey: cheaper, and uninsured against a
  missed connection. Strictly worse only once act II runs, which is exactly what makes it a
  trap rather than a tax.
* **Accessibility distributed by geography, not sprinkled.** A district's step-free share is
  a property of the district, so Venice and hillside Kyoto are genuinely hard for a wheelchair
  user and Marunouchi is not. Accessibility therefore *interacts* with the location
  preference instead of sitting beside it.
* **Every instant is correct.** Offsets come from `zoneinfo`, arrival is departure plus a
  duration derived from real distance, and `clock.assert_world_coherent` proves it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from travel_env.clock import WORLD_MARGIN, haversine_km, resolve_offset
from travel_env.contracts import (
    AccessibilityNeed,
    Activity,
    ActivityCategory,
    Airport,
    BoardType,
    CabinClass,
    City,
    CityCode,
    Coordinates,
    DietaryNeed,
    District,
    FareBrand,
    Flight,
    GroundLeg,
    Hotel,
    Key,
    LodgingType,
    Nationality,
    RatePlan,
    Region,
    RoomOffer,
    TransportMode,
    VisaRule,
    World,
)
from travel_env.regions import (
    ACTIVITIES_BY_CITY,
    AIRPORT_SPECS,
    CITY_SPECS,
    CORRIDOR_SPECS,
    DISTRICTS_BY_CITY,
    ORIGIN,
    SEASON_SPECS,
    TAIL_FORMATS,
    TAIL_THEMES,
    VISA_POLICY,
    CitySpec,
    district_key,
)

_EPOCH = datetime(2027, 1, 1, tzinfo=UTC)
"""Fixed origin for `World.generated_at`, offset by the seed. Wall-clock here silently broke
`generate_world(42) == generate_world(42)` in v1, which is the whole determinism claim."""

DEFAULT_TRIP_START = datetime(2027, 4, 3, tzinfo=UTC)

# Block speed: 810 km/h in the air plus 25 minutes of taxi, push-back and approach. Deriving
# duration from distance rather than drawing it is what lets `assert_world_coherent` check
# that no leg is a teleport — a duration nobody derived is a duration nobody can check.
def _clamp(value: float, low: float, high: float) -> float:
    """Scalar clamp. Deliberately NOT `np.clip`.

    `np.clip` on a Python float goes through numpy's full dispatch machinery at roughly
    2.8us a call. Profiling one world found 3,261 such calls costing 9ms — a sixth of
    generation — for arithmetic the interpreter does in nanoseconds. Vectorised `np.clip`
    over an array is still the right tool; scalar `np.clip` in a loop never is.
    """
    return low if value < low else high if value > high else value


_CRUISE_KM_PER_MIN = 13.5
_FLIGHT_OVERHEAD_MIN = 25

_CARRIERS: tuple[tuple[str, str, float, float], ...] = (
    # carrier, alliance, on-time rate, price tier. Reliability and price correlate but not
    # monotonically: the cheapest is the worst, the priciest is not the best.
    ("JL", "oneworld", 0.93, 1.25),
    ("NH", "star", 0.92, 1.22),
    ("UA", "star", 0.81, 1.00),
    ("EK", "none", 0.88, 1.14),
    ("ZG", "none", 0.72, 0.78),
    ("MM", "none", 0.68, 0.62),
)

_LODGING_MIX: tuple[tuple[LodgingType, float, float, int, int, int, float], ...] = (
    # type, share, price multiplier, max occupancy, check-in from, check-in until, step-free bias
    (LodgingType.HOTEL, 0.40, 1.00, 3, 15, 24, 1.00),
    (LodgingType.APARTMENT, 0.16, 1.35, 6, 16, 21, 0.75),
    (LodgingType.HOSTEL, 0.12, 0.34, 1, 14, 23, 0.55),
    (LodgingType.GUESTHOUSE, 0.12, 0.62, 2, 15, 21, 0.45),
    (LodgingType.RYOKAN, 0.08, 1.55, 4, 15, 19, 0.25),
    (LodgingType.RESORT, 0.07, 1.70, 4, 15, 24, 0.90),
    (LodgingType.CAPSULE, 0.05, 0.22, 1, 16, 24, 0.60),
)

_HOTEL_WORDS: dict[Region, tuple[str, ...]] = {
    Region.JAPAN: ("Kohaku", "Sakura", "Mitsu", "Aoi", "Hoshi", "Yanagi", "Tsuru", "Momiji", "Kaze"),
    Region.ITALY: ("Aurelia", "Bellini", "Corso", "Fiore", "Lucania", "Navona", "Prima", "Verdi"),
    Region.THAILAND: ("Baan", "Chao", "Lamai", "Nakara", "Ploen", "Sawasdee", "Thara", "Ubon"),
}
_HOTEL_SUFFIX: dict[LodgingType, tuple[str, ...]] = {
    LodgingType.HOTEL: ("Hotel", "Grand", "Central"),
    LodgingType.APARTMENT: ("Residences", "Suites", "Apartments"),
    LodgingType.HOSTEL: ("Hostel", "Backpackers"),
    LodgingType.GUESTHOUSE: ("Guesthouse", "House", "Inn"),
    LodgingType.RYOKAN: ("Ryokan", "Onsen"),
    LodgingType.RESORT: ("Resort", "Beach Resort", "Spa Resort"),
    LodgingType.CAPSULE: ("Capsule", "Pod"),
}


# ---- Seasonality and pricing ------------------------------------------------------------


def season_multiplier(region: Region, day_of_year: int) -> float:
    """Regional demand curve. Sum of Gaussians over a flat base, per `SEASON_SPECS`."""
    spec = SEASON_SPECS[region]
    total = 1.0
    for amplitude, centre, width in spec.peaks:
        # Wrap the year so a peak near new year is not cut in half at the boundary.
        delta = min(abs(day_of_year - centre), 365 - abs(day_of_year - centre))
        total += amplitude * math.exp(-((delta / width) ** 2))
    return total


def weather_risk(region: Region, day_of_year: int) -> float:
    """Probability mass of a whole-day regional weather event on a given date.

    Deliberately *not* the same curve as price: in Thailand the cheapest months are the
    wettest, so the budget option carries the ferry risk. That coupling is the reason
    seasonality is modelled at all rather than being a flat multiplier.
    """
    spec = SEASON_SPECS[region]
    total = 0.0
    for amplitude, centre, width in spec.weather_risk:
        delta = min(abs(day_of_year - centre), 365 - abs(day_of_year - centre))
        total += amplitude * math.exp(-((delta / width) ** 2))
    return float(min(0.85, total))


def lead_time_multiplier(days_ahead: int) -> float:
    """Airline pricing: cheap in the 21-90 day window, punitive inside two weeks."""
    if days_ahead <= 3:
        return 2.10
    if days_ahead <= 14:
        return 1.45
    if days_ahead <= 21:
        return 1.15
    if days_ahead <= 90:
        return 1.00
    return 1.08


# ---- Region draw --------------------------------------------------------------------------


def draw_regions(rng: np.random.Generator, difficulty: float) -> tuple[Region, ...]:
    """One region, or two. Two is an open-jaw and is materially harder.

    Only the chosen regions get inventory generated. That keeps a reset cheap while leaving
    the *catalogue* thirteen cities wide — the agent faces a different subset every episode,
    which is what stops a policy memorising one map.
    """
    all_regions = list(Region)
    two = rng.random() < (0.22 + 0.38 * difficulty)
    if not two:
        return (all_regions[int(rng.integers(0, len(all_regions)))],)
    pair = rng.choice(len(all_regions), size=2, replace=False)
    return (all_regions[int(pair[0])], all_regions[int(pair[1])])


# ---- The generator -------------------------------------------------------------------------


def generate_world(
    seed: int,
    *,
    trip_start: datetime | None = None,
    nights: int = 7,
    difficulty: float = 0.5,
    regions: tuple[Region, ...] | None = None,
) -> World:
    """Build one complete, self-consistent, multi-region world. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    start = trip_start or DEFAULT_TRIP_START
    window_start = start - WORLD_MARGIN
    window_end = start + timedelta(days=nights) + WORLD_MARGIN
    chosen = regions or draw_regions(rng, difficulty)

    city_specs = [c for c in CITY_SPECS if c.region in chosen]
    cities, districts_by_city = _build_cities(city_specs, window_start)
    airports = _build_airports(city_specs, window_start)
    airport_index = {a.key: a for a in airports}

    day_of_year = start.timetuple().tm_yday
    flights = _generate_flights(rng, chosen, city_specs, airport_index, start, nights, day_of_year)
    ground = _generate_ground(rng, cities, start, nights, day_of_year)
    hotels, offers = _generate_lodging(rng, city_specs, districts_by_city, start, day_of_year)
    activities = _generate_activities(rng, city_specs, districts_by_city, start, day_of_year)
    visa_rules = _visa_rules(city_specs)

    return World(
        seed=seed,
        generated_at=_EPOCH + timedelta(seconds=seed),
        cities=tuple(cities.values()),
        airports=airports,
        flights=flights,
        rail=(),  # v2 folds rail into `ground`, which carries the mode
        hotels=hotels,
        room_offers=offers,
        activities=activities,
        regions=tuple(chosen),
        ground=ground,
        visa_rules=visa_rules,
        window_start_utc=window_start,
        window_end_utc=window_end,
        origin_airport=ORIGIN.key,
    )


def _build_cities(
    city_specs: list[CitySpec], window_start: datetime
) -> tuple[dict[CityCode, City], dict[CityCode, list[District]]]:
    districts_by_city: dict[CityCode, list[District]] = {}
    cities: dict[CityCode, City] = {}
    for spec in city_specs:
        built = [
            District(
                key=district_key(d.name),
                name=d.name,
                city=d.city,
                coords=Coordinates(lat=d.lat, lon=d.lon),
                walkability=d.walkability,
                nightlife=d.nightlife,
                dining_density=d.dining,
                cultural_density=d.cultural,
                quiet=d.quiet,
                price_index=d.price_index,
            )
            for d in DISTRICTS_BY_CITY[spec.code]
        ]
        districts_by_city[spec.code] = built
        cities[spec.code] = City(
            code=spec.code,
            name=spec.name,
            country=spec.country,
            timezone=spec.timezone,
            centre=Coordinates(lat=spec.lat, lon=spec.lon),
            districts=tuple(built),
            airports=spec.airports,
            region=spec.region,
            # Resolved from real tz data at the window start, then frozen. One instant, one
            # rendering — see clock.resolve_offset.
            utc_offset_minutes=resolve_offset(spec.timezone, window_start),
        )
    return cities, districts_by_city


def _build_airports(city_specs: list[CitySpec], window_start: datetime) -> tuple[Airport, ...]:
    codes = {c.code for c in city_specs}
    wanted = [a for a in AIRPORT_SPECS if a.city in codes and a.key != ORIGIN.key]
    wanted.append(ORIGIN)
    return tuple(
        Airport(
            key=a.key,
            name=a.name,
            city=a.city,
            coords=Coordinates(lat=a.lat, lon=a.lon),
            minutes_to_centre=a.minutes_to_centre,
            country=a.country,
            utc_offset_minutes=resolve_offset(a.timezone, window_start),
            min_connection_minutes=a.min_connection_minutes,
        )
        for a in wanted
    )


def _flight_minutes(a: Coordinates, b: Coordinates, *, stops: int) -> int:
    """Block time derived from distance, so every leg passes the speed check by construction."""
    direct = _FLIGHT_OVERHEAD_MIN + haversine_km(a, b) / _CRUISE_KM_PER_MIN
    return int(round(direct * (1.0 + 0.35 * stops)))


def _generate_flights(
    rng: np.random.Generator,
    chosen: tuple[Region, ...],
    city_specs: list[CitySpec],
    airports: dict[Key, Airport],
    start: datetime,
    nights: int,
    day_of_year: int,
) -> tuple[Flight, ...]:
    """Long-haul in, long-haul out, the inter-region hop, and worthwhile domestic legs.

    Fare brands and the self-transfer flag are the load-bearing details. In act I a BASIC
    self-transfer and a FLEX through-fare look like the same seat at different prices. In act
    II they are different products, and the difference is the whole reason the data model
    carries them.
    """
    out: list[Flight] = []
    origin_port = airports[ORIGIN.key]
    gateways = [
        airports[a.key]
        for spec in city_specs
        for a in AIRPORT_SPECS
        if a.city is spec.code and a.intercontinental and a.key in airports
    ]
    if not gateways:  # pragma: no cover — every region has at least one gateway
        gateways = [airports[k] for k in airports if k != ORIGIN.key][:1]

    # Booking lead time is a world constant rather than a wall-clock difference: deriving it
    # from `now()` would make the same seed price differently on different days.
    lead_days = 45
    first_region_gateways = [g for g in gateways if _region_of(g.city) == chosen[0]]
    last_region_gateways = [g for g in gateways if _region_of(g.city) == chosen[-1]]

    def emit(
        origin: Airport,
        dest: Airport,
        depart: datetime,
        base_price: int,
        *,
        stops: int,
        season: float,
        domestic: bool,
    ) -> None:
        duration = _flight_minutes(origin.coords, dest.coords, stops=stops)
        carriers = _CARRIERS[:4] if not domestic else _CARRIERS[3:5]
        for carrier, alliance, on_time, tier in carriers:
            cabins = (
                ((CabinClass.ECONOMY, 1.0),)
                if domestic
                else ((CabinClass.ECONOMY, 1.0), (CabinClass.PREMIUM_ECONOMY, 1.9), (CabinClass.BUSINESS, 3.6))
            )
            for cabin, cabin_mult in cabins:
                brands = (
                    ((FareBrand.BASIC, 0.86, False, 28_000, False), (FareBrand.FLEX, 1.58, True, 0, True))
                    if domestic
                    else (
                        (FareBrand.BASIC, 0.86, False, 28_000, False),
                        (FareBrand.STANDARD, 1.00, False, 12_000, True),
                        (FareBrand.FLEX, 1.58, True, 0, True),
                    )
                )
                for brand, brand_mult, refundable, change_fee, bag in brands:
                    if cabin is CabinClass.BUSINESS and brand is FareBrand.BASIC:
                        continue  # airlines do not sell a basic business fare
                    noise = float(rng.normal(1.0, 0.11))
                    price = int(
                        base_price * tier * cabin_mult * brand_mult * season
                        * lead_time_multiplier(lead_days) * max(0.6, noise)
                    )
                    # Self-transfer only exists on connecting itineraries, is meaningfully
                    # cheaper, and carries no protection. The discount is the bait.
                    self_transfer = bool(stops > 0 and brand is FareBrand.BASIC and rng.random() < 0.45)
                    if self_transfer:
                        price = int(price * 0.72)
                    scarcity = 1.0 if brand is FareBrand.FLEX else 0.45
                    seats = int(
                        max(0, rng.poisson(6.0 * scarcity * (1.6 if cabin is CabinClass.BUSINESS else 1.0)))
                    )
                    jitter = int(rng.integers(-45, 46))
                    departure = depart + timedelta(minutes=jitter)
                    out.append(
                        Flight(
                            key=f"fl_{len(out):05d}",
                            carrier=carrier,
                            flight_number=f"{carrier}{int(rng.integers(100, 999))}",
                            origin=origin.key,
                            destination=dest.key,
                            departure_utc=departure,
                            arrival_utc=departure + timedelta(minutes=duration),
                            # DERIVED from the instant and the ORIGIN's offset. v1 applied the
                            # destination's offset here and was wrong on every long-haul leg.
                            local_departure_hour=(
                                (departure + timedelta(minutes=origin.utc_offset_minutes)).hour
                            ),
                            duration_minutes=duration,
                            stops=stops,
                            connection_buffer_minutes=(
                                int(rng.choice([40, 55, 70, 95, 140])) if stops else None
                            ),
                            cabin=cabin,
                            fare_brand=brand,
                            price=price,
                            refundable=refundable,
                            change_fee=change_fee,
                            checked_bag_included=bag,
                            seats_remaining=seats,
                            on_time_rate=_clamp(on_time + float(rng.normal(0, 0.035)), 0.5, 0.99),
                            alliance=alliance,
                            is_self_transfer=self_transfer,
                            # Four scattered middles is not the same product as four together.
                            adjacent_seats_available=int(max(0, min(seats, rng.binomial(max(seats, 1), 0.55)))),
                            wheelchair_service=bool(rng.random() < (0.95 if not domestic else 0.78)),
                            checked_bag_fee=0 if bag else int(rng.choice([4_500, 6_000, 7_500])),
                        )
                    )

    # --- Long-haul inbound: SFO into the first region's gateways ---
    for gateway in first_region_gateways:
        season = season_multiplier(_region_of(gateway.city), day_of_year)
        base = int(22_000 + haversine_km(origin_port.coords, gateway.coords) * 7)
        for hour, stops in ((1, 0), (11, 0), (17, 1)):
            emit(origin_port, gateway, start - timedelta(days=1) + timedelta(hours=hour),
                 base, stops=stops, season=season, domestic=False)

    # --- Inter-region hop, for the open-jaw case ---
    if len(chosen) > 1:
        for a in first_region_gateways:
            for b in last_region_gateways:
                season = season_multiplier(chosen[-1], day_of_year)
                base = int(12_000 + haversine_km(a.coords, b.coords) * 6.5)
                for day, hour in ((nights // 2, 9), (nights // 2, 15)):
                    emit(a, b, start + timedelta(days=day, hours=hour), base,
                         stops=0, season=season, domestic=False)

    # --- Long-haul outbound: last region's gateways back to SFO ---
    ret = start + timedelta(days=nights)
    for gateway in (last_region_gateways or gateways):
        season = season_multiplier(_region_of(gateway.city), day_of_year)
        base = int(22_000 + haversine_km(gateway.coords, origin_port.coords) * 7)
        for hour, stops in ((2, 0), (9, 0), (16, 1)):
            emit(gateway, origin_port, ret + timedelta(hours=hour), base,
                 stops=stops, season=season, domestic=False)

    # --- Domestic legs, only where they beat the ground option ---
    for region in chosen:
        primary = {spec.airports[0] for spec in CITY_SPECS if spec.region is region and spec.airports}
        ports = [airports[k] for k in primary if k in airports]
        season = season_multiplier(region, day_of_year)
        for a in ports:
            for b in ports:
                if a.key == b.key or a.city is b.city:
                    continue
                km = haversine_km(a.coords, b.coords)
                if km < 300:
                    continue  # the train wins; a 25-minute flight is not a real option
                base = int(4_000 + km * 9)
                for day in (2, max(3, nights - 3)):
                    for hour in (8, 16):
                        emit(a, b, start + timedelta(days=day, hours=hour), base,
                             stops=0, season=season, domestic=True)

    return tuple(out)


_REGION_OF: dict[CityCode, Region] = {spec.code: spec.region for spec in CITY_SPECS}


def _region_of(city: CityCode) -> Region | None:
    """The region a city belongs to, or None for the client's home.

    None rather than a default: the home airport is not in any region, and defaulting it
    into one is exactly how it ended up being generated as its own destination.
    """
    return _REGION_OF.get(city)


def _generate_ground(
    rng: np.random.Generator,
    cities: dict[CityCode, City],
    start: datetime,
    nights: int,
    day_of_year: int,
) -> tuple[GroundLeg, ...]:
    """Scheduled surface legs, both directions, on each corridor's real local timetable.

    Departure hours are LOCAL and converted through the origin city's frozen offset, which is
    what makes an overnight bus actually depart at night.
    """
    out: list[GroundLeg] = []
    for spec in CORRIDOR_SPECS:
        if spec.a not in cities or spec.b not in cities:
            continue
        region = cities[spec.a].region
        season = season_multiplier(region, day_of_year)
        for origin_city, destination_city in ((spec.a, spec.b), (spec.b, spec.a)):
            offset = cities[origin_city].utc_offset_minutes
            for day in range(nights + 1):
                for hour in spec.departures_local:
                    local = start.replace(tzinfo=None) + timedelta(days=day)
                    local = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=hour)
                    departure = (local - timedelta(minutes=offset)).replace(tzinfo=UTC)
                    reserved = bool(rng.random() < 0.6)
                    price = int(
                        spec.base_price_cents
                        * SEASON_SPECS[region].price_level
                        * season
                        * (1.18 if reserved and spec.mode is TransportMode.RAIL else 1.0)
                        * float(rng.normal(1.0, 0.05))
                    )
                    out.append(
                        GroundLeg(
                            key=f"gr_{len(out):05d}",
                            mode=spec.mode,
                            operator=spec.operator,
                            origin_city=origin_city,
                            destination_city=destination_city,
                            departure_utc=departure,
                            arrival_utc=departure + timedelta(minutes=spec.minutes),
                            duration_minutes=spec.minutes,
                            price=max(100, price),
                            refundable=reserved or spec.mode is TransportMode.CAR_HIRE,
                            seats_remaining=int(rng.integers(2, 44)),
                            on_time_rate=_clamp(float(rng.normal(spec.on_time, 0.02)), 0.5, 0.999),
                            step_free=spec.step_free,
                            luggage_allowance=spec.luggage_allowance,
                            weather_sensitive=spec.weather_sensitive,
                            requires_driver=spec.requires_driver,
                            booking_lead_days=spec.booking_lead_days,
                        )
                    )
    return tuple(out)


def _generate_lodging(
    rng: np.random.Generator,
    city_specs: list[CitySpec],
    districts_by_city: dict[CityCode, list[District]],
    start: datetime,
    day_of_year: int,
) -> tuple[tuple[Hotel, ...], tuple[RoomOffer, ...]]:
    """Properties of seven kinds, priced by quality x district x season x region.

    Two things carry most of the difficulty:

    **The rating trap, kept from v1.** `guest_rating` is drawn around a star-implied mean with
    variance shrinking as review count grows, so sparsely-reviewed properties scatter. Because
    price tracks stars rather than displayed rating, the lucky small-sample hotel is both cheap
    and apparently excellent — sorting on rating systematically selects it and overpays.

    **Accessibility follows the district.** A property's chance of being step-free is the
    district's `step_free` share, so accessibility clusters geographically the way it does in
    reality. A wheelchair user who also wants Venice or Gion is a genuinely constrained
    problem, not a filtered list.
    """
    hotels: list[Hotel] = []
    offers: list[RoomOffer] = []
    amenity_pool = (
        "wifi", "onsen", "gym", "breakfast_buffet", "airport_shuttle", "laundry",
        "kitchenette", "family_rooms", "24h_desk", "luggage_storage", "pool", "bar",
    )
    types = [t for t, *_ in _LODGING_MIX]
    type_p = np.array([share for _, share, *_ in _LODGING_MIX], dtype=float)
    type_p /= type_p.sum()
    by_type = {m[0]: m[2:] for m in _LODGING_MIX}

    for spec in city_specs:
        price_level = SEASON_SPECS[spec.region].price_level
        season = season_multiplier(spec.region, day_of_year)
        district_specs = {district_key(d.name): d for d in DISTRICTS_BY_CITY[spec.code]}
        for district in districts_by_city[spec.code]:
            raw = district_specs[district.key]
            for _ in range(int(rng.integers(4, 8))):
                lodging_type = types[int(rng.choice(len(types), p=type_p))]
                price_mult, max_occ, ci_from, ci_until, access_bias = by_type[lodging_type]

                stars = int(_clamp(rng.integers(1, 6) + (1 if district.price_index > 1.4 else 0), 1, 5))
                review_count = int(rng.pareto(1.1) * 220) + 8
                rating_sd = _clamp(2.4 / math.sqrt(review_count), 0.06, 1.5)
                guest_rating = _clamp(float(rng.normal(5.6 + 0.72 * stars, rating_sd)), 3.0, 10.0)

                # The district decides how likely step-free access is; the property type
                # nudges it. A capsule block is never wheelchair accessible whatever the
                # neighbourhood, and a modern resort usually is.
                access: set[AccessibilityNeed] = set()
                if rng.random() < raw.step_free * access_bias:
                    access.add(AccessibilityNeed.STEP_FREE)
                    if rng.random() < 0.72:
                        access.add(AccessibilityNeed.WHEELCHAIR)
                if stars >= 3 and rng.random() < 0.8:
                    access.add(AccessibilityNeed.ELEVATOR)
                if rng.random() < 0.45:
                    access.add(AccessibilityNeed.LOW_STIMULUS if district.quiet >= 6 else AccessibilityNeed.ELEVATOR)
                if lodging_type is LodgingType.CAPSULE:
                    access.discard(AccessibilityNeed.WHEELCHAIR)

                dietary: set[DietaryNeed] = set()
                for need, p in (
                    (DietaryNeed.VEGETARIAN, 0.62), (DietaryNeed.VEGAN, 0.28),
                    (DietaryNeed.GLUTEN_FREE, 0.34), (DietaryNeed.HALAL, 0.22),
                    (DietaryNeed.NUT_ALLERGY, 0.40),
                ):
                    if rng.random() < p:
                        dietary.add(need)

                key = f"ht_{len(hotels):04d}"
                words = _HOTEL_WORDS[spec.region]
                hotels.append(
                    Hotel(
                        key=key,
                        name=f"{rng.choice(words)} {rng.choice(_HOTEL_SUFFIX[lodging_type])} {district.name}",
                        city=spec.code,
                        district=district.key,
                        coords=Coordinates(
                            lat=district.coords.lat + float(rng.normal(0, 0.006)),
                            lon=district.coords.lon + float(rng.normal(0, 0.006)),
                        ),
                        star_rating=stars,
                        guest_rating=round(guest_rating, 1),
                        review_count=review_count,
                        amenities=frozenset(
                            rng.choice(amenity_pool, size=int(rng.integers(2, 7)), replace=False).tolist()
                        ),
                        minutes_to_station=int(_clamp(rng.gamma(2.0, 3.0), 1, 25)),
                        lodging_type=lodging_type,
                        accessibility=frozenset(access),
                        allows_service_animal=bool(rng.random() < 0.62),
                        dietary_supported=frozenset(dietary),
                        check_in_from_hour=ci_from,
                        # A 24-hour desk is a real amenity and a minority of properties have
                        # one. The rest is where "land at 23:40" becomes a lost room.
                        check_in_until_hour=(
                            24 if (ci_until >= 24 and rng.random() < 0.55) else int(rng.choice([21, 22, 23]))
                        ),
                        min_nights=int(rng.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.14, 0.1, 0.14, 0.07, 0.05])),
                    )
                )

                base_nightly = int(spec.base_nightly_cents * price_mult * (0.55 + 0.22 * stars) * price_level)
                for board, board_mult in (
                    (BoardType.ROOM_ONLY, 1.0),
                    (BoardType.BREAKFAST, 1.14),
                    (BoardType.HALF_BOARD, 1.34),
                ):
                    # Noise drawn ONCE per (hotel, board), then the plan multiplier applied on
                    # top. Drawing per plan let noise invert the ordering so a non-refundable
                    # room cost more than the flexible one, destroying the act I / act II
                    # tradeoff the rate-plan structure exists to create.
                    noise = float(rng.normal(1.0, 0.13))
                    cot = bool(max_occ >= 2 and rng.random() < 0.45)
                    for plan, plan_mult in (
                        (RatePlan.NON_REFUNDABLE, 0.85),
                        (RatePlan.FREE_CANCELLATION, 1.0),
                    ):
                        price = int(
                            base_nightly * district.price_index * board_mult * plan_mult
                            * season * max(0.55, noise)
                        )
                        offers.append(
                            RoomOffer(
                                key=f"ro_{len(offers):05d}",
                                hotel=key,
                                board=board,
                                rate_plan=plan,
                                price_per_night=max(400, price),
                                max_occupancy=max_occ,
                                free_cancellation_until=(
                                    start - timedelta(days=2) if plan is RatePlan.FREE_CANCELLATION else None
                                ),
                                rooms_remaining=int(
                                    max(0, rng.poisson(3.2 if plan is RatePlan.NON_REFUNDABLE else 5.0))
                                ),
                                cot_available=cot,
                                cot_price_per_night=int(price * 0.22) if cot else 0,
                                advance_purchase_days=int(rng.choice([0, 0, 0, 7, 14], p=[0.6, 0.15, 0.1, 0.1, 0.05])),
                            )
                        )

    return tuple(hotels), tuple(offers)


def _generate_activities(
    rng: np.random.Generator,
    city_specs: list[CitySpec],
    districts_by_city: dict[CityCode, list[District]],
    start: datetime,
    day_of_year: int,
) -> tuple[Activity, ...]:
    """Hand-authored anchors plus a procedural long tail.

    v1 shipped eighteen activities, which is small enough that an agent can enumerate them.
    The anchors carry the facts a generator cannot invent honestly — that Fushimi Inari is a
    hill, that a kaiseki table books three weeks out. The tail supplies the volume that any
    real marketplace has, so the catalogue scales with the world rather than being a constant.
    """
    out: list[Activity] = []
    month = start.month

    for spec in city_specs:
        price_level = SEASON_SPECS[spec.region].price_level
        season = season_multiplier(spec.region, day_of_year)
        districts = districts_by_city[spec.code]

        for anchor in ACTIVITIES_BY_CITY[spec.code]:
            review_count = int(rng.pareto(1.2) * 150) + 12
            rating_sd = _clamp(2.2 / math.sqrt(review_count), 0.05, 1.4)
            out.append(
                Activity(
                    key=f"ac_{len(out):04d}",
                    name=anchor.name,
                    city=spec.code,
                    district=anchor.district,
                    category=anchor.category,
                    duration_minutes=anchor.duration,
                    price_per_person=int(max(300, rng.gamma(2.2, 1_900)) * season * price_level),
                    rating=round(_clamp(float(rng.normal(8.3, rating_sd)), 4.0, 10.0), 1),
                    opens_hour=anchor.opens,
                    closes_hour=anchor.closes,
                    closed_weekdays=frozenset({anchor.closed_weekday} if anchor.closed_weekday else set()),
                    booking_required=bool(anchor.lead_time_days > 0 or rng.random() < 0.45),
                    daily_capacity=int(rng.integers(4, 40)),
                    physical_intensity=anchor.intensity,
                    accessible=anchor.accessible,
                    low_stimulus=anchor.low_stimulus,
                    min_age=anchor.min_age,
                    max_age=anchor.max_age,
                    max_group_size=anchor.max_group,
                    lead_time_days=anchor.lead_time_days,
                    weather_dependent=anchor.weather_dependent,
                    dietary_supported=frozenset(anchor.dietary),
                    season_months=frozenset(anchor.season_months),
                )
            )

        for _ in range(int(rng.integers(10, 16))):
            district = districts[int(rng.integers(0, len(districts)))]
            category = _tail_category(rng, district)
            theme = str(rng.choice(TAIL_THEMES[category]))
            fmt = str(rng.choice(TAIL_FORMATS))
            opens = int(rng.choice([8, 9, 10, 11, 14, 16, 18]))
            duration = int(rng.choice([60, 90, 120, 150, 180]))
            # The window must be able to contain the activity or it can never be booked, and
            # `assert_world_coherent` treats an unbookable listing as a generator fault.
            closes = min(24, max(opens + math.ceil(duration / 60) + 1, opens + int(rng.integers(3, 8))))
            review_count = int(rng.pareto(1.3) * 90) + 5
            rating_sd = _clamp(2.2 / math.sqrt(review_count), 0.05, 1.6)
            out.append(
                Activity(
                    key=f"ac_{len(out):04d}",
                    name=f"{district.name} {theme} {fmt}",
                    city=spec.code,
                    district=district.key,
                    category=category,
                    duration_minutes=duration,
                    price_per_person=int(max(200, rng.gamma(1.9, 1_400)) * season * price_level),
                    rating=round(_clamp(float(rng.normal(7.7, rating_sd)), 3.5, 10.0), 1),
                    opens_hour=opens,
                    closes_hour=closes,
                    closed_weekdays=frozenset({int(rng.integers(1, 8))} if rng.random() < 0.3 else set()),
                    booking_required=bool(rng.random() < 0.35),
                    daily_capacity=int(rng.integers(2, 30)),
                    physical_intensity=int(rng.integers(1, 8)),
                    accessible=bool(rng.random() < 0.35 + 0.6 * _step_free_share(district)),
                    low_stimulus=bool(rng.random() < 0.55),
                    min_age=int(rng.choice([0, 0, 0, 6, 12, 18], p=[0.6, 0.1, 0.05, 0.1, 0.1, 0.05])),
                    max_group_size=int(rng.choice([6, 8, 12, 20, 99], p=[0.15, 0.2, 0.25, 0.2, 0.2])),
                    lead_time_days=int(rng.choice([0, 0, 0, 3, 7], p=[0.6, 0.15, 0.05, 0.12, 0.08])),
                    weather_dependent=bool(category in (ActivityCategory.NATURE,) or rng.random() < 0.15),
                    dietary_supported=frozenset(
                        d for d in DietaryNeed if rng.random() < 0.4
                    ) if category is ActivityCategory.FOOD else frozenset(),
                    season_months=(
                        frozenset(range(1, 13))
                        if rng.random() < 0.85
                        else frozenset(((month + k - 1) % 12) + 1 for k in range(-3, 4))
                    ),
                )
            )

    return tuple(out)


def _tail_category(rng: np.random.Generator, district: District) -> ActivityCategory:
    """Draw a category weighted by what the district is actually like.

    A nightlife district produces bar tours; a culturally dense one produces temple walks.
    Uniform categories would make districts interchangeable, and interchangeable districts
    make the location preference meaningless.
    """
    weights = {
        ActivityCategory.FOOD: 1.0 + district.dining_density / 4.0,
        ActivityCategory.HISTORY: 0.5 + district.cultural_density / 3.5,
        ActivityCategory.MUSEUM: 0.4 + district.cultural_density / 5.0,
        ActivityCategory.NATURE: 0.4 + district.quiet / 4.5,
        ActivityCategory.NIGHTLIFE: 0.2 + district.nightlife / 3.0,
        ActivityCategory.SHOPPING: 0.4 + district.walkability / 6.0,
        ActivityCategory.FAMILY: 0.5,
    }
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _step_free_share(district: District) -> float:
    for spec in DISTRICTS_BY_CITY[district.city]:
        if district_key(spec.name) == district.key:
            return spec.step_free
    return 0.5


def _visa_rules(city_specs: list[CitySpec]) -> tuple[VisaRule, ...]:
    """Policy for every passport against every country in this world."""
    countries = sorted({spec.country for spec in city_specs})
    out: list[VisaRule] = []
    for nationality in Nationality:
        for country in countries:
            required, lead, fee = VISA_POLICY.get((nationality, country), (True, 30, 9_800))
            out.append(
                VisaRule(
                    nationality=nationality,
                    country=country,
                    visa_required=required,
                    lead_time_days=lead,
                    fee_cents=fee,
                )
            )
    return tuple(out)
