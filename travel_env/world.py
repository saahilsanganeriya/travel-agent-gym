"""Seeded synthetic-world generator for the Tokyo-Kyoto-Osaka corridor.

The quality of the synthetic data is itself a signal, so the goal here is
*correlated structure*, not random values in plausible ranges. Three properties are
deliberate:

1. **Price correlates with quality, but noisily.** `price = base(quality, season, lead
   time, demand) * noise`. If the correlation were deterministic, "always pick the
   highest-rated hotel" would be a free win rather than a tradeoff; if there were no
   correlation, quality would be unlearnable. The noise term is what makes it a judgement
   call. Review count varies too, so a 9.8 from 11 reviews is weaker evidence than an 8.6
   from 2,400 — an agent that ignores sample size will overpay.
2. **Geography is real.** Districts carry coordinates and travel times are derived from
   haversine distance with a transit-mode multiplier, so an itinerary that requires
   teleportation across Tokyo cannot be constructed. Feasibility is enforced by the
   simulator, never priced into the reward.
3. **Availability is uneven.** Seats and rooms deplete on a Zipf-ish popularity draw, so
   the cheap well-located options are frequently the ones that are gone — which is what
   creates genuine search pressure.

The Tokyo-Kyoto-Osaka corridor is chosen because it is rail-connected. That makes
"fly or take the Shinkansen" a real robustness tradeoff rather than a formality: rail is
slower and pricier per km but almost never cancels, which only matters once the disruption
engine runs in act II.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from travel_env.contracts import (
    Activity,
    ActivityCategory,
    Airport,
    BoardType,
    CabinClass,
    City,
    CityCode,
    Coordinates,
    District,
    FareBrand,
    Flight,
    Hotel,
    RailLeg,
    RatePlan,
    RoomOffer,
    World,
)

_EARTH_RADIUS_KM = 6371.0

# Real coordinates and plausible district character. Values are hand-set rather than drawn
# so that geography stays consistent across every seed — only prices, availability and
# inventory vary. An agent can learn that Shibuya is central; it cannot learn that a
# particular hotel is cheap.
_DISTRICT_SPECS: tuple[tuple[CityCode, str, float, float, int, int, int, int, int, float], ...] = (
    # city, name, lat, lon, walkability, nightlife, dining, cultural, quiet, price_index
    (CityCode.TOKYO, "Shinjuku", 35.6938, 139.7034, 9, 9, 9, 4, 1, 1.35),
    (CityCode.TOKYO, "Shibuya", 35.6595, 139.7005, 9, 9, 8, 3, 1, 1.45),
    (CityCode.TOKYO, "Ginza", 35.6717, 139.7650, 8, 5, 8, 6, 4, 1.70),
    (CityCode.TOKYO, "Asakusa", 35.7148, 139.7967, 7, 4, 6, 9, 6, 0.95),
    (CityCode.TOKYO, "Ueno", 35.7138, 139.7770, 7, 5, 6, 8, 5, 0.88),
    (CityCode.TOKYO, "Marunouchi", 35.6812, 139.7671, 8, 3, 6, 5, 6, 1.60),
    (CityCode.TOKYO, "Kichijoji", 35.7030, 139.5800, 8, 6, 7, 4, 7, 0.82),
    (CityCode.KYOTO, "Gion", 35.0037, 135.7788, 8, 5, 7, 9, 5, 1.55),
    (CityCode.KYOTO, "Higashiyama", 34.9948, 135.7850, 7, 3, 5, 9, 7, 1.30),
    (CityCode.KYOTO, "Kyoto Station", 34.9858, 135.7588, 9, 4, 6, 4, 3, 1.10),
    (CityCode.KYOTO, "Arashiyama", 35.0094, 135.6668, 5, 2, 4, 9, 9, 1.05),
    (CityCode.OSAKA, "Namba", 34.6659, 135.5010, 9, 9, 9, 4, 1, 1.20),
    (CityCode.OSAKA, "Umeda", 34.7025, 135.4959, 9, 7, 8, 3, 2, 1.25),
    (CityCode.OSAKA, "Tennoji", 34.6465, 135.5133, 7, 5, 7, 6, 4, 0.85),
)

_CITY_META: dict[CityCode, tuple[str, str, str, float, float, tuple[str, ...]]] = {
    CityCode.TOKYO: ("Tokyo", "JP", "Asia/Tokyo", 35.6762, 139.6503, ("HND", "NRT")),
    CityCode.KYOTO: ("Kyoto", "JP", "Asia/Tokyo", 35.0116, 135.7681, ()),
    CityCode.OSAKA: ("Osaka", "JP", "Asia/Tokyo", 34.6937, 135.5023, ("KIX",)),
}

_AIRPORT_SPECS: tuple[tuple[str, str, CityCode, float, float, int], ...] = (
    ("HND", "Tokyo Haneda", CityCode.TOKYO, 35.5494, 139.7798, 35),
    ("NRT", "Tokyo Narita", CityCode.TOKYO, 35.7720, 140.3929, 95),
    ("KIX", "Kansai International", CityCode.OSAKA, 34.4342, 135.2328, 65),
    ("SFO", "San Francisco", CityCode.TOKYO, 37.6213, -122.3790, 30),  # origin only
)

# (carrier, on-time rate, price tier). Reliability and price are correlated but not
# perfectly: the cheapest carrier is the least reliable, but the priciest is not the most
# reliable, so "pay more for reliability" has a knee rather than being monotone.
_CARRIERS: tuple[tuple[str, float, float], ...] = (
    ("JL", 0.93, 1.25),
    ("NH", 0.92, 1.22),
    ("UA", 0.81, 1.00),
    ("ZG", 0.72, 0.78),
    ("MM", 0.68, 0.62),
)

_ACTIVITY_SPECS: tuple[tuple[str, CityCode, str, ActivityCategory, int, int, int, int, int], ...] = (
    # name, city, district, category, duration_min, opens, closes, closed_weekday, intensity
    ("Tsukiji Outer Market Food Walk", CityCode.TOKYO, "ginza", ActivityCategory.FOOD, 180, 6, 14, 0, 4),
    ("Shinjuku Izakaya Crawl", CityCode.TOKYO, "shinjuku", ActivityCategory.NIGHTLIFE, 210, 18, 23, 0, 3),
    ("Senso-ji Temple & Nakamise", CityCode.TOKYO, "asakusa", ActivityCategory.HISTORY, 120, 6, 17, 0, 3),
    ("Tokyo National Museum", CityCode.TOKYO, "ueno", ActivityCategory.MUSEUM, 150, 9, 17, 1, 2),
    ("Ramen Making Workshop", CityCode.TOKYO, "shibuya", ActivityCategory.FOOD, 150, 10, 16, 3, 2),
    ("Inokashira Park & Ghibli Area", CityCode.TOKYO, "kichijoji", ActivityCategory.NATURE, 180, 8, 18, 2, 4),
    ("teamLab Digital Art", CityCode.TOKYO, "ginza", ActivityCategory.MUSEUM, 180, 10, 19, 2, 3),
    ("Imperial Palace Gardens", CityCode.TOKYO, "marunouchi", ActivityCategory.HISTORY, 120, 9, 16, 1, 4),
    ("Fushimi Inari Night Hike", CityCode.KYOTO, "higashiyama", ActivityCategory.NATURE, 180, 16, 22, 0, 7),
    ("Gion Geisha District Walk", CityCode.KYOTO, "gion", ActivityCategory.HISTORY, 120, 15, 21, 0, 3),
    ("Kaiseki Dinner", CityCode.KYOTO, "gion", ActivityCategory.FOOD, 150, 18, 21, 1, 1),
    ("Arashiyama Bamboo Grove", CityCode.KYOTO, "arashiyama", ActivityCategory.NATURE, 150, 7, 17, 0, 5),
    ("Tea Ceremony Experience", CityCode.KYOTO, "higashiyama", ActivityCategory.HISTORY, 90, 10, 16, 2, 1),
    ("Nishiki Market Tasting", CityCode.KYOTO, "kyoto_station", ActivityCategory.FOOD, 120, 9, 17, 2, 3),
    ("Dotonbori Street Food", CityCode.OSAKA, "namba", ActivityCategory.FOOD, 150, 11, 23, 0, 3),
    ("Osaka Castle", CityCode.OSAKA, "tennoji", ActivityCategory.HISTORY, 150, 9, 17, 0, 4),
    ("Umeda Sky Building", CityCode.OSAKA, "umeda", ActivityCategory.MUSEUM, 90, 10, 22, 0, 2),
    ("Kuromon Market & Kushikatsu", CityCode.OSAKA, "namba", ActivityCategory.FOOD, 120, 9, 18, 3, 2),
)

_HOTEL_BRANDS = (
    "Kohaku", "Sakura", "Mitsu", "Aoi", "Hoshi", "Yanagi", "Tsuru", "Kiri",
    "Momiji", "Suzume", "Hanabi", "Kaze",
)
_HOTEL_TYPES = ("Hotel", "Ryokan", "Inn", "Residences", "Suites", "House")


def _district_key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def haversine_km(a: Coordinates, b: Coordinates) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def transfer_minutes(a: Coordinates, b: Coordinates) -> int:
    """Door-to-door transit time. The 2.2 detour factor and 22 km/h effective speed are
    the difference between straight-line distance and how a metro system actually moves
    you; the 8-minute constant is platform and walking overhead. This is what makes an
    over-packed itinerary infeasible rather than merely low-scoring."""
    km = haversine_km(a, b)
    if km < 1.2:
        return max(5, int(round(km * 13)))  # walking pace
    return int(round(8 + (km * 2.2) / 22.0 * 60))


def _season_multiplier(day_of_year: int) -> float:
    """Cherry blossom (late Mar-Apr) and autumn colour (Nov) are the demand peaks; the
    Jan-Feb trough is the cheapest. Modelled as two Gaussians over a flat base."""
    spring = 0.42 * math.exp(-(((day_of_year - 95) / 18.0) ** 2))
    autumn = 0.34 * math.exp(-(((day_of_year - 318) / 16.0) ** 2))
    trough = -0.16 * math.exp(-(((day_of_year - 40) / 25.0) ** 2))
    return 1.0 + spring + autumn + trough


def _lead_time_multiplier(days_ahead: int) -> float:
    """Airline pricing: cheap in the 21-90 day window, punitive inside two weeks."""
    if days_ahead <= 3:
        return 2.10
    if days_ahead <= 14:
        return 1.45
    if days_ahead <= 21:
        return 1.15
    if days_ahead <= 90:
        return 1.00
    return 1.08  # schedules not fully loaded yet


def generate_world(seed: int, *, trip_start: datetime | None = None, nights: int = 7) -> World:
    """Build one complete, self-consistent world. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    start = trip_start or datetime(2027, 4, 3, tzinfo=UTC)
    day_of_year = start.timetuple().tm_yday
    season = _season_multiplier(day_of_year)

    districts_by_city: dict[CityCode, list[District]] = {c: [] for c in CityCode}
    for city, name, lat, lon, walk, night, dining, cultural, quiet, price_index in _DISTRICT_SPECS:
        districts_by_city[city].append(
            District(
                key=_district_key(name), name=name, city=city,
                coords=Coordinates(lat=lat, lon=lon),
                walkability=walk, nightlife=night, dining_density=dining,
                cultural_density=cultural, quiet=quiet, price_index=price_index,
            )
        )

    cities = tuple(
        City(
            code=code, name=meta[0], country=meta[1], timezone=meta[2],
            centre=Coordinates(lat=meta[3], lon=meta[4]),
            districts=tuple(districts_by_city[code]),
            airports=meta[5],
        )
        for code, meta in _CITY_META.items()
    )
    airports = tuple(
        Airport(key=k, name=n, city=c, coords=Coordinates(lat=lat, lon=lon), minutes_to_centre=m)
        for k, n, c, lat, lon, m in _AIRPORT_SPECS
    )

    flights = _generate_flights(rng, start, nights, season)
    rail = _generate_rail(rng, start, nights)
    hotels, room_offers = _generate_lodging(rng, districts_by_city, season, day_of_year, nights)
    activities = _generate_activities(rng, season)

    return World(
        seed=seed, generated_at=datetime.now(tz=UTC), cities=cities, airports=airports,
        flights=flights, rail=rail, hotels=hotels, room_offers=room_offers, activities=activities,
    )


def _generate_flights(
    rng: np.random.Generator, start: datetime, nights: int, season: float
) -> tuple[Flight, ...]:
    """Long-haul inbound/outbound plus the domestic HND-KIX shuttle.

    Fare brands are the load-bearing detail: BASIC is cheapest and non-refundable with a
    high change fee, FLEX costs ~1.6x and is fully refundable. In act I those look like
    the same seat at different prices. In act II they are not.
    """
    out: list[Flight] = []
    lead_days = 45

    def add_leg(
        origin: str, destination: str, dep: datetime, duration: int, base_price: int,
        stops: int, buffer_minutes: int | None, idx: int,
    ) -> None:
        for carrier, on_time, tier in _CARRIERS:
            for cabin, cabin_mult in (
                (CabinClass.ECONOMY, 1.0),
                (CabinClass.PREMIUM_ECONOMY, 1.9),
                (CabinClass.BUSINESS, 3.6),
            ):
                for brand, brand_mult, refundable, change_fee, bag in (
                    (FareBrand.BASIC, 0.86, False, 28_000, False),
                    (FareBrand.STANDARD, 1.00, False, 12_000, True),
                    (FareBrand.FLEX, 1.58, True, 0, True),
                ):
                    if cabin is CabinClass.BUSINESS and brand is FareBrand.BASIC:
                        continue  # airlines don't sell a basic business fare
                    noise = float(rng.normal(1.0, 0.11))
                    price = int(
                        base_price * tier * cabin_mult * brand_mult * season
                        * _lead_time_multiplier(lead_days) * max(0.6, noise)
                    )
                    # Availability is uneven and skewed against the cheap seats: a Zipf
                    # draw means the well-priced options are often the sold-out ones.
                    scarcity = 1.0 if brand is FareBrand.FLEX else 0.45
                    seats = int(max(0, rng.poisson(6.0 * scarcity * (1.6 if cabin is CabinClass.BUSINESS else 1.0))))
                    jitter = int(rng.integers(-90, 91))
                    out.append(
                        Flight(
                            key=f"fl_{len(out):05d}",
                            carrier=carrier,
                            flight_number=f"{carrier}{int(rng.integers(100, 999))}",
                            origin=origin, destination=destination,
                            departure_utc=dep + timedelta(minutes=jitter),
                            arrival_utc=dep + timedelta(minutes=jitter + duration),
                            local_departure_hour=(dep.hour + 9) % 24,  # JST
                            duration_minutes=duration, stops=stops,
                            connection_buffer_minutes=buffer_minutes,
                            cabin=cabin, fare_brand=brand, price=price,
                            refundable=refundable, change_fee=change_fee,
                            checked_bag_included=bag, seats_remaining=seats,
                            on_time_rate=float(np.clip(on_time + rng.normal(0, 0.035), 0.5, 0.99)),
                        )
                    )

    for idx, hour in enumerate((1, 11, 17)):
        add_leg("SFO", "HND", start - timedelta(days=1) + timedelta(hours=hour), 660, 78_000, 0, None, idx)
        # One-stop options are cheaper and carry a connection buffer that act II can break.
        add_leg("SFO", "NRT", start - timedelta(days=1) + timedelta(hours=hour + 2), 810, 61_000, 1,
                int(rng.choice([45, 55, 70, 95, 140])), idx)

    ret = start + timedelta(days=nights)
    for idx, hour in enumerate((2, 9, 16)):
        add_leg("HND", "SFO", ret + timedelta(hours=hour), 585, 76_000, 0, None, idx)
        add_leg("KIX", "SFO", ret + timedelta(hours=hour), 620, 72_000, 0, None, idx)
        add_leg("NRT", "SFO", ret + timedelta(hours=hour + 1), 780, 58_000, 1,
                int(rng.choice([50, 65, 90, 130])), idx)

    for day in range(1, nights):
        for hour in (8, 13, 19):
            add_leg("HND", "KIX", start + timedelta(days=day, hours=hour), 80, 14_000, 0, None, day)

    return tuple(out)


def _generate_rail(rng: np.random.Generator, start: datetime, nights: int) -> tuple[RailLeg, ...]:
    """Shinkansen. Slower per-km than flying and not much cheaper, but on_time_rate is
    ~0.995 — the robustness option, which only pays off once disruptions run."""
    legs: list[RailLeg] = []
    corridor = (
        (CityCode.TOKYO, CityCode.KYOTO, 140, 1_380),
        (CityCode.KYOTO, CityCode.TOKYO, 140, 1_380),
        (CityCode.KYOTO, CityCode.OSAKA, 15, 420),
        (CityCode.OSAKA, CityCode.KYOTO, 15, 420),
        (CityCode.TOKYO, CityCode.OSAKA, 150, 1_480),
        (CityCode.OSAKA, CityCode.TOKYO, 150, 1_480),
    )
    for day in range(nights + 1):
        for hour in (7, 9, 11, 14, 16, 18, 20):
            for origin, destination, minutes, base in corridor:
                dep = start + timedelta(days=day, hours=hour)
                reserved = bool(rng.random() < 0.6)
                legs.append(
                    RailLeg(
                        key=f"rl_{len(legs):05d}", operator="JR Central",
                        origin_city=origin, destination_city=destination,
                        departure_utc=dep, arrival_utc=dep + timedelta(minutes=minutes),
                        duration_minutes=minutes,
                        price=int(base * (1.18 if reserved else 1.0) * float(rng.normal(1.0, 0.03))),
                        refundable=reserved,
                        seats_remaining=int(rng.integers(4, 40)),
                        on_time_rate=float(np.clip(rng.normal(0.995, 0.004), 0.95, 0.999)),
                    )
                )
    return tuple(legs)


def _generate_lodging(
    rng: np.random.Generator,
    districts_by_city: dict[CityCode, list[District]],
    season: float,
    day_of_year: int,
    nights: int,
) -> tuple[tuple[Hotel, ...], tuple[RoomOffer, ...]]:
    """Hotels priced from quality x district x season, with noise.

    The rating model is the important part. `guest_rating` is drawn around a star-implied
    mean but with variance that shrinks as review_count grows, so highly-reviewed hotels
    cluster near their true quality while sparsely-reviewed ones scatter. An agent that
    sorts on rating alone will systematically pick lucky small-sample hotels and overpay
    for them — which is exactly the "always pick the highest-rated" exploit, defused by
    the data rather than by a penalty.
    """
    hotels: list[Hotel] = []
    offers: list[RoomOffer] = []
    amenity_pool = (
        "wifi", "onsen", "gym", "breakfast_buffet", "airport_shuttle", "laundry",
        "kitchenette", "family_rooms", "accessible", "24h_desk", "luggage_storage",
    )

    for city, districts in districts_by_city.items():
        if not districts:
            continue
        for district in districts:
            for _ in range(int(rng.integers(4, 8))):
                stars = int(np.clip(rng.integers(1, 6) + (1 if district.price_index > 1.4 else 0), 1, 5))
                review_count = int(rng.pareto(1.1) * 220) + 8
                # Confidence scales with sqrt(n): sparse reviews are noisy evidence.
                rating_sd = float(np.clip(2.4 / math.sqrt(review_count), 0.06, 1.5))
                guest_rating = float(np.clip(rng.normal(5.6 + 0.72 * stars, rating_sd), 3.0, 10.0))
                brand = str(rng.choice(_HOTEL_BRANDS))
                htype = str(rng.choice(_HOTEL_TYPES))
                key = f"ht_{len(hotels):04d}"
                hotels.append(
                    Hotel(
                        key=key, name=f"{brand} {htype} {district.name}", city=city,
                        district=district.key,
                        coords=Coordinates(
                            lat=district.coords.lat + float(rng.normal(0, 0.006)),
                            lon=district.coords.lon + float(rng.normal(0, 0.006)),
                        ),
                        star_rating=stars, guest_rating=round(guest_rating, 1),
                        review_count=review_count,
                        amenities=frozenset(
                            rng.choice(amenity_pool, size=int(rng.integers(2, 7)), replace=False).tolist()
                        ),
                        minutes_to_station=int(np.clip(rng.gamma(2.0, 3.0), 1, 25)),
                    )
                )

                base_nightly = 5_500 + 4_200 * stars
                for board, board_mult in (
                    (BoardType.ROOM_ONLY, 1.0),
                    (BoardType.BREAKFAST, 1.14),
                    (BoardType.HALF_BOARD, 1.34),
                ):
                    for plan, plan_mult in (
                        (RatePlan.NON_REFUNDABLE, 0.85),
                        (RatePlan.FREE_CANCELLATION, 1.0),
                    ):
                        noise = float(rng.normal(1.0, 0.13))
                        price = int(
                            base_nightly * district.price_index * board_mult * plan_mult
                            * season * max(0.55, noise)
                        )
                        offers.append(
                            RoomOffer(
                                key=f"ro_{len(offers):05d}", hotel=key, board=board, rate_plan=plan,
                                price_per_night=price,
                                max_occupancy=int(rng.integers(2, 5)),
                                free_cancellation_until=None,
                                rooms_remaining=int(max(0, rng.poisson(3.2 if plan is RatePlan.NON_REFUNDABLE else 5.0))),
                            )
                        )

    return tuple(hotels), tuple(offers)


def _generate_activities(rng: np.random.Generator, season: float) -> tuple[Activity, ...]:
    out: list[Activity] = []
    for name, city, district, category, duration, opens, closes, closed_wd, intensity in _ACTIVITY_SPECS:
        review_count = int(rng.pareto(1.2) * 150) + 12
        rating_sd = float(np.clip(2.2 / math.sqrt(review_count), 0.05, 1.4))
        out.append(
            Activity(
                key=f"ac_{len(out):04d}", name=name, city=city, district=district,
                category=category, duration_minutes=duration,
                price_per_person=int(max(0, rng.gamma(2.2, 1_900)) * season),
                rating=round(float(np.clip(rng.normal(8.1, rating_sd), 4.0, 10.0)), 1),
                opens_hour=opens, closes_hour=closes,
                closed_weekdays=frozenset({closed_wd} if closed_wd else set()),
                booking_required=bool(rng.random() < 0.45),
                daily_capacity=int(rng.integers(6, 60)),
                physical_intensity=intensity,
            )
        )
    return tuple(out)
