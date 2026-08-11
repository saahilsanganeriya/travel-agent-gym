"""The world's geography, hand-authored and constant across every seed.

Coordinates, timezones, district character, transport corridors, visa policy and seasonality
are *facts*, not draws. A seed varies prices, availability, ratings and inventory; it never
varies where Kyoto is or whether a Nigerian passport needs a visa for Japan. That split is
what makes the environment learnable: an agent can discover that Arashiyama is quiet and far
out, and cannot memorise that a particular hotel is cheap.

**Why three regions rather than one big one.** Widening a single country mostly adds more of
the same decision. Three regions add *structurally different* decisions, and it is structure
that a better agent exploits and a worse one does not:

| | Japan | Italy | Thailand |
|---|---|---|---|
| intercity | superb rail, flights rarely worth it | good rail on one spine, thin off it | poor rail; ferries and domestic air |
| price level | high | high | low — a budget goes three times as far |
| peak | sakura (Apr) and kōyō (Nov) | high summer | cool-dry (Dec-Feb) |
| the weather risk | typhoon, Aug-Oct | August closures | monsoon Jun-Oct, and it cancels ferries |
| visas | visa-free for some passports, weeks of lead for others | Schengen | visa-free or on-arrival for most |

A party of four on a mid-range budget can afford two weeks in Thailand or five days in
Japan, and which is the *right* answer depends on the client's hidden weights. That is a real
travel-agent decision and it does not exist in a one-country world.
"""

from __future__ import annotations

from typing import NamedTuple

from travel_env.contracts import (
    ActivityCategory,
    CityCode,
    DietaryNeed,
    Nationality,
    Region,
    TransportMode,
)


class DistrictSpec(NamedTuple):
    city: CityCode
    name: str
    lat: float
    lon: float
    walkability: int
    nightlife: int
    dining: int
    cultural: int
    quiet: int
    price_index: float
    step_free: float
    """Share of this district's stock that is step-free. Historic hill districts are low, and
    that is a real and unevenly-distributed constraint rather than a uniform coin flip."""


class CitySpec(NamedTuple):
    code: CityCode
    name: str
    region: Region
    country: str
    timezone: str
    lat: float
    lon: float
    airports: tuple[str, ...]
    base_nightly_cents: int
    """Mid-market room, room-only, before stars, district, season and noise."""


class AirportSpec(NamedTuple):
    key: str
    name: str
    city: CityCode
    lat: float
    lon: float
    minutes_to_centre: int
    country: str
    timezone: str
    min_connection_minutes: int
    intercontinental: bool
    """Whether the long-haul gateway flights use it. Regional fields are domestic-only."""


class CorridorSpec(NamedTuple):
    """A surface link between two cities, and how it is actually served."""

    a: CityCode
    b: CityCode
    mode: TransportMode
    operator: str
    minutes: int
    base_price_cents: int
    departures_local: tuple[int, ...]
    """Local departure hours. Frequency is a real difference between a Shinkansen every
    twenty minutes and a ferry that sails twice a day — and a twice-daily ferry is what turns
    a sloppy plan into a lost day."""
    on_time: float
    step_free: bool
    weather_sensitive: bool = False
    luggage_allowance: int = 2
    requires_driver: bool = False
    booking_lead_days: int = 0


class ActivitySpec(NamedTuple):
    city: CityCode
    district: str
    name: str
    category: ActivityCategory
    duration: int
    opens: int
    closes: int
    closed_weekday: int
    """ISO weekday, 0 for never closed."""
    intensity: int
    accessible: bool = True
    low_stimulus: bool = True
    min_age: int = 0
    max_age: int = 120
    weather_dependent: bool = False
    lead_time_days: int = 0
    season_months: tuple[int, ...] = tuple(range(1, 13))
    dietary: tuple[DietaryNeed, ...] = ()
    max_group: int = 99


def district_key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "")


# ---- Cities ---------------------------------------------------------------------------

CITY_SPECS: tuple[CitySpec, ...] = (
    # Japan — dense rail, high prices, two sharp seasonal peaks.
    CitySpec(CityCode.TOKYO, "Tokyo", Region.JAPAN, "JP", "Asia/Tokyo", 35.6762, 139.6503, ("HND", "NRT"), 16_500),
    CitySpec(CityCode.KYOTO, "Kyoto", Region.JAPAN, "JP", "Asia/Tokyo", 35.0116, 135.7681, (), 17_200),
    CitySpec(CityCode.OSAKA, "Osaka", Region.JAPAN, "JP", "Asia/Tokyo", 34.6937, 135.5023, ("KIX", "ITM"), 14_000),
    CitySpec(CityCode.HAKONE, "Hakone", Region.JAPAN, "JP", "Asia/Tokyo", 35.2324, 139.1069, (), 24_000),
    CitySpec(CityCode.KANAZAWA, "Kanazawa", Region.JAPAN, "JP", "Asia/Tokyo", 36.5613, 136.6562, ("KMQ",), 12_800),
    # Italy — one strong rail spine, thin away from it, August is a trap.
    CitySpec(CityCode.ROME, "Rome", Region.ITALY, "IT", "Europe/Rome", 41.9028, 12.4964, ("FCO", "CIA"), 15_500),
    CitySpec(CityCode.FLORENCE, "Florence", Region.ITALY, "IT", "Europe/Rome", 43.7696, 11.2558, ("FLR",), 16_800),
    CitySpec(CityCode.VENICE, "Venice", Region.ITALY, "IT", "Europe/Rome", 45.4408, 12.3155, ("VCE",), 21_000),
    CitySpec(CityCode.NAPLES, "Naples", Region.ITALY, "IT", "Europe/Rome", 40.8518, 14.2681, ("NAP",), 10_500),
    # Thailand — cheap, hot, and the only region where the sea is part of the network.
    CitySpec(CityCode.BANGKOK, "Bangkok", Region.THAILAND, "TH", "Asia/Bangkok", 13.7563, 100.5018, ("BKK", "DMK"), 5_400),
    CitySpec(CityCode.CHIANG_MAI, "Chiang Mai", Region.THAILAND, "TH", "Asia/Bangkok", 18.7883, 98.9853, ("CNX",), 4_100),
    CitySpec(CityCode.PHUKET, "Phuket", Region.THAILAND, "TH", "Asia/Bangkok", 7.8804, 98.3923, ("HKT",), 7_800),
    CitySpec(CityCode.KRABI, "Krabi", Region.THAILAND, "TH", "Asia/Bangkok", 8.0863, 98.9063, ("KBV",), 6_200),
)

REGION_CITIES: dict[Region, tuple[CityCode, ...]] = {
    region: tuple(c.code for c in CITY_SPECS if c.region is region) for region in Region
}

# The long-haul origin. Not a destination; it exists so the trip has a real starting point on
# the clock, in a zone eight to sixteen hours away from everything else.
ORIGIN = AirportSpec(
    "SFO", "San Francisco International", CityCode.HOME, 37.6213, -122.3790, 30,
    "US", "America/Los_Angeles", 75, True,
)


AIRPORT_SPECS: tuple[AirportSpec, ...] = (
    AirportSpec("HND", "Tokyo Haneda", CityCode.TOKYO, 35.5494, 139.7798, 35, "JP", "Asia/Tokyo", 60, True),
    AirportSpec("NRT", "Tokyo Narita", CityCode.TOKYO, 35.7720, 140.3929, 95, "JP", "Asia/Tokyo", 75, True),
    AirportSpec("KIX", "Kansai International", CityCode.OSAKA, 34.4342, 135.2328, 65, "JP", "Asia/Tokyo", 60, True),
    AirportSpec("ITM", "Osaka Itami", CityCode.OSAKA, 34.7855, 135.4382, 30, "JP", "Asia/Tokyo", 40, False),
    AirportSpec("KMQ", "Komatsu", CityCode.KANAZAWA, 36.3946, 136.4075, 45, "JP", "Asia/Tokyo", 40, False),
    AirportSpec("FCO", "Rome Fiumicino", CityCode.ROME, 41.8003, 12.2389, 45, "IT", "Europe/Rome", 60, True),
    AirportSpec("CIA", "Rome Ciampino", CityCode.ROME, 41.7994, 12.5949, 40, "IT", "Europe/Rome", 45, False),
    AirportSpec("FLR", "Florence Peretola", CityCode.FLORENCE, 43.8100, 11.2051, 20, "IT", "Europe/Rome", 40, False),
    AirportSpec("VCE", "Venice Marco Polo", CityCode.VENICE, 45.5053, 12.3519, 45, "IT", "Europe/Rome", 50, True),
    AirportSpec("NAP", "Naples Capodichino", CityCode.NAPLES, 40.8860, 14.2908, 25, "IT", "Europe/Rome", 45, False),
    AirportSpec("BKK", "Suvarnabhumi", CityCode.BANGKOK, 13.6900, 100.7501, 50, "TH", "Asia/Bangkok", 90, True),
    AirportSpec("DMK", "Don Mueang", CityCode.BANGKOK, 13.9126, 100.6068, 45, "TH", "Asia/Bangkok", 60, False),
    AirportSpec("CNX", "Chiang Mai", CityCode.CHIANG_MAI, 18.7669, 98.9626, 15, "TH", "Asia/Bangkok", 45, False),
    AirportSpec("HKT", "Phuket", CityCode.PHUKET, 8.1132, 98.3169, 45, "TH", "Asia/Bangkok", 50, True),
    AirportSpec("KBV", "Krabi", CityCode.KRABI, 8.0992, 98.9862, 25, "TH", "Asia/Bangkok", 40, False),
    ORIGIN,
)


# ---- Districts ------------------------------------------------------------------------
# walkability, nightlife, dining, cultural, quiet are 0-9 ordinals; price_index multiplies
# lodging; step_free is the share of stock a wheelchair user can actually use. Historic hill
# districts are low on that last column and that is the point — accessibility is a
# *geographic* constraint, so it interacts with location preference rather than sitting
# beside it. A client who wants Gion and needs step-free access has a genuinely hard trip.

DISTRICT_SPECS: tuple[DistrictSpec, ...] = (
    # Tokyo
    DistrictSpec(CityCode.TOKYO, "Shinjuku", 35.6938, 139.7034, 9, 9, 9, 4, 1, 1.35, 0.80),
    DistrictSpec(CityCode.TOKYO, "Shibuya", 35.6595, 139.7005, 9, 9, 8, 3, 1, 1.45, 0.78),
    DistrictSpec(CityCode.TOKYO, "Ginza", 35.6717, 139.7650, 8, 5, 8, 6, 4, 1.70, 0.88),
    DistrictSpec(CityCode.TOKYO, "Asakusa", 35.7148, 139.7967, 7, 4, 6, 9, 6, 0.95, 0.55),
    DistrictSpec(CityCode.TOKYO, "Marunouchi", 35.6812, 139.7671, 8, 3, 6, 5, 6, 1.60, 0.92),
    DistrictSpec(CityCode.TOKYO, "Kichijoji", 35.7030, 139.5800, 8, 6, 7, 4, 7, 0.82, 0.62),
    # Kyoto
    DistrictSpec(CityCode.KYOTO, "Gion", 35.0037, 135.7788, 8, 5, 7, 9, 5, 1.55, 0.30),
    DistrictSpec(CityCode.KYOTO, "Higashiyama", 34.9948, 135.7850, 7, 3, 5, 9, 7, 1.30, 0.22),
    DistrictSpec(CityCode.KYOTO, "Kyoto Station", 34.9858, 135.7588, 9, 4, 6, 4, 3, 1.10, 0.90),
    DistrictSpec(CityCode.KYOTO, "Arashiyama", 35.0094, 135.6668, 5, 2, 4, 9, 9, 1.05, 0.35),
    # Osaka
    DistrictSpec(CityCode.OSAKA, "Namba", 34.6659, 135.5010, 9, 9, 9, 4, 1, 1.20, 0.72),
    DistrictSpec(CityCode.OSAKA, "Umeda", 34.7025, 135.4959, 9, 7, 8, 3, 2, 1.25, 0.88),
    DistrictSpec(CityCode.OSAKA, "Tennoji", 34.6465, 135.5133, 7, 5, 7, 6, 4, 0.85, 0.70),
    # Hakone — a mountain resort. Almost nothing here is step-free and that is honest.
    DistrictSpec(CityCode.HAKONE, "Gora", 35.2456, 139.0511, 4, 2, 4, 6, 9, 1.30, 0.25),
    DistrictSpec(CityCode.HAKONE, "Hakone-Yumoto", 35.2325, 139.1057, 6, 2, 5, 5, 8, 1.10, 0.45),
    # Kanazawa
    DistrictSpec(CityCode.KANAZAWA, "Kenrokuen", 36.5620, 136.6626, 6, 2, 5, 9, 8, 1.15, 0.48),
    DistrictSpec(CityCode.KANAZAWA, "Katamachi", 36.5610, 136.6500, 8, 7, 8, 4, 4, 0.95, 0.68),
    # Rome
    DistrictSpec(CityCode.ROME, "Centro Storico", 41.8986, 12.4768, 9, 6, 8, 9, 3, 1.55, 0.42),
    DistrictSpec(CityCode.ROME, "Trastevere", 41.8890, 12.4695, 8, 8, 9, 7, 3, 1.30, 0.38),
    DistrictSpec(CityCode.ROME, "Prati", 41.9109, 12.4585, 8, 4, 7, 6, 6, 1.20, 0.80),
    DistrictSpec(CityCode.ROME, "Termini", 41.9010, 12.5010, 9, 5, 5, 3, 2, 0.82, 0.85),
    # Florence
    DistrictSpec(CityCode.FLORENCE, "Duomo", 43.7731, 11.2560, 9, 5, 8, 9, 2, 1.60, 0.40),
    DistrictSpec(CityCode.FLORENCE, "Oltrarno", 43.7663, 11.2482, 8, 7, 8, 8, 5, 1.25, 0.33),
    DistrictSpec(CityCode.FLORENCE, "Santa Croce", 43.7686, 11.2620, 8, 6, 7, 8, 4, 1.20, 0.44),
    # Venice — canals and bridges. The least step-free place in the world model, by fact.
    DistrictSpec(CityCode.VENICE, "San Marco", 45.4340, 12.3388, 7, 4, 7, 9, 2, 1.85, 0.18),
    DistrictSpec(CityCode.VENICE, "Cannaregio", 45.4438, 12.3268, 7, 5, 7, 7, 6, 1.35, 0.20),
    DistrictSpec(CityCode.VENICE, "Dorsoduro", 45.4319, 12.3268, 7, 6, 7, 8, 6, 1.45, 0.16),
    # Naples
    DistrictSpec(CityCode.NAPLES, "Centro Antico", 40.8500, 14.2570, 8, 7, 9, 9, 1, 1.00, 0.30),
    DistrictSpec(CityCode.NAPLES, "Chiaia", 40.8330, 14.2350, 8, 6, 8, 5, 5, 1.25, 0.66),
    DistrictSpec(CityCode.NAPLES, "Vomero", 40.8450, 14.2280, 7, 4, 6, 4, 7, 0.90, 0.58),
    # Bangkok
    DistrictSpec(CityCode.BANGKOK, "Sukhumvit", 13.7379, 100.5600, 8, 9, 9, 2, 2, 1.25, 0.82),
    DistrictSpec(CityCode.BANGKOK, "Riverside", 13.7230, 100.5140, 6, 5, 8, 7, 6, 1.45, 0.74),
    DistrictSpec(CityCode.BANGKOK, "Rattanakosin", 13.7524, 100.4930, 7, 3, 7, 9, 5, 1.00, 0.40),
    DistrictSpec(CityCode.BANGKOK, "Silom", 13.7280, 100.5340, 8, 8, 8, 3, 3, 1.15, 0.78),
    # Chiang Mai
    DistrictSpec(CityCode.CHIANG_MAI, "Old City", 18.7880, 98.9860, 8, 5, 8, 9, 5, 1.05, 0.44),
    DistrictSpec(CityCode.CHIANG_MAI, "Nimman", 18.7970, 98.9670, 8, 8, 9, 3, 5, 1.20, 0.72),
    DistrictSpec(CityCode.CHIANG_MAI, "Riverside CM", 18.7860, 99.0020, 6, 4, 7, 5, 7, 1.00, 0.60),
    # Phuket
    DistrictSpec(CityCode.PHUKET, "Patong", 7.8960, 98.2960, 7, 9, 7, 1, 1, 1.10, 0.68),
    DistrictSpec(CityCode.PHUKET, "Kata", 7.8200, 98.2980, 5, 4, 6, 2, 7, 1.20, 0.55),
    DistrictSpec(CityCode.PHUKET, "Old Town", 7.8840, 98.3880, 8, 5, 8, 8, 5, 0.95, 0.50),
    # Krabi
    DistrictSpec(CityCode.KRABI, "Ao Nang", 8.0320, 98.8210, 6, 6, 7, 2, 5, 1.05, 0.52),
    DistrictSpec(CityCode.KRABI, "Railay", 8.0110, 98.8380, 3, 3, 5, 3, 9, 1.35, 0.05),
    DistrictSpec(CityCode.KRABI, "Krabi Town", 8.0863, 98.9063, 7, 4, 7, 5, 6, 0.80, 0.62),
)

DISTRICTS_BY_CITY: dict[CityCode, tuple[DistrictSpec, ...]] = {
    city.code: tuple(d for d in DISTRICT_SPECS if d.city is city.code) for city in CITY_SPECS
}


# ---- Surface corridors ------------------------------------------------------------------
# Every entry is bidirectional; the generator emits both directions. Frequency is the
# variable that matters most: Tokyo-Kyoto runs all day and forgives a sloppy plan, while the
# Krabi ferry sails twice and does not.

CORRIDOR_SPECS: tuple[CorridorSpec, ...] = (
    # --- Japan: the rail network is the reason to come here ---
    CorridorSpec(CityCode.TOKYO, CityCode.KYOTO, TransportMode.RAIL, "JR Central", 140, 1_380,
                 (6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20), 0.995, True),
    CorridorSpec(CityCode.KYOTO, CityCode.OSAKA, TransportMode.RAIL, "JR West", 15, 420,
                 (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 0.994, True),
    CorridorSpec(CityCode.TOKYO, CityCode.OSAKA, TransportMode.RAIL, "JR Central", 150, 1_480,
                 (6, 8, 10, 12, 14, 16, 18, 20), 0.995, True),
    CorridorSpec(CityCode.TOKYO, CityCode.HAKONE, TransportMode.RAIL, "Odakyu", 95, 680,
                 (7, 9, 11, 13, 15, 17, 19), 0.985, False),
    CorridorSpec(CityCode.TOKYO, CityCode.KANAZAWA, TransportMode.RAIL, "JR East", 165, 1_450,
                 (6, 8, 10, 12, 14, 16, 18), 0.993, True),
    CorridorSpec(CityCode.KYOTO, CityCode.KANAZAWA, TransportMode.RAIL, "JR West", 135, 780,
                 (7, 9, 12, 15, 18), 0.990, True),
    CorridorSpec(CityCode.OSAKA, CityCode.HAKONE, TransportMode.RAIL, "JR Central", 210, 1_620,
                 (7, 10, 14, 17), 0.990, False),
    CorridorSpec(CityCode.TOKYO, CityCode.KYOTO, TransportMode.BUS, "Willer Express", 480, 420,
                 (8, 22, 23), 0.94, False, luggage_allowance=1),
    CorridorSpec(CityCode.TOKYO, CityCode.OSAKA, TransportMode.BUS, "JR Bus", 520, 460,
                 (9, 21, 22), 0.93, False, luggage_allowance=1),
    # --- Italy: one fast spine, and everything off it is slower than it looks ---
    CorridorSpec(CityCode.ROME, CityCode.FLORENCE, TransportMode.RAIL, "Trenitalia", 95, 3_400,
                 (6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20), 0.93, True),
    CorridorSpec(CityCode.FLORENCE, CityCode.VENICE, TransportMode.RAIL, "Italo", 125, 3_900,
                 (7, 9, 11, 13, 15, 17, 19), 0.92, True),
    CorridorSpec(CityCode.ROME, CityCode.VENICE, TransportMode.RAIL, "Trenitalia", 225, 5_600,
                 (7, 9, 12, 15, 18), 0.91, True),
    CorridorSpec(CityCode.ROME, CityCode.NAPLES, TransportMode.RAIL, "Italo", 70, 2_600,
                 (6, 7, 8, 9, 11, 13, 15, 17, 19, 21), 0.94, True),
    CorridorSpec(CityCode.NAPLES, CityCode.FLORENCE, TransportMode.RAIL, "Trenitalia", 175, 4_800,
                 (7, 10, 13, 16, 19), 0.90, True),
    CorridorSpec(CityCode.ROME, CityCode.FLORENCE, TransportMode.BUS, "FlixBus", 210, 1_100,
                 (7, 13, 23), 0.86, False, luggage_allowance=1),
    CorridorSpec(CityCode.ROME, CityCode.NAPLES, TransportMode.CAR_HIRE, "Europcar", 165, 7_200,
                 (8, 10, 12, 14), 0.97, True, requires_driver=True, luggage_allowance=4),
    CorridorSpec(CityCode.FLORENCE, CityCode.VENICE, TransportMode.CAR_HIRE, "Hertz", 200, 8_400,
                 (8, 10, 13), 0.97, True, requires_driver=True, luggage_allowance=4),
    # --- Thailand: the sea is part of the network, and the sea has weather ---
    CorridorSpec(CityCode.BANGKOK, CityCode.CHIANG_MAI, TransportMode.RAIL, "SRT", 760, 1_450,
                 (8, 18, 19, 22), 0.72, False, luggage_allowance=3),
    CorridorSpec(CityCode.BANGKOK, CityCode.CHIANG_MAI, TransportMode.BUS, "Nakhonchai Air", 590, 880,
                 (8, 20, 21), 0.80, False, luggage_allowance=2),
    CorridorSpec(CityCode.BANGKOK, CityCode.PHUKET, TransportMode.BUS, "Phuket Central", 720, 1_050,
                 (7, 17, 18), 0.76, False, luggage_allowance=2),
    CorridorSpec(CityCode.PHUKET, CityCode.KRABI, TransportMode.FERRY, "Ao Nang Princess", 135, 750,
                 (8, 13), 0.82, False, weather_sensitive=True, luggage_allowance=2),
    CorridorSpec(CityCode.PHUKET, CityCode.KRABI, TransportMode.BUS, "Krabi Transfer", 210, 480,
                 (7, 9, 11, 14, 16), 0.85, False, luggage_allowance=2),
    CorridorSpec(CityCode.CHIANG_MAI, CityCode.PHUKET, TransportMode.BUS, "Southern Line", 1_320, 1_600,
                 (6, 15), 0.68, False, luggage_allowance=2),
    CorridorSpec(CityCode.BANGKOK, CityCode.KRABI, TransportMode.BUS, "Lignite Tour", 690, 980,
                 (8, 19), 0.77, False, luggage_allowance=2),
)


# ---- Visa policy -------------------------------------------------------------------------
# (nationality, country) -> (visa_required, consular lead days, fee in cents). Lead time is
# the gate that bites: a trip departing in three weeks is simply not available to a passport
# that needs thirty days of processing, and no amount of budget fixes it.

VISA_POLICY: dict[tuple[Nationality, str], tuple[bool, int, int]] = {
    (Nationality.US, "JP"): (False, 0, 0),
    (Nationality.US, "IT"): (False, 0, 0),
    (Nationality.US, "TH"): (False, 0, 0),
    (Nationality.GB, "JP"): (False, 0, 0),
    (Nationality.GB, "IT"): (False, 0, 0),
    (Nationality.GB, "TH"): (False, 0, 0),
    (Nationality.DE, "JP"): (False, 0, 0),
    (Nationality.DE, "IT"): (False, 0, 0),
    (Nationality.DE, "TH"): (False, 0, 0),
    (Nationality.BR, "JP"): (False, 0, 0),
    (Nationality.BR, "IT"): (False, 0, 0),
    (Nationality.BR, "TH"): (False, 0, 0),
    (Nationality.IN, "JP"): (True, 12, 3_100),
    (Nationality.IN, "IT"): (True, 21, 9_800),
    (Nationality.IN, "TH"): (True, 3, 3_500),
    (Nationality.CN, "JP"): (True, 9, 3_100),
    (Nationality.CN, "IT"): (True, 18, 9_800),
    (Nationality.CN, "TH"): (False, 0, 0),
    (Nationality.PH, "JP"): (True, 10, 3_100),
    (Nationality.PH, "IT"): (True, 20, 9_800),
    (Nationality.PH, "TH"): (False, 0, 0),
    (Nationality.NG, "JP"): (True, 24, 3_100),
    (Nationality.NG, "IT"): (True, 32, 9_800),
    (Nationality.NG, "TH"): (True, 15, 3_500),
}

# ---- Activity anchors --------------------------------------------------------------------
# Hand-authored real places, which the generator surrounds with a procedural long tail. The
# anchors carry the facts a generator cannot invent honestly: that Fushimi Inari is a hill and
# therefore not step-free, that a bar crawl has an age floor, that a kaiseki table books weeks
# out, that the Blue Grotto only runs when the sea is calm.
#
# `accessible` and `low_stimulus` are where accessibility stops being a checkbox and becomes
# geography: the most culturally dense experiences in Kyoto, Venice and Naples are the least
# accessible ones, so a client who wants culture *and* needs step-free access is a genuinely
# constrained problem rather than a filtered list.

_FOOD, _HIST = ActivityCategory.FOOD, ActivityCategory.HISTORY
_MUSE, _NAT = ActivityCategory.MUSEUM, ActivityCategory.NATURE
_NIGHT, _SHOP, _FAM = ActivityCategory.NIGHTLIFE, ActivityCategory.SHOPPING, ActivityCategory.FAMILY
_VEG, _VGN, _HAL, _GF = (
    DietaryNeed.VEGETARIAN, DietaryNeed.VEGAN, DietaryNeed.HALAL, DietaryNeed.GLUTEN_FREE
)

ACTIVITY_SPECS: tuple[ActivitySpec, ...] = (
    # --- Tokyo ---
    ActivitySpec(CityCode.TOKYO, "ginza", "Tsukiji Outer Market Food Walk", _FOOD, 180, 6, 14, 0, 4,
                 low_stimulus=False, dietary=(_VEG,), max_group=12),
    ActivitySpec(CityCode.TOKYO, "shinjuku", "Omoide Yokocho Izakaya Crawl", _NIGHT, 210, 18, 23, 0, 3,
                 accessible=False, low_stimulus=False, min_age=20, max_group=8),
    ActivitySpec(CityCode.TOKYO, "asakusa", "Senso-ji Temple and Nakamise", _HIST, 120, 6, 17, 0, 3,
                 low_stimulus=False),
    ActivitySpec(CityCode.TOKYO, "asakusa", "Tokyo National Museum", _MUSE, 150, 9, 17, 1, 2),
    ActivitySpec(CityCode.TOKYO, "shibuya", "Ramen Making Workshop", _FOOD, 150, 10, 16, 3, 2,
                 dietary=(_VEG, _VGN), max_group=10, lead_time_days=4),
    ActivitySpec(CityCode.TOKYO, "kichijoji", "Inokashira Park and Ghibli Quarter", _NAT, 180, 8, 18, 2, 4,
                 weather_dependent=True),
    ActivitySpec(CityCode.TOKYO, "ginza", "teamLab Digital Art", _MUSE, 180, 10, 19, 2, 3,
                 low_stimulus=False, lead_time_days=10),
    ActivitySpec(CityCode.TOKYO, "marunouchi", "Imperial Palace East Gardens", _HIST, 120, 9, 16, 1, 4,
                 weather_dependent=True),
    # --- Kyoto ---
    ActivitySpec(CityCode.KYOTO, "higashiyama", "Fushimi Inari Night Hike", _NAT, 180, 16, 22, 0, 7,
                 accessible=False, weather_dependent=True, min_age=8),
    ActivitySpec(CityCode.KYOTO, "gion", "Gion Geisha District Walk", _HIST, 120, 15, 21, 0, 3,
                 accessible=False),
    ActivitySpec(CityCode.KYOTO, "gion", "Kaiseki Dinner", _FOOD, 150, 18, 21, 1, 1,
                 accessible=False, lead_time_days=21, max_group=6, dietary=(_VEG,)),
    ActivitySpec(CityCode.KYOTO, "arashiyama", "Arashiyama Bamboo Grove and Monkey Park", _NAT, 150, 7, 17, 0, 5,
                 accessible=False, low_stimulus=False, weather_dependent=True),
    ActivitySpec(CityCode.KYOTO, "higashiyama", "Tea Ceremony", _HIST, 90, 10, 16, 2, 1,
                 max_group=8, lead_time_days=5),
    ActivitySpec(CityCode.KYOTO, "kyoto_station", "Nishiki Market Tasting", _FOOD, 120, 9, 17, 2, 3,
                 low_stimulus=False, dietary=(_VEG,)),
    # --- Osaka ---
    ActivitySpec(CityCode.OSAKA, "namba", "Dotonbori Street Food", _FOOD, 150, 11, 23, 0, 3,
                 low_stimulus=False, dietary=(_VEG, _HAL)),
    ActivitySpec(CityCode.OSAKA, "tennoji", "Osaka Castle and Nishinomaru", _HIST, 150, 9, 17, 0, 4,
                 weather_dependent=True),
    ActivitySpec(CityCode.OSAKA, "umeda", "Umeda Sky Building", _MUSE, 90, 10, 22, 0, 2),
    ActivitySpec(CityCode.OSAKA, "namba", "Kuromon Market and Kushikatsu", _FOOD, 120, 9, 18, 3, 2,
                 low_stimulus=False),
    ActivitySpec(CityCode.OSAKA, "tennoji", "Osaka Aquarium Kaiyukan", _FAM, 180, 10, 19, 0, 2),
    # --- Hakone ---
    ActivitySpec(CityCode.HAKONE, "gora", "Hakone Open-Air Museum", _MUSE, 150, 9, 17, 0, 3,
                 weather_dependent=True),
    ActivitySpec(CityCode.HAKONE, "gora", "Owakudani Ropeway and Sulphur Valley", _NAT, 180, 9, 16, 0, 5,
                 accessible=False, weather_dependent=True, min_age=6),
    ActivitySpec(CityCode.HAKONE, "hakone_yumoto", "Private Onsen Bathing", _NAT, 120, 11, 21, 0, 1,
                 accessible=False, min_age=12, max_group=4),
    ActivitySpec(CityCode.HAKONE, "hakone_yumoto", "Lake Ashi Pirate Cruise", _FAM, 90, 9, 16, 0, 2,
                 weather_dependent=True, season_months=(3, 4, 5, 6, 7, 8, 9, 10, 11)),
    # --- Kanazawa ---
    ActivitySpec(CityCode.KANAZAWA, "kenrokuen", "Kenroku-en Garden", _NAT, 120, 7, 18, 0, 3,
                 weather_dependent=True),
    ActivitySpec(CityCode.KANAZAWA, "kenrokuen", "21st Century Museum of Contemporary Art", _MUSE, 120, 10, 18, 1, 2),
    ActivitySpec(CityCode.KANAZAWA, "katamachi", "Omicho Market Seafood Lunch", _FOOD, 90, 9, 15, 3, 2,
                 low_stimulus=False),
    ActivitySpec(CityCode.KANAZAWA, "kenrokuen", "Gold Leaf Workshop", _SHOP, 90, 10, 16, 2, 1,
                 max_group=8, lead_time_days=3),
    # --- Rome ---
    ActivitySpec(CityCode.ROME, "centro_storico", "Colosseum and Roman Forum", _HIST, 210, 9, 18, 0, 5,
                 accessible=False, low_stimulus=False, lead_time_days=14, weather_dependent=True),
    ActivitySpec(CityCode.ROME, "prati", "Vatican Museums and Sistine Chapel", _MUSE, 240, 8, 18, 7, 4,
                 low_stimulus=False, lead_time_days=21),
    ActivitySpec(CityCode.ROME, "trastevere", "Trastevere Food Tour", _FOOD, 180, 18, 22, 1, 3,
                 accessible=False, dietary=(_VEG, _GF), max_group=10),
    ActivitySpec(CityCode.ROME, "centro_storico", "Pantheon and Piazza Navona Walk", _HIST, 120, 9, 19, 0, 3,
                 weather_dependent=True),
    ActivitySpec(CityCode.ROME, "prati", "Pasta Making Class", _FOOD, 180, 10, 20, 0, 1,
                 dietary=(_VEG, _VGN, _GF), max_group=12, lead_time_days=5),
    ActivitySpec(CityCode.ROME, "termini", "Borghese Gallery", _MUSE, 120, 9, 17, 1, 2, lead_time_days=14),
    # --- Florence ---
    ActivitySpec(CityCode.FLORENCE, "duomo", "Uffizi Gallery", _MUSE, 180, 8, 18, 1, 3,
                 low_stimulus=False, lead_time_days=10),
    ActivitySpec(CityCode.FLORENCE, "duomo", "Duomo Dome Climb", _HIST, 90, 8, 19, 7, 8,
                 accessible=False, min_age=8, lead_time_days=7),
    ActivitySpec(CityCode.FLORENCE, "oltrarno", "Oltrarno Artisan Workshops", _SHOP, 150, 10, 18, 7, 2,
                 accessible=False),
    ActivitySpec(CityCode.FLORENCE, "santa_croce", "Chianti Vineyard Afternoon", _FOOD, 300, 12, 19, 0, 2,
                 min_age=18, weather_dependent=True, max_group=14, dietary=(_VEG,)),
    ActivitySpec(CityCode.FLORENCE, "santa_croce", "Leather School Demonstration", _SHOP, 90, 10, 17, 7, 1),
    # --- Venice ---
    ActivitySpec(CityCode.VENICE, "san_marco", "St Mark's Basilica and Campanile", _HIST, 120, 9, 17, 0, 4,
                 accessible=False, low_stimulus=False, lead_time_days=7),
    ActivitySpec(CityCode.VENICE, "cannaregio", "Cicchetti and Wine Crawl", _FOOD, 150, 17, 22, 1, 3,
                 accessible=False, min_age=18, dietary=(_VEG,), max_group=8),
    ActivitySpec(CityCode.VENICE, "dorsoduro", "Peggy Guggenheim Collection", _MUSE, 120, 10, 18, 2, 2),
    ActivitySpec(CityCode.VENICE, "cannaregio", "Murano and Burano by Boat", _NAT, 300, 9, 17, 0, 3,
                 accessible=False, weather_dependent=True, season_months=(3, 4, 5, 6, 7, 8, 9, 10)),
    ActivitySpec(CityCode.VENICE, "san_marco", "Gondola Workshop Visit", _HIST, 90, 10, 16, 7, 2, accessible=False),
    # --- Naples ---
    ActivitySpec(CityCode.NAPLES, "centro_antico", "Pompeii Day Trip", _HIST, 360, 9, 17, 0, 6,
                 accessible=False, weather_dependent=True, min_age=6),
    ActivitySpec(CityCode.NAPLES, "centro_antico", "Naples Pizza Crawl", _FOOD, 150, 12, 22, 0, 3,
                 low_stimulus=False, dietary=(_VEG,), max_group=10),
    ActivitySpec(CityCode.NAPLES, "centro_antico", "Napoli Sotterranea Underground", _HIST, 120, 10, 18, 0, 6,
                 accessible=False, low_stimulus=False, min_age=10),
    ActivitySpec(CityCode.NAPLES, "vomero", "Capodimonte Museum", _MUSE, 150, 9, 17, 3, 2),
    ActivitySpec(CityCode.NAPLES, "chiaia", "Capri and the Blue Grotto", _NAT, 420, 8, 18, 0, 4,
                 accessible=False, weather_dependent=True, season_months=(4, 5, 6, 7, 8, 9, 10)),
    # --- Bangkok ---
    ActivitySpec(CityCode.BANGKOK, "rattanakosin", "Grand Palace and Wat Pho", _HIST, 180, 8, 15, 0, 4,
                 low_stimulus=False, weather_dependent=True),
    ActivitySpec(CityCode.BANGKOK, "riverside", "Chao Phraya Dinner Cruise", _FOOD, 150, 18, 22, 0, 1,
                 weather_dependent=True, dietary=(_VEG, _HAL), lead_time_days=3),
    ActivitySpec(CityCode.BANGKOK, "sukhumvit", "Street Food Tuk-Tuk Night Tour", _FOOD, 210, 18, 23, 0, 3,
                 accessible=False, low_stimulus=False, min_age=12, dietary=(_HAL,), max_group=8),
    ActivitySpec(CityCode.BANGKOK, "rattanakosin", "Thai Cooking Class and Market Visit", _FOOD, 240, 9, 15, 0, 2,
                 dietary=(_VEG, _VGN, _HAL, _GF), max_group=12, lead_time_days=4),
    ActivitySpec(CityCode.BANGKOK, "silom", "Rooftop Bar Evening", _NIGHT, 120, 18, 24, 0, 1,
                 min_age=20, low_stimulus=False),
    ActivitySpec(CityCode.BANGKOK, "riverside", "Jim Thompson House", _MUSE, 90, 9, 18, 0, 2),
    # --- Chiang Mai ---
    ActivitySpec(CityCode.CHIANG_MAI, "old_city", "Doi Suthep Temple at Sunrise", _HIST, 240, 5, 11, 0, 5,
                 accessible=False, weather_dependent=True),
    ActivitySpec(CityCode.CHIANG_MAI, "riverside_cm", "Ethical Elephant Sanctuary", _NAT, 420, 8, 16, 0, 4,
                 accessible=False, weather_dependent=True, min_age=5, max_group=10, lead_time_days=7),
    ActivitySpec(CityCode.CHIANG_MAI, "nimman", "Northern Thai Cooking School", _FOOD, 240, 9, 16, 0, 2,
                 dietary=(_VEG, _VGN, _GF), max_group=10, lead_time_days=3),
    ActivitySpec(CityCode.CHIANG_MAI, "old_city", "Sunday Walking Street Market", _SHOP, 180, 16, 22, 0, 3,
                 low_stimulus=False, season_months=tuple(range(1, 13))),
    ActivitySpec(CityCode.CHIANG_MAI, "riverside_cm", "Bamboo Rafting on the Mae Taeng", _NAT, 300, 8, 15, 0, 6,
                 accessible=False, weather_dependent=True, min_age=10,
                 season_months=(11, 12, 1, 2, 3, 4, 5)),
    # --- Phuket ---
    ActivitySpec(CityCode.PHUKET, "kata", "Phi Phi Islands Speedboat Day", _NAT, 480, 7, 17, 0, 4,
                 accessible=False, weather_dependent=True, min_age=4, max_group=20,
                 season_months=(11, 12, 1, 2, 3, 4)),
    ActivitySpec(CityCode.PHUKET, "old_town", "Old Town Heritage and Peranakan Food", _FOOD, 180, 10, 17, 1, 2,
                 dietary=(_VEG, _HAL)),
    ActivitySpec(CityCode.PHUKET, "patong", "Muay Thai Fight Night", _NIGHT, 150, 20, 23, 0, 1,
                 low_stimulus=False, min_age=16),
    ActivitySpec(CityCode.PHUKET, "kata", "Beginner Surf Lesson", _NAT, 120, 8, 16, 0, 7,
                 accessible=False, weather_dependent=True, min_age=8, max_group=6,
                 season_months=(5, 6, 7, 8, 9, 10)),
    ActivitySpec(CityCode.PHUKET, "old_town", "Big Buddha and Wat Chalong", _HIST, 180, 8, 17, 0, 3,
                 weather_dependent=True),
    # --- Krabi ---
    ActivitySpec(CityCode.KRABI, "railay", "Railay Rock Climbing", _NAT, 300, 8, 16, 0, 9,
                 accessible=False, weather_dependent=True, min_age=12, max_age=65, max_group=6,
                 season_months=(11, 12, 1, 2, 3, 4)),
    ActivitySpec(CityCode.KRABI, "ao_nang", "Four Islands Longtail Tour", _NAT, 420, 8, 17, 0, 4,
                 accessible=False, weather_dependent=True, min_age=4,
                 season_months=(11, 12, 1, 2, 3, 4, 5)),
    ActivitySpec(CityCode.KRABI, "krabi_town", "Krabi Night Market", _FOOD, 120, 17, 22, 0, 2,
                 low_stimulus=False, dietary=(_VEG, _HAL)),
    ActivitySpec(CityCode.KRABI, "ao_nang", "Emerald Pool and Hot Springs", _NAT, 300, 8, 16, 0, 5,
                 accessible=False, weather_dependent=True),
)

ACTIVITIES_BY_CITY: dict[CityCode, tuple[ActivitySpec, ...]] = {
    city.code: tuple(a for a in ACTIVITY_SPECS if a.city is city.code) for city in CITY_SPECS
}

# The procedural long tail. Anchors are the famous things; these are the hundreds of ordinary
# listings any real marketplace is mostly made of. Composed as
# "{theme} {format}" in a district, which is genuinely how such listings are named — and it
# means the catalogue scales with the world instead of being a fixed 18.
TAIL_THEMES: dict[ActivityCategory, tuple[str, ...]] = {
    _FOOD: ("Market", "Street Food", "Dessert", "Coffee", "Night Eats", "Home Kitchen", "Wine"),
    _HIST: ("Old Quarter", "Temple", "Backstreets", "Architecture", "Wartime", "Royal"),
    _MUSE: ("Design", "Photography", "Local History", "Modern Art", "Craft", "Science"),
    _NAT: ("Riverside", "Park", "Sunrise", "Garden", "Hillside", "Coastal"),
    _NIGHT: ("Live Music", "Jazz", "Rooftop", "Craft Beer", "Late Night"),
    _SHOP: ("Vintage", "Artisan", "Antiques", "Bookshop", "Ceramics"),
    _FAM: ("Kids", "Playground", "Hands-On", "Puppet", "Animal"),
}
TAIL_FORMATS: tuple[str, ...] = ("Walk", "Tour", "Workshop", "Experience", "Half Day", "Tasting", "Class")


NATIONALITY_WEIGHTS: dict[Nationality, float] = {
    Nationality.US: 0.34, Nationality.GB: 0.14, Nationality.DE: 0.11,
    Nationality.IN: 0.13, Nationality.CN: 0.10, Nationality.BR: 0.07,
    Nationality.PH: 0.06, Nationality.NG: 0.05,
}


# ---- Seasonality -------------------------------------------------------------------------
# Each region gets a price curve over the year and a weather-risk curve, both as sums of
# Gaussians over day-of-year. They are NOT the same curve: Thailand's cheapest months are its
# wettest, so the budget option carries the ferry risk. That coupling is the whole reason
# seasonality is in the model rather than a flat multiplier.

class SeasonSpec(NamedTuple):
    peaks: tuple[tuple[float, int, float], ...]
    """(amplitude, day-of-year centre, width) added to a base of 1.0."""
    weather_risk: tuple[tuple[float, int, float], ...]
    """Probability mass of a whole-day regional weather event, same Gaussian form."""
    price_level: float
    """Region-wide multiplier on everything. This is what makes Thailand a different trip
    rather than the same trip in a different place."""


SEASON_SPECS: dict[Region, SeasonSpec] = {
    Region.JAPAN: SeasonSpec(
        peaks=((0.42, 95, 18.0), (0.34, 318, 16.0), (-0.16, 40, 25.0)),
        weather_risk=((0.28, 250, 38.0),),  # typhoon season, late summer into autumn
        price_level=1.00,
    ),
    Region.ITALY: SeasonSpec(
        peaks=((0.38, 200, 34.0), (0.14, 115, 22.0), (-0.22, 25, 30.0)),
        weather_risk=((0.12, 300, 40.0), (0.10, 205, 25.0)),  # autumn rain; August heat
        price_level=1.05,
    ),
    Region.THAILAND: SeasonSpec(
        peaks=((0.34, 15, 30.0), (0.18, 350, 20.0), (-0.24, 240, 45.0)),
        weather_risk=((0.46, 255, 48.0),),  # southwest monsoon — this cancels ferries
        price_level=0.45,
    ),
}
