"""Data contracts for the travel-agent environment.

Every shape that crosses a module boundary is a frozen Pydantic model, so the world a
generator produces cannot be mutated by the environment that consumes it.

Two conventions are load-bearing and worth stating once:

* **Money is integer minor units (cents).** Floats accumulate rounding error across a
  book -> disrupt -> cancel -> refund -> rebook chain, and the ledger must reconcile to
  the cent or the run is a bug (see `ledger.Ledger.assert_reconciled`).
* **All datetimes are timezone-aware UTC.** Local wall-clock is derived at presentation
  only. A flight's `local_departure` is stored separately because travellers reason in
  local time and the reward's "reasonable hour" term depends on it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Cents: TypeAlias = Annotated[int, Field(ge=0)]
SignedCents: TypeAlias = int
Level: TypeAlias = Annotated[int, Field(ge=0, le=9)]
"""A 10-point ordinal scale. Used for every latent preference axis so that personas are
drawn from a 10^k grid rather than a handful of hand-written archetypes."""

Score01: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
Key: TypeAlias = Annotated[str, Field(min_length=1, max_length=64)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---- Traveller needs ---------------------------------------------------------------
# Declared before the supply-side models because lodging, transport and activities all
# have to advertise which of these they can satisfy.


class AccessibilityNeed(StrEnum):
    """Needs that are satisfied or not — never satisfied *a bit*.

    These are the reason the environment has hard gates at all. A hotel with stairs is
    not a slightly worse hotel for a wheelchair user; it is a booking that fails at
    check-in. Modelling them as utility would teach an agent to trade them away.
    """

    STEP_FREE = "step_free"
    WHEELCHAIR = "wheelchair"
    ELEVATOR = "elevator"
    SERVICE_ANIMAL = "service_animal"
    LOW_STIMULUS = "low_stimulus"
    """Sensory: no crowds, no loud venues. Real, and it constrains activities not rooms."""


class DietaryNeed(StrEnum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    GLUTEN_FREE = "gluten_free"
    NUT_ALLERGY = "nut_allergy"


class Nationality(StrEnum):
    """Passport country. Drives visa eligibility, which is a hard gate on whole regions."""

    US = "US"
    GB = "GB"
    DE = "DE"
    IN = "IN"
    CN = "CN"
    BR = "BR"
    NG = "NG"
    PH = "PH"


class Traveller(FrozenModel):
    """One person on the booking.

    The party is a manifest, not a headcount. A scalar `party_size` cannot express "one
    of the four is 7 years old and one uses a wheelchair", and those are exactly the
    facts that decide whether a plan is bookable.
    """

    key: Key
    age: Annotated[int, Field(ge=0, le=110)]
    nationality: Nationality
    passport_expiry: date
    accessibility: frozenset[AccessibilityNeed] = frozenset()
    dietary: frozenset[DietaryNeed] = frozenset()

    @property
    def is_minor(self) -> bool:
        return self.age < 18

    @property
    def needs_adjacent_seat(self) -> bool:
        """Under-12s must be seated with an accompanying adult. Airlines enforce this."""
        return self.age < 12


class FixedCommitment(FrozenModel):
    """An immovable point in the trip. Everything else routes around it."""

    key: Key
    label: str
    city: "CityCode"
    start_utc: datetime
    end_utc: datetime


# ---- Geography -------------------------------------------------------------------


class Region(StrEnum):
    """A destination region. Regions differ structurally, not cosmetically: rail quality,
    visa regime, seasonality, price level and the transport modes that exist at all."""

    JAPAN = "japan"
    ITALY = "italy"
    THAILAND = "thailand"


class CityCode(StrEnum):
    HOME = "SFO"
    """Where the client lives. Never a destination and never in a `Region`, but it is a
    place on the clock in its own zone, so it needs a code like anywhere else."""

    # Japan
    TOKYO = "TYO"
    KYOTO = "UKY"
    OSAKA = "OSA"
    HAKONE = "HAK"
    KANAZAWA = "KMQ"
    # Italy
    ROME = "ROM"
    FLORENCE = "FLR"
    VENICE = "VCE"
    NAPLES = "NAP"
    # Thailand
    BANGKOK = "BKK"
    CHIANG_MAI = "CNX"
    PHUKET = "HKT"
    KRABI = "KBV"


class Coordinates(FrozenModel):
    lat: Annotated[float, Field(ge=-90.0, le=90.0)]
    lon: Annotated[float, Field(ge=-180.0, le=180.0)]


class District(FrozenModel):
    """A named neighbourhood. Hotels sit in districts; activities sit in districts;
    the travel time between two districts is what makes a plan feasible or not."""

    key: Key
    name: str
    city: CityCode
    coords: Coordinates
    walkability: Level
    nightlife: Level
    dining_density: Level
    cultural_density: Level
    quiet: Level
    price_index: Annotated[float, Field(gt=0.0)]
    """Multiplier on baseline lodging price. Central//desirable districts run 1.2-1.6."""


class City(FrozenModel):
    code: CityCode
    name: str
    country: str
    timezone: str
    centre: Coordinates
    districts: tuple[District, ...]
    airports: tuple[Key, ...]
    """Airport keys serving this city; empty for rail-only destinations (Kyoto)."""

    region: Region = Region.JAPAN
    utc_offset_minutes: int = 540
    """Offset from UTC **for this world's trip window**, resolved from `timezone` with
    `zoneinfo` once at generation time and then frozen.

    Storing the resolved offset rather than re-deriving it per call is the same discipline
    as recasting one instant into several formats instead of drawing it twice: the offset
    is a property of (zone, date), and a world spans days, not months, so it is constant
    across the window. Recomputing it on a hot path would be slower and — worse — would let
    two call sites disagree about what "9am local" meant.
    """


class Airport(FrozenModel):
    key: Key
    name: str
    city: CityCode
    coords: Coordinates
    minutes_to_centre: Annotated[int, Field(ge=0)]

    country: str = "JP"
    utc_offset_minutes: int = 540
    """An airport is a *place on the clock*. The origin airport and the destination airport
    of one flight sit in different zones, which is precisely the fact `local_departure_hour`
    got wrong in v1 by applying the destination's offset to a departure."""
    min_connection_minutes: Annotated[int, Field(ge=0)] = 60
    """Published MCT. A self-transfer itinerary does not get to use it."""


# ---- Transport -------------------------------------------------------------------


class TransportMode(StrEnum):
    """How a leg is actually travelled. Mode is not flavour: it decides reliability,
    luggage rules, step-free access, booking lead time and what act II can do to it."""

    FLIGHT = "flight"
    RAIL = "rail"
    BUS = "bus"
    FERRY = "ferry"
    CAR_HIRE = "car_hire"


class CabinClass(StrEnum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"


class FareBrand(StrEnum):
    """Fare families, in the order airlines actually sell them. The brand — not the
    price — decides what happens when the trip breaks, which is the whole point of
    carrying it in the data model."""

    BASIC = "basic"
    STANDARD = "standard"
    FLEX = "flex"


class Flight(FrozenModel):
    key: Key
    carrier: str
    flight_number: str
    origin: Key
    destination: Key
    departure_utc: datetime
    arrival_utc: datetime
    local_departure_hour: Annotated[int, Field(ge=0, le=23)]
    duration_minutes: Annotated[int, Field(gt=0)]
    stops: Annotated[int, Field(ge=0, le=2)]
    connection_buffer_minutes: Annotated[int, Field(ge=0)] | None = None
    """Layover slack on a one-stop itinerary. Below ~60 min a delay misconnects."""
    cabin: CabinClass
    fare_brand: FareBrand
    price: Cents
    refundable: bool
    change_fee: Cents
    checked_bag_included: bool
    seats_remaining: Annotated[int, Field(ge=0)]
    on_time_rate: Score01
    """Historical reliability. Drives the disruption sampler, and is visible to the
    agent — so booking a cheap unreliable carrier is an informed bet, not a trap."""

    # ---- v2 fields. Defaults reproduce v1 behaviour exactly. ----
    alliance: str = ""
    is_self_transfer: bool = False
    """Two separate tickets sold as one journey. Cheaper, and catastrophically worse under
    disruption: miss the second leg and no carrier owes you anything. The best trap in the
    world model, because it is strictly dominated only once act II runs."""
    adjacent_seats_available: Annotated[int, Field(ge=0)] = 9
    """How many of the remaining seats sit together. A party of four with a 7-year-old needs
    the child beside an adult; four scattered middle seats is not the same product."""
    wheelchair_service: bool = True
    checked_bag_fee: Cents = 0
    """Charged per traveller at booking when the fare excludes bags. Part of why the
    headline fare is not the price."""


class RailLeg(FrozenModel):
    """Intercity rail (Shinkansen). Modelled separately from flights because the
    realistic Tokyo-Kyoto-Osaka corridor is rail, and rail almost never cancels —
    which makes "fly or take the train" a genuine robustness tradeoff."""

    key: Key
    operator: str
    origin_city: CityCode
    destination_city: CityCode
    departure_utc: datetime
    arrival_utc: datetime
    duration_minutes: Annotated[int, Field(gt=0)]
    price: Cents
    refundable: bool
    seats_remaining: Annotated[int, Field(ge=0)]
    on_time_rate: Score01


class GroundLeg(FrozenModel):
    """Any scheduled surface leg between two cities — rail, bus, ferry or a car hire block.

    Generalises `RailLeg` (kept unchanged so the v1 world still builds). The mode is what
    makes the choice interesting: a ferry to Krabi is the only sensible way to get there and
    it is the single most weather-fragile thing in the world; an overnight bus is cheap and
    costs you a hotel night; a car hire has no schedule but needs a licensed driver and
    cannot be taken by a party whose only adult does not drive.
    """

    key: Key
    mode: TransportMode
    operator: str
    origin_city: CityCode
    destination_city: CityCode
    departure_utc: datetime
    arrival_utc: datetime
    duration_minutes: Annotated[int, Field(gt=0)]
    price: Cents
    """Per traveller."""
    refundable: bool
    seats_remaining: Annotated[int, Field(ge=0)]
    on_time_rate: Score01
    step_free: bool = True
    """Whether a wheelchair can board. Regional buses and older ferries frequently cannot."""
    luggage_allowance: Annotated[int, Field(ge=0)] = 2
    """Bags per traveller. Zero on some budget buses, which strands a family with cases."""
    weather_sensitive: bool = False
    """Ferries and mountain rail. Act II cancels these on a whole-day weather roll."""
    requires_driver: bool = False
    """Car hire: someone in the party must be 21+ and it is their whole day."""
    booking_lead_days: Annotated[int, Field(ge=0)] = 0
    """Some passes and reserved seats cannot be bought at the last minute."""


# ---- Lodging ---------------------------------------------------------------------


class LodgingType(StrEnum):
    """Occupancy rules differ by type, and that is the point.

    A capsule sleeps exactly one person and cannot take a family. An apartment sleeps six in
    one unit and is far cheaper per head than three hotel rooms. A ryokan comes with dinner
    and a futon on a tatami floor, which is charming and is not step-free.
    """

    HOTEL = "hotel"
    RYOKAN = "ryokan"
    HOSTEL = "hostel"
    APARTMENT = "apartment"
    CAPSULE = "capsule"
    RESORT = "resort"
    GUESTHOUSE = "guesthouse"


class BoardType(StrEnum):
    ROOM_ONLY = "room_only"
    BREAKFAST = "breakfast"
    HALF_BOARD = "half_board"


class RatePlan(StrEnum):
    NON_REFUNDABLE = "non_refundable"
    FREE_CANCELLATION = "free_cancellation"


class Hotel(FrozenModel):
    key: Key
    name: str
    city: CityCode
    district: Key
    coords: Coordinates
    star_rating: Annotated[int, Field(ge=1, le=5)]
    guest_rating: Annotated[float, Field(ge=0.0, le=10.0)]
    review_count: Annotated[int, Field(ge=0)]
    """Low review counts make guest_rating noisy on purpose: a 9.8 from 11 reviews is
    weaker evidence than an 8.6 from 2,400, and a good agent should discount it."""
    amenities: frozenset[str]
    minutes_to_station: Annotated[int, Field(ge=0)]

    # ---- v2 fields ----
    lodging_type: LodgingType = LodgingType.HOTEL
    accessibility: frozenset[AccessibilityNeed] = frozenset()
    """What this property can actually satisfy. Absence is a hard gate, not a minus point."""
    allows_service_animal: bool = True
    dietary_supported: frozenset[DietaryNeed] = frozenset()
    check_in_from_hour: Annotated[int, Field(ge=0, le=23)] = 15
    check_in_until_hour: Annotated[int, Field(ge=1, le=24)] = 24
    """LOCAL hour after which reception closes; 24 means a 24-hour desk. Landing at 23:40
    against a 22:00 cutoff loses the room — free realism, and it is decided by the flight
    the agent chose, which is what makes it causal rather than random."""
    min_nights: Annotated[int, Field(ge=1)] = 1


class RoomOffer(FrozenModel):
    key: Key
    hotel: Key
    board: BoardType
    rate_plan: RatePlan
    price_per_night: Cents
    max_occupancy: Annotated[int, Field(ge=1, le=6)]
    """Heads this unit sleeps WITHOUT a cot."""
    free_cancellation_until: datetime | None = None
    rooms_remaining: Annotated[int, Field(ge=0)]

    # ---- v2 fields ----
    cot_available: bool = False
    cot_price_per_night: Cents = 0
    """A cot raises occupancy by one, for a child only, at a nightly charge. This is the
    detail that decides whether a family of five needs one apartment or three hotel rooms."""
    advance_purchase_days: Annotated[int, Field(ge=0)] = 0
    """Rates that only exist far enough ahead. Inside the window the offer is simply gone."""


# ---- Activities ------------------------------------------------------------------


class ActivityCategory(StrEnum):
    FOOD = "food"
    HISTORY = "history"
    MUSEUM = "museum"
    NATURE = "nature"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    FAMILY = "family"


class Activity(FrozenModel):
    key: Key
    name: str
    city: CityCode
    district: Key
    category: ActivityCategory
    duration_minutes: Annotated[int, Field(gt=0)]
    price_per_person: Cents
    rating: Annotated[float, Field(ge=0.0, le=10.0)]
    opens_hour: Annotated[int, Field(ge=0, le=23)]
    closes_hour: Annotated[int, Field(ge=1, le=24)]
    closed_weekdays: frozenset[int]
    """ISO weekday numbers (Mon=1). Most museums close one day a week; scheduling one
    on its closed day is a plan that fails to construct, not a plan that scores low."""
    booking_required: bool
    daily_capacity: Annotated[int, Field(ge=0)]
    physical_intensity: Level

    # ---- v2 fields ----
    accessible: bool = True
    low_stimulus: bool = True
    """False for anything crowded or loud. A sensory-sensitive traveller cannot attend."""
    min_age: Annotated[int, Field(ge=0)] = 0
    max_age: Annotated[int, Field(ge=1)] = 120
    """Bar crawls have a floor; strenuous hikes effectively have a ceiling."""
    max_group_size: Annotated[int, Field(ge=1)] = 99
    lead_time_days: Annotated[int, Field(ge=0)] = 0
    """Kaiseki and famous studios book out weeks ahead. Inside the window it is unavailable."""
    weather_dependent: bool = False
    dietary_supported: frozenset[DietaryNeed] = frozenset()
    """Only meaningful for FOOD. A kaiseki dinner that cannot do vegan is not a plan for a
    vegan client, and finding that out at the table is an act II failure."""
    season_months: frozenset[int] = frozenset(range(1, 13))
    """Calendar months this runs at all. Cherry-blossom cruises do not sail in August."""


# ---- Client model ----------------------------------------------------------------


class PreferenceAxis(StrEnum):
    """The latent utility dimensions. The client's weights over these are hidden; the
    opening request reveals only a couple of them. Eliciting the rest is the skill."""

    COST = "cost"
    COMFORT = "comfort"
    LOCATION = "location"
    PACING = "pacing"
    FOOD = "food"
    CULTURE = "culture"
    NOVELTY = "novelty"
    RELIABILITY = "reliability"


class AgeBand(StrEnum):
    YOUNG_ADULT = "18_29"
    ADULT = "30_44"
    MIDLIFE = "45_59"
    SENIOR = "60_plus"


class PartyType(StrEnum):
    SOLO = "solo"
    COUPLE = "couple"
    FRIENDS = "friends"
    FAMILY_WITH_KIDS = "family_with_kids"
    BUSINESS = "business"


class CommunicationStyle(StrEnum):
    TERSE = "terse"
    CASUAL = "casual"
    FORMAL = "formal"
    CHATTY = "chatty"


class ClientPersona(FrozenModel):
    """A client. `weights` is the hidden utility vector the whole reward turns on.

    Demographics are not decoration: they are correlated with the behavioural
    parameters (see `personas.draw_persona`), so a 68-year-old formal traveller and a
    24-year-old terse one produce genuinely different episodes rather than the same
    episode with different prose.
    """

    key: Key
    age_band: AgeBand
    party_type: PartyType
    party_size: Annotated[int, Field(ge=1, le=6)]
    communication_style: CommunicationStyle
    weights: dict[PreferenceAxis, Level]
    """Hidden. Never appears in an observation; only its consequences do."""
    stated_axes: frozenset[PreferenceAxis]
    """The subset the opening request actually mentions. Everything else must be asked."""
    budget_cents: Cents
    budget_flexibility: Score01
    """0.0 = hard ceiling, 1.0 = will stretch ~25% for something they love."""
    initial_patience: Annotated[int, Field(gt=0)]
    patience_decay_per_turn: Annotated[float, Field(gt=0.0)]
    tech_savvy: Level
    risk_tolerance: Level
    reply_verbosity: Level
    """Drives how much a single reply reveals. Most real clients answer in a few words."""

    # ---- v2 fields. Empty tuples reproduce v1 exactly. ----
    travellers: tuple[Traveller, ...] = ()
    """The manifest. When non-empty, `len(travellers) == party_size` and the manifest is
    authoritative for every hard gate; `party_size` remains the headcount used for pricing.

    HIDDEN, like `weights`. The opening request says "family of four" — it does not say one
    of them uses a wheelchair and another is seven years old. Those facts have to be asked
    for, and an agent that does not ask books a trip that fails at the door.
    """
    fixed_commitments: tuple[FixedCommitment, ...] = ()
    """Immovable points: a wedding, a conference session, a non-movable meeting. The whole
    itinerary has to route around them."""
    home_region: str = "US"


# ---- Documents and eligibility ---------------------------------------------------


class VisaRule(FrozenModel):
    """Whether this passport may enter this country, and what it takes.

    A hard gate at the level of a whole destination: no visa, no region. It is the one
    constraint an agent cannot fix by choosing a different hotel.
    """

    nationality: Nationality
    country: str
    visa_required: bool
    lead_time_days: Annotated[int, Field(ge=0)] = 0
    """Working days the consulate needs. Inside this window the trip cannot happen."""
    fee_cents: Cents = 0
    passport_validity_months: Annotated[int, Field(ge=0)] = 6
    """The six-month rule: the passport must outlast the return date by this margin."""


# ---- Bookings and the ledger -----------------------------------------------------


class BookingKind(StrEnum):
    FLIGHT = "flight"
    RAIL = "rail"
    HOTEL = "hotel"
    ACTIVITY = "activity"
    GROUND = "ground"
    ANCILLARY = "ancillary"
    """Cots, bags, visa fees, seat reservations. Real money, not part of any headline fare."""


class BookingStatus(StrEnum):
    HELD = "held"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    DISRUPTED = "disrupted"
    """The supplier broke it. Distinct from CANCELLED, which the agent chose."""
    COMPLETED = "completed"


class Booking(FrozenModel):
    key: Key
    kind: BookingKind
    inventory_key: Key
    status: BookingStatus
    price_paid: Cents
    refundable: bool
    refund_deadline: datetime | None
    travellers: Annotated[int, Field(ge=1)]
    service_date: date
    depends_on: tuple[Key, ...] = ()
    """Bookings whose failure cascades into this one — the edge set of the cascade
    graph. A hotel night depends on the flight that gets you there."""


class LedgerEntryKind(StrEnum):
    CHARGE = "charge"
    REFUND = "refund"
    CHANGE_FEE = "change_fee"
    PENALTY = "penalty"


class LedgerEntry(FrozenModel):
    """One money movement, recorded at the instant it happens.

    Reward is computed from the ledger, never re-derived from the final itinerary:
    searching a finished plan for "what did this cost" finds the wrong number the
    moment a refund or a rebooking has moved things around.
    """

    sequence: Annotated[int, Field(ge=0)]
    turn: Annotated[int, Field(ge=0)]
    kind: LedgerEntryKind
    booking_key: Key
    amount: SignedCents
    """Signed against the client's wallet: charges negative, refunds positive."""
    balance_after: SignedCents
    reason: str


# ---- Disruption ------------------------------------------------------------------


class DisruptionKind(StrEnum):
    FLIGHT_CANCELLED = "flight_cancelled"
    FLIGHT_DELAYED = "flight_delayed"
    MISSED_CONNECTION = "missed_connection"
    HOTEL_OVERBOOKED = "hotel_overbooked"
    ACTIVITY_CLOSED = "activity_closed"
    WEATHER_CLOSURE = "weather_closure"
    STRIKE = "strike"

    # ---- v2 ----
    GROUND_CANCELLED = "ground_cancelled"
    GROUND_DELAYED = "ground_delayed"
    SELF_TRANSFER_STRANDED = "self_transfer_stranded"
    """A missed connection on two separate tickets. No carrier owes anything; the party
    buys a walk-up fare. The worst single outcome in the world, and it is fully foreseeable
    at booking time — which is what makes it a trap rather than a tax."""
    DENIED_BOARDING = "denied_boarding"
    """Documents. The passport was short of the six-month rule, or the visa never came."""
    ACCESS_FAILURE = "access_failure"
    """The room, leg or activity could not physically be used by someone in the party."""
    CAPACITY_FAILURE = "capacity_failure"
    """Not enough beds, or a child seated away from every adult."""
    UNREACHABLE_IN_TIME = "unreachable_in_time"
    """The timeline broke: the party could not physically be there when it started. This is
    the cross-booking cascade, and it is derived from the clock rather than sampled."""
    COMMITMENT_MISSED = "commitment_missed"
    """The client's own immovable point — the wedding, the meeting — was not attended."""


class DisruptionEvent(FrozenModel):
    key: Key
    kind: DisruptionKind
    target_booking: Key
    trip_day: Annotated[int, Field(ge=0)]
    delay_minutes: Annotated[int, Field(ge=0)] = 0
    caused_by: Key | None = None
    """Set when this event is a downstream consequence of another — the cascade edge."""
    severity: Score01


# ---- The world -------------------------------------------------------------------


class World(FrozenModel):
    """Everything an episode draws on. Produced once per `reset(seed=...)`, immutable
    thereafter; all mutable episode state lives in the environment, never here."""

    seed: int
    generated_at: datetime
    cities: tuple[City, ...]
    airports: tuple[Airport, ...]
    flights: tuple[Flight, ...]
    rail: tuple[RailLeg, ...]
    hotels: tuple[Hotel, ...]
    room_offers: tuple[RoomOffer, ...]
    activities: tuple[Activity, ...]

    # ---- v2 ----
    regions: tuple[Region, ...] = ()
    ground: tuple[GroundLeg, ...] = ()
    visa_rules: tuple[VisaRule, ...] = ()
    window_start_utc: datetime | None = None
    window_end_utc: datetime | None = None
    """The world's clock bounds. Every instant in every record must lie inside them, and
    `clock.assert_world_coherent` proves it. A schedule that walks past the end of the world
    is a generator fault, so it raises there rather than being scored here."""
    origin_airport: Key = "SFO"

    def district_index(self) -> dict[Key, District]:
        return {d.key: d for c in self.cities for d in c.districts}

    def city_index(self) -> dict[CityCode, City]:
        return {c.code: c for c in self.cities}

    def airport_index(self) -> dict[Key, Airport]:
        return {a.key: a for a in self.airports}


# `FixedCommitment` is declared above `CityCode` because the traveller-needs block has to
# precede the supply-side models that reference it, so its annotation is a forward
# reference and Pydantic cannot resolve it until the module finishes executing.
FixedCommitment.model_rebuild()
