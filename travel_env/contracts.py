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


# ---- Geography -------------------------------------------------------------------


class CityCode(StrEnum):
    TOKYO = "TYO"
    KYOTO = "UKY"
    OSAKA = "OSA"


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


class Airport(FrozenModel):
    key: Key
    name: str
    city: CityCode
    coords: Coordinates
    minutes_to_centre: Annotated[int, Field(ge=0)]


# ---- Transport -------------------------------------------------------------------


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


# ---- Lodging ---------------------------------------------------------------------


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


class RoomOffer(FrozenModel):
    key: Key
    hotel: Key
    board: BoardType
    rate_plan: RatePlan
    price_per_night: Cents
    max_occupancy: Annotated[int, Field(ge=1, le=6)]
    free_cancellation_until: datetime | None = None
    rooms_remaining: Annotated[int, Field(ge=0)]


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


# ---- Bookings and the ledger -----------------------------------------------------


class BookingKind(StrEnum):
    FLIGHT = "flight"
    RAIL = "rail"
    HOTEL = "hotel"
    ACTIVITY = "activity"


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

    def district_index(self) -> dict[Key, District]:
        return {d.key: d for c in self.cities for d in c.districts}
