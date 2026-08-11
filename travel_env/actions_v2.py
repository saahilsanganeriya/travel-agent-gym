"""The v2 action space: 89 actions over a route-building travel agent.

Kept in its own module so `actions.py` — and therefore the v1 `Discrete(63)` layout and the
checkpoint trained against it — is untouched. Versioned ids, not a silent break.

**What is new, and why.** v1 let the agent search hotels in one of three fixed cities. With
thirteen cities across three regions that does not scale, and more importantly it skips the
decision that actually matters: *where does this trip go, and for how long in each place?*
So v2 makes the route an explicit object the agent builds:

    SEARCH_DESTINATIONS -> a slate of candidate cities
    ADD_CITY(i)         -> append that city, or extend the current stay by one night
    DROP_CITY           -> undo the last one

`ADD_CITY` adding a single night is the design decision worth defending. A one-shot "add city
with N nights" would need a second argument and double the space; repeated calls express the
same allocation, cost the agent turns it must budget, and make "four nights in Kyoto, two in
Osaka" a sequence of deliberate choices rather than one lucky draw. Random cannot stumble
into a coherent allocation over a seven-night trip.

Everything else follows v1's compression: search with a filter preset, get at most eight
candidates, select by index. Illegal actions are masked, never penalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

SLATE_K = 8

MAX_ROUTE_CITIES = 6
"""Cap on route length. Six is more than any sane one-week trip and keeps the observation
fixed-width; the constraint that actually binds is nights, not this."""


class Verb(IntEnum):
    ASK = 0
    ASK_CONSTRAINTS = 1
    SEARCH_DESTINATIONS = 2
    ADD_CITY = 3
    DROP_CITY = 4
    SEARCH_FLIGHTS = 5
    SEARCH_GROUND = 6
    SEARCH_HOTELS = 7
    SEARCH_ACTIVITIES = 8
    SELECT_FLIGHT = 9
    SELECT_GROUND = 10
    SELECT_HOTEL = 11
    SELECT_ACTIVITY = 12
    PROPOSE = 13
    BOOK = 14
    FINALIZE = 15
    REBOOK = 16


class FlightPreset(StrEnum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    MOST_RELIABLE = "most_reliable"
    FLEXIBLE = "flexible"


class GroundPreset(StrEnum):
    """Surface transport, filtered by what the client is buying.

    Not sort orders — genuinely different candidate sets. `ROBUST` excludes the weather-
    sensitive ferry and the unreserved bus; `OVERNIGHT` returns only legs that span the night
    and therefore save a hotel, which is a real budget lever and a real comfort cost.
    """

    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    ROBUST = "robust"
    OVERNIGHT = "overnight"


class HotelSort(StrEnum):
    CHEAPEST = "cheapest"
    BEST_RATED = "best_rated"
    MOST_CENTRAL = "most_central"


class LodgingFilter(StrEnum):
    """Which slice of the lodging market to look at.

    A party of five is a different problem from a solo traveller: `WHOLE_UNIT` surfaces
    apartments that sleep everyone in one booking, `BUDGET` surfaces hostels and capsules
    that do not. Making this a search axis rather than a sort is what turns occupancy into a
    decision instead of a filter the agent never sees.
    """

    ANY = "any"
    WHOLE_UNIT = "whole_unit"
    BUDGET = "budget"


_N_AXES = 8
_N_CONSTRAINT_CLASSES = 4
_N_HOTEL_SORTS = len(HotelSort)
_N_LODGING_FILTERS = len(LodgingFilter)
_N_CATEGORIES = 7

_SEGMENTS: tuple[tuple[Verb, int], ...] = (
    (Verb.ASK, _N_AXES),
    (Verb.ASK_CONSTRAINTS, _N_CONSTRAINT_CLASSES),
    (Verb.SEARCH_DESTINATIONS, 1),
    (Verb.ADD_CITY, SLATE_K),
    (Verb.DROP_CITY, 1),
    (Verb.SEARCH_FLIGHTS, len(FlightPreset)),
    (Verb.SEARCH_GROUND, len(GroundPreset)),
    (Verb.SEARCH_HOTELS, _N_HOTEL_SORTS * _N_LODGING_FILTERS),
    (Verb.SEARCH_ACTIVITIES, _N_CATEGORIES),
    (Verb.SELECT_FLIGHT, SLATE_K),
    (Verb.SELECT_GROUND, SLATE_K),
    (Verb.SELECT_HOTEL, SLATE_K),
    (Verb.SELECT_ACTIVITY, SLATE_K),
    (Verb.PROPOSE, 1),
    (Verb.BOOK, 1),
    (Verb.FINALIZE, 1),
    (Verb.REBOOK, SLATE_K),
)

_OFFSETS: dict[Verb, int] = {}
_SIZES: dict[Verb, int] = {}
_cursor = 0
for _verb, _size in _SEGMENTS:
    _OFFSETS[_verb] = _cursor
    _SIZES[_verb] = _size
    _cursor += _size
N_ACTIONS = _cursor  # 89

# Decoding on the hot path: one table lookup instead of a linear scan over the segments,
# because `decode` runs on every step of every environment of every rollout.
_DECODE: tuple[tuple[Verb, int], ...] = tuple(
    (verb, index - _OFFSETS[verb])
    for index in range(N_ACTIONS)
    for verb, size in _SEGMENTS
    if _OFFSETS[verb] <= index < _OFFSETS[verb] + size
)


@dataclass(frozen=True)
class DecodedAction:
    verb: Verb
    arg: int

    @property
    def sort_index(self) -> int:
        """SEARCH_HOTELS packs (sort, lodging filter) into one argument."""
        return self.arg // _N_LODGING_FILTERS

    @property
    def filter_index(self) -> int:
        return self.arg % _N_LODGING_FILTERS


def decode(action: int) -> DecodedAction:
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} outside [0, {N_ACTIONS})")
    verb, arg = _DECODE[action]
    return DecodedAction(verb=verb, arg=arg)


def encode(verb: Verb, arg: int = 0) -> int:
    size = _SIZES[verb]
    if not 0 <= arg < size:
        raise ValueError(f"arg {arg} outside [0, {size}) for {verb.name}")
    return _OFFSETS[verb] + arg


def encode_hotel_search(sort: HotelSort, lodging: LodgingFilter) -> int:
    return encode(
        Verb.SEARCH_HOTELS,
        list(HotelSort).index(sort) * _N_LODGING_FILTERS + list(LodgingFilter).index(lodging),
    )


def verb_slice(verb: Verb) -> slice:
    base = _OFFSETS[verb]
    return slice(base, base + _SIZES[verb])


def empty_mask() -> np.ndarray:
    """All-illegal starting point. Callers switch on what is legal, never off what is not, so
    a forgotten case fails closed rather than open."""
    return np.zeros(N_ACTIONS, dtype=np.int8)


def describe(action: int) -> str:
    d = decode(action)
    match d.verb:
        case Verb.ASK:
            return f"ask about axis {d.arg}"
        case Verb.ASK_CONSTRAINTS:
            return f"ask about constraint class {d.arg}"
        case Verb.SEARCH_HOTELS:
            return (
                f"search lodging ({list(HotelSort)[d.sort_index].value}, "
                f"{list(LodgingFilter)[d.filter_index].value})"
            )
        case Verb.SEARCH_FLIGHTS:
            return f"search flights ({list(FlightPreset)[d.arg].value})"
        case Verb.SEARCH_GROUND:
            return f"search ground ({list(GroundPreset)[d.arg].value})"
        case Verb.SEARCH_ACTIVITIES:
            return f"search activities (category {d.arg})"
        case Verb.ADD_CITY:
            return f"add a night in destination [{d.arg}]"
        case (
            Verb.SELECT_FLIGHT | Verb.SELECT_GROUND | Verb.SELECT_HOTEL
            | Verb.SELECT_ACTIVITY | Verb.REBOOK
        ):
            return f"{d.verb.name.lower()} [{d.arg}]"
        case _:
            return d.verb.name.lower()
