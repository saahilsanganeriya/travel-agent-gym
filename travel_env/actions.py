"""The action space, its encoding, and the masks that make illegal actions unavailable.

The design decision that matters here is **compression**. The agent never picks "flight
#1,847 out of 2,000". It issues a *search with a filter preset*, which produces a slate of
at most eight candidates, and then *selects by index*. That mirrors how a travel agent
actually works, keeps `Discrete(N)` small enough to learn, and makes the observation a
fixed-width tensor rather than an unbounded catalogue.

Every illegal action is removed by `action_masks()` rather than penalised. The evidence for
that choice is unusually strong: Huang & Ontañón (arXiv:2006.14171) found that across every
penalty weight they tried, the agent "struggles to even find the very first reward" on any
non-trivial map, while masking stayed scale-invariant — and the invalid-action policy
gradient is provably the gradient of the masked policy, with the gradient w.r.t. an invalid
logit exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

SLATE_K = 8
"""Candidates shown per search. Fixed so the observation is fixed-width."""


class Verb(IntEnum):
    ASK = 0
    SEARCH_FLIGHTS = 1
    SEARCH_HOTELS = 2
    SEARCH_ACTIVITIES = 3
    SELECT_FLIGHT = 4
    SELECT_HOTEL = 5
    SELECT_ACTIVITY = 6
    PROPOSE = 7
    BOOK = 8
    FINALIZE = 9
    REBOOK = 10


class FlightPreset(StrEnum):
    """Search filters, not sort orders — each returns a different candidate set. The agent
    must learn which preset surfaces the option its client actually wants."""

    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    MOST_RELIABLE = "most_reliable"
    FLEXIBLE = "flexible"


class HotelSort(StrEnum):
    CHEAPEST = "cheapest"
    BEST_RATED = "best_rated"
    MOST_CENTRAL = "most_central"


# Layout of the flattened Discrete space. Kept as an explicit table rather than computed,
# because an off-by-one in an action encoding is invisible at runtime — the agent simply
# learns the wrong thing — and a table can be asserted against in a test.
_N_AXES = 8
_N_FLIGHT_PRESETS = len(FlightPreset)
_N_CITIES = 3
_N_HOTEL_SORTS = len(HotelSort)
_N_CATEGORIES = 7

_SEGMENTS: tuple[tuple[Verb, int], ...] = (
    (Verb.ASK, _N_AXES),
    (Verb.SEARCH_FLIGHTS, _N_FLIGHT_PRESETS),
    (Verb.SEARCH_HOTELS, _N_CITIES * _N_HOTEL_SORTS),
    (Verb.SEARCH_ACTIVITIES, _N_CATEGORIES),
    (Verb.SELECT_FLIGHT, SLATE_K),
    (Verb.SELECT_HOTEL, SLATE_K),
    (Verb.SELECT_ACTIVITY, SLATE_K),
    (Verb.PROPOSE, 1),
    (Verb.BOOK, 1),
    (Verb.FINALIZE, 1),
    (Verb.REBOOK, SLATE_K),
)

_OFFSETS: dict[Verb, int] = {}
_cursor = 0
for _verb, _size in _SEGMENTS:
    _OFFSETS[_verb] = _cursor
    _cursor += _size
N_ACTIONS = _cursor  # 63


@dataclass(frozen=True)
class DecodedAction:
    verb: Verb
    arg: int
    """Meaning depends on the verb: axis index, preset index, slate index, or 0."""

    @property
    def city_index(self) -> int:
        """For SEARCH_HOTELS, the packed (city, sort) argument splits like this."""
        return self.arg // _N_HOTEL_SORTS

    @property
    def sort_index(self) -> int:
        return self.arg % _N_HOTEL_SORTS


def decode(action: int) -> DecodedAction:
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} outside [0, {N_ACTIONS})")
    for verb, size in _SEGMENTS:
        base = _OFFSETS[verb]
        if base <= action < base + size:
            return DecodedAction(verb=verb, arg=action - base)
    raise AssertionError("unreachable: segment table does not cover the space")


def encode(verb: Verb, arg: int = 0) -> int:
    size = dict(_SEGMENTS)[verb]
    if not 0 <= arg < size:
        raise ValueError(f"arg {arg} outside [0, {size}) for {verb.name}")
    return _OFFSETS[verb] + arg


def verb_slice(verb: Verb) -> slice:
    """The contiguous span a verb occupies. Used by the mask builder so mask construction
    and action decoding cannot drift apart."""
    base = _OFFSETS[verb]
    return slice(base, base + dict(_SEGMENTS)[verb])


def empty_mask() -> np.ndarray:
    """All-illegal starting point. Callers switch on what is legal, never off what is not —
    the default is refusal, so a forgotten case fails closed rather than open."""
    return np.zeros(N_ACTIONS, dtype=np.int8)


def describe(action: int) -> str:
    """Human-readable form, for render() and for test failure messages."""
    d = decode(action)
    match d.verb:
        case Verb.ASK:
            return f"ask about axis {d.arg}"
        case Verb.SEARCH_FLIGHTS:
            return f"search flights ({list(FlightPreset)[d.arg].value})"
        case Verb.SEARCH_HOTELS:
            return f"search hotels (city {d.city_index}, {list(HotelSort)[d.sort_index].value})"
        case Verb.SEARCH_ACTIVITIES:
            return f"search activities (category {d.arg})"
        case Verb.SELECT_FLIGHT | Verb.SELECT_HOTEL | Verb.SELECT_ACTIVITY:
            return f"{d.verb.name.lower()} [{d.arg}]"
        case Verb.REBOOK:
            return f"rebook [{d.arg}]"
        case _:
            return d.verb.name.lower()
