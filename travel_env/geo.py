"""How long it actually takes to get from one district to another.

Transfer time is the tightest constraint in the environment — it is what makes an
over-packed day impossible to construct rather than merely low-scoring — so getting it
roughly right matters more than almost any price.

**Three sources, in priority order:**

1. `data/transfer_matrix.json`, a real Google Distance Matrix result fetched **once** at
   build time and committed. `scripts/fetch_transfer_matrix.py` writes it.
2. A per-city transit profile (below), when the matrix has no entry for the pair.
3. Straight-line distance with a generic factor, for anything else.

**Why the API is never called at runtime.** A network call inside `step()` would destroy
determinism, add tail latency to a function that runs a million times per training run, and
make the environment unrunnable offline. Fetch once, commit the numbers, load them into a
dict. The data is real; the runtime dependency is zero.

**Why per-city profiles rather than one constant.** v1 used a single detour factor of 2.2 at
22 km/h everywhere. That is roughly right for Tokyo and badly wrong elsewhere, in ways that
change what plans are feasible:

* **Venice** has no roads. Movement is walking plus vaporetto, at about 4 km/h effective
  with a large detour for the canals — a 1.5 km hop is half an hour.
* **Bangkok** is traffic-bound: the straight-line distance is meaningless and the metro
  covers a fraction of the city.
* **Hakone** is a mountain served by a switchback railway and a ropeway.
* **Railay** in Krabi is reachable only by longtail boat — there is no land route at all.

Those are facts about places, and a travel agent who does not know them plans trips that do
not work. Encoding them is the difference between a world with geography and a world with
coordinates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

from travel_env.contracts import CityCode, Coordinates, District, Key

_MATRIX_PATH = Path(__file__).resolve().parent.parent / "data" / "transfer_matrix.json"

_EARTH_RADIUS_KM = 6371.0


class TransitProfile(NamedTuple):
    """How a city actually moves people.

    `detour` is the ratio of travelled distance to straight-line distance; `kmh` is the
    effective speed once waiting, transfers and the last walk are included; `overhead` is the
    fixed cost of any journey at all — finding the platform, buying the ticket, waiting.
    """

    detour: float
    kmh: float
    overhead_min: int
    walk_kmh: float = 4.6
    note: str = ""


# Effective speeds, not vehicle speeds. A Tokyo metro train does 60 km/h between stations and
# still only gets you across the city at ~22 km/h door to door.
_DEFAULT_PROFILE = TransitProfile(2.2, 22.0, 8, note="generic dense-metro city")

TRANSIT_PROFILES: dict[CityCode, TransitProfile] = {
    CityCode.TOKYO: TransitProfile(2.1, 24.0, 8, note="one of the best metros on earth"),
    CityCode.OSAKA: TransitProfile(2.0, 23.0, 7),
    CityCode.KYOTO: TransitProfile(2.3, 15.0, 9, note="buses, not trains; slow and crowded"),
    CityCode.KANAZAWA: TransitProfile(1.9, 17.0, 8, note="compact, bus-served"),
    CityCode.HAKONE: TransitProfile(3.1, 11.0, 14, note="switchback rail and a ropeway up a mountain"),
    CityCode.ROME: TransitProfile(2.4, 14.0, 10, note="two metro lines and a great deal of traffic"),
    CityCode.FLORENCE: TransitProfile(1.7, 6.0, 5, walk_kmh=4.6, note="walkable centre; driving is restricted"),
    CityCode.VENICE: TransitProfile(2.9, 4.2, 12, walk_kmh=3.4, note="no roads — walking and vaporetto only"),
    CityCode.NAPLES: TransitProfile(2.3, 13.0, 10, note="funiculars and heavy traffic"),
    CityCode.BANGKOK: TransitProfile(2.6, 12.0, 12, note="traffic-bound; BTS/MRT cover part of the city"),
    CityCode.CHIANG_MAI: TransitProfile(1.9, 16.0, 9, note="songthaew and grab; small centre"),
    CityCode.PHUKET: TransitProfile(2.2, 26.0, 12, note="island roads, long distances, few buses"),
    CityCode.KRABI: TransitProfile(2.4, 24.0, 14, note="road network plus boat transfers"),
}

# Pairs where no land route exists. The only way in or out is a boat on somebody else's
# timetable, so the cost is dominated by waiting rather than by distance.
BOAT_ONLY_DISTRICTS: frozenset[str] = frozenset({"railay"})
_BOAT_TRANSFER_MIN = 55

SAME_DISTRICT_MIN = 10
"""Two venues in one district are still not in the same place."""


def haversine_km(a: Coordinates, b: Coordinates) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _load_matrix() -> dict[str, int]:
    """Committed Distance Matrix results, keyed `"<from>|<to>"`. Absent is fine."""
    if not _MATRIX_PATH.exists():
        return {}
    try:
        payload = json.loads(_MATRIX_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: int(v) for k, v in payload.get("minutes", {}).items()}


_MATRIX: dict[str, int] = _load_matrix()


def matrix_coverage() -> int:
    """How many ordered pairs the committed matrix supplies. Reported by `dump_data`, so a
    reader can see at a glance whether the geography is measured or modelled."""
    return len(_MATRIX)


def transfer_minutes(a: District, b: District) -> int:
    """Door-to-door minutes between two districts.

    Cross-city pairs are not this function's job — moving between cities is a *booked leg*
    with a departure time, not a transfer. Asking for one is a caller error and yields a
    deliberately large number so a plan built on it is infeasible rather than quietly wrong.
    """
    if a.key == b.key:
        return SAME_DISTRICT_MIN

    measured = _MATRIX.get(f"{a.key}|{b.key}")
    if measured is not None:
        return measured

    if a.city is not b.city:
        return 24 * 60

    if a.key in BOAT_ONLY_DISTRICTS or b.key in BOAT_ONLY_DISTRICTS:
        return _BOAT_TRANSFER_MIN

    profile = TRANSIT_PROFILES.get(a.city, _DEFAULT_PROFILE)
    km = haversine_km(a.coords, b.coords)
    # Short hops are walked: nobody waits eight minutes for a train to go 700 metres.
    if km * profile.detour < 1.3:
        return max(6, int(round(km * profile.detour / profile.walk_kmh * 60)))
    return int(round(profile.overhead_min + (km * profile.detour) / profile.kmh * 60))


def airport_transfer_minutes(minutes_to_centre: int, district: District) -> int:
    """Airport to a specific district, scaled off the published airport-to-centre time.

    Districts away from the centre cost more, and the scaling uses the city's own transit
    profile so an outlying district in Venice is punished harder than one in Tokyo.
    """
    profile = TRANSIT_PROFILES.get(district.city, _DEFAULT_PROFILE)
    # `walkability` is a decent proxy for centrality within a city.
    off_centre = (9 - district.walkability) / 9.0
    return int(round(minutes_to_centre * (1.0 + 0.7 * off_centre * profile.detour / 2.2)))


def district_pairs(districts: dict[Key, District]) -> list[tuple[District, District]]:
    """Every ordered same-city pair — the exact set `fetch_transfer_matrix.py` requests."""
    out: list[tuple[District, District]] = []
    values = list(districts.values())
    for a in values:
        for b in values:
            if a.key != b.key and a.city is b.city:
                out.append((a, b))
    return out
