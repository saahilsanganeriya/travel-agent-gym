"""Fetch the district-to-district transfer matrix from Google, ONCE, and commit the result.

Run this by hand when the geography changes. It writes `data/transfer_matrix.json`, which
`travel_env/geo.py` loads at import time. **Nothing calls Google at runtime** — a network
call inside `step()` would destroy determinism, add tail latency to a function that runs
millions of times per training run, and make the environment unusable offline.

    python scripts/fetch_transfer_matrix.py               # reads GOOGLE_MAPS_API_KEY from <repo>/.env
    python scripts/fetch_transfer_matrix.py --dry-run     # show the request plan, call nothing

Only same-city ordered pairs are requested: moving between cities is a *booked leg* with a
departure time, not a transfer, so it is not this matrix's job. That is ~160 elements across
13 cities, which is a handful of requests.

Mode selection is per city rather than global, because the honest answer differs by place:
a Venice "drive" time is meaningless, and a Phuket "transit" time mostly does not exist.
Where several modes are plausible the minimum is taken, which is what a traveller would
actually do.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from travel_env.config import google_maps_key
from travel_env.contracts import CityCode
from travel_env.geo import BOAT_ONLY_DISTRICTS
from travel_env.regions import DISTRICTS_BY_CITY, DistrictSpec, district_key

ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
OUT_PATH = Path("data/transfer_matrix.json")

# Which travel modes are meaningful where. Venice has no roads; Phuket and Krabi have almost
# no scheduled transit; Florence's centre is closed to traffic.
MODES_BY_CITY: dict[CityCode, tuple[str, ...]] = {
    CityCode.VENICE: ("transit", "walking"),
    CityCode.FLORENCE: ("walking", "transit"),
    CityCode.PHUKET: ("driving",),
    CityCode.KRABI: ("driving",),
    CityCode.CHIANG_MAI: ("driving", "transit"),
    CityCode.HAKONE: ("transit", "driving"),
    CityCode.KANAZAWA: ("transit", "driving"),
}
_DEFAULT_MODES = ("transit", "driving")


def departure_timestamp() -> int:
    """A fixed future weekday mid-morning.

    Transit routing needs a departure time and refuses one in the past, so this cannot be a
    hard-coded constant that ages out. Reproducibility comes from the committed JSON, not
    from re-running this script — the timestamp used is recorded in the file's metadata.
    """
    when = datetime.now(UTC) + timedelta(days=30)
    while when.isoweekday() > 5:  # a Tuesday-ish weekday, not a Sunday timetable
        when += timedelta(days=1)
    return int(when.replace(hour=10, minute=0, second=0, microsecond=0).timestamp())


def request_matrix(
    key: str, districts: list[DistrictSpec], mode: str, depart_at: int
) -> tuple[list[list[int | None]], str]:
    """One Distance Matrix call for a whole city. Returns (minutes grid, top-level status)."""
    coords = "|".join(f"{d.lat},{d.lon}" for d in districts)
    params = {
        "origins": coords,
        "destinations": coords,
        "mode": mode,
        "units": "metric",
        "key": key,
    }
    if mode in ("transit", "driving"):
        params["departure_time"] = str(depart_at)

    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 — fixed https host
        payload = json.loads(response.read().decode())

    status = payload.get("status", "UNKNOWN")
    if status != "OK":
        return [], f"{status}: {payload.get('error_message', '')}".strip()

    grid: list[list[int | None]] = []
    for row in payload.get("rows", []):
        out_row: list[int | None] = []
        for element in row.get("elements", []):
            duration = element.get("duration")
            if element.get("status") != "OK" or not isinstance(duration, dict):
                out_row.append(None)
                continue
            out_row.append(int(round(duration["value"] / 60)))
        grid.append(out_row)
    return grid, "OK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the plan, call nothing")
    args = parser.parse_args()

    plan = {city: list(specs) for city, specs in DISTRICTS_BY_CITY.items() if len(specs) > 1}
    elements = sum(len(s) * (len(s) - 1) for s in plan.values())
    print(f"{len(plan)} cities, {elements} ordered same-city pairs")

    if args.dry_run:
        for city, specs in plan.items():
            modes = MODES_BY_CITY.get(city, _DEFAULT_MODES)
            print(f"  {city.value:>4}  {len(specs)} districts  modes={'+'.join(modes)}")
        return 0

    key = google_maps_key()
    if not key:
        print(
            "No key found. Add GOOGLE_MAPS_API_KEY to <repo>/.env (see .env.example).\n"
            "Without it geo.py falls back to per-city transit profiles, which still work."
        )
        return 1

    depart_at = departure_timestamp()
    minutes: dict[str, int] = {}
    sources: dict[str, str] = {}
    failures: list[str] = []

    for city, specs in plan.items():
        modes = MODES_BY_CITY.get(city, _DEFAULT_MODES)
        best: dict[tuple[int, int], tuple[int, str]] = {}
        for mode in modes:
            grid, status = request_matrix(key, specs, mode, depart_at)
            if status != "OK":
                failures.append(f"{city.value}/{mode}: {status}")
                continue
            for i, row in enumerate(grid):
                for j, value in enumerate(row):
                    if i == j or value is None:
                        continue
                    prior = best.get((i, j))
                    # A traveller takes whichever mode is quicker, so the minimum is the
                    # honest number rather than a per-mode average.
                    if prior is None or value < prior[0]:
                        best[(i, j)] = (value, mode)
            time.sleep(0.12)  # be polite; this is a one-off, throughput does not matter

        for (i, j), (value, mode) in best.items():
            pair = f"{district_key(specs[i].name)}|{district_key(specs[j].name)}"
            minutes[pair] = max(6, value)
            sources[pair] = mode
        print(f"  {city.value:>4}  {len(best):>3} pairs measured")

    if not minutes:
        print("\nNothing measured. Failures:\n  " + "\n  ".join(failures or ["(none reported)"]))
        print(
            "\nIf this says REQUEST_DENIED, enable 'Distance Matrix API' for the project in\n"
            "Google Cloud Console -> APIs & Services -> Library, and confirm billing is on."
        )
        return 1

    # Boat-only districts are not a routing result — Google will happily return a driving
    # time to Railay that does not exist, because there is no road. Keep the modelled value.
    dropped = [p for p in minutes if p.split("|")[0] in BOAT_ONLY_DISTRICTS or p.split("|")[1] in BOAT_ONLY_DISTRICTS]
    for pair in dropped:
        del minutes[pair]
        sources.pop(pair, None)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "Google Distance Matrix, fetched once and committed. Never called at "
                    "runtime — see travel_env/geo.py. Keys are '<from_district>|<to_district>', "
                    "values are door-to-door minutes."
                ),
                "fetched_at": datetime.now(UTC).isoformat(),
                "departure_time_used": depart_at,
                "mode_by_pair": sources,
                "boat_only_pairs_excluded": sorted(dropped),
                "failures": failures,
                "minutes": dict(sorted(minutes.items())),
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT_PATH} — {len(minutes)} pairs")
    if failures:
        print("partial failures:\n  " + "\n  ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
