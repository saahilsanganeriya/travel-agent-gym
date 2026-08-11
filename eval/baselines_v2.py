"""Baseline policies for v2, and the measurement that justifies v2 existing.

These are a design instrument, not a scoreboard. v1's baselines said the environment was
too easy — random 25%, heuristic 96%, both far above their targets — and no amount
of reading the code would have revealed that. The same instrument has to be pointed at v2.

`heuristic` is deliberately the *same competent agent as v1's*: interview the client on the
preference axes, then search against what you learned, then book. What it does NOT do is ask
about accessibility, capacity or documents — because a competent-but-incurious agent is
exactly the thing v2 is built to catch. `expert` is the same policy plus those four
questions, and the gap between them is the value of interviewing, measured.
"""

from __future__ import annotations

import numpy as np

from travel_env.actions_v2 import (
    SLATE_K,
    FlightPreset,
    GroundPreset,
    HotelSort,
    LodgingFilter,
    Verb,
    encode,
    encode_hotel_search,
)
from travel_env.constraints import ASKABLE
from travel_env.contracts import PreferenceAxis


def _legal(info: dict) -> np.ndarray:
    return np.flatnonzero(info["action_mask"])


def _first(info: dict, *candidates: int) -> int | None:
    mask = info["action_mask"]
    return next((c for c in candidates if mask[c]), None)


class NullPolicy:
    """Does the least it can. The mask guarantees ASK is always available, so this is a
    single repeated question. Should score ~0."""

    name = "null"

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        return int(_legal(info)[0])


class RandomPolicy:
    """Uniform over legal actions. The reward-scale reference point, and the number that
    says whether the task can be completed by accident."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        return int(self.rng.choice(_legal(info)))


class _RoutedGreedy:
    """Shared skeleton for the two obvious exploits.

    Both have to build a route before they can search anything, so the route-building is
    common; only the search preset differs. Neither ever asks the client anything, which is
    the whole point — they are betting on a prior.
    """

    name = "greedy"
    flight_preset = FlightPreset.CHEAPEST
    hotel_sort = HotelSort.CHEAPEST
    ground_preset = GroundPreset.CHEAPEST

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        mask = info["action_mask"]
        core = env.unwrapped
        plan = [
            encode(Verb.SELECT_FLIGHT, 0),
            encode(Verb.SELECT_GROUND, 0),
            encode(Verb.SELECT_HOTEL, 0),
            encode(Verb.PROPOSE),
            encode(Verb.BOOK),
            encode(Verb.FINALIZE),
        ]
        if not core.destination_slate:
            a = _first(info, encode(Verb.SEARCH_DESTINATIONS))
            if a is not None:
                return a
        if sum(n for _, n in core.route) < core.nights:
            a = _first(info, *(encode(Verb.ADD_CITY, i) for i in range(SLATE_K)))
            if a is not None:
                return a
        for a in plan:
            if mask[a]:
                return a
        for a in (
            encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(self.flight_preset)),
            encode(Verb.SEARCH_GROUND, list(GroundPreset).index(self.ground_preset)),
            encode_hotel_search(self.hotel_sort, LodgingFilter.ANY),
        ):
            if mask[a]:
                return a
        return int(_legal(info)[0])


class GreedyCheapestPolicy(_RoutedGreedy):
    """'Always book the cheapest thing.' Implemented so it can be measured, not assumed."""

    name = "greedy_cheapest"


class GreedyQualityPolicy(_RoutedGreedy):
    """'Always pick the highest-rated hotel.' The other named exploit."""

    name = "greedy_quality"
    flight_preset = FlightPreset.MOST_RELIABLE
    hotel_sort = HotelSort.BEST_RATED
    ground_preset = GroundPreset.ROBUST


class HeuristicPolicy:
    """A competent human travel agent who never asks about hard constraints.

    Interviews on preferences, searches against what it learned, builds a route, books, and
    confirms. This is the policy that scored 96% in v1. In v2 it should fall a long way,
    because it books rooms the party cannot use and flights it cannot legally board — and it
    never finds out until the trip runs.
    """

    name = "heuristic"
    ask_constraints = False

    def __init__(self, ask_budget: int = 5) -> None:
        self.ask_budget = ask_budget
        self.asked: list[int] = []
        self.asked_classes = 0
        self.tried: set[int] = set()

    def reset(self) -> None:
        self.asked = []
        self.asked_classes = 0
        self.tried: set[int] = set()

    def act(self, obs: dict, info: dict, env) -> int:
        mask = info["action_mask"]
        core = env.unwrapped

        # 1. Interview. The hidden weights are the whole game.
        if len(self.asked) < self.ask_budget:
            for axis_i in np.argsort(obs["belief"][:, 1]):
                a = encode(Verb.ASK, int(axis_i))
                if mask[a] and int(axis_i) not in self.asked:
                    self.asked.append(int(axis_i))
                    return a
        if self.ask_constraints and self.asked_classes < len(ASKABLE):
            a = _first(info, *(encode(Verb.ASK_CONSTRAINTS, i) for i in range(len(ASKABLE))))
            if a is not None:
                self.asked_classes += 1
                return a

        belief = obs["belief"]
        cost_w = belief[list(PreferenceAxis).index(PreferenceAxis.COST), 0]
        rel_w = belief[list(PreferenceAxis).index(PreferenceAxis.RELIABILITY), 0]
        comfort_w = belief[list(PreferenceAxis).index(PreferenceAxis.COMFORT), 0]

        # 2. Build a route before anything can be searched for.
        if not core.destination_slate:
            a = _first(info, encode(Verb.SEARCH_DESTINATIONS))
            if a is not None:
                return a
        if sum(n for _, n in core.route) < core.nights:
            a = _first(info, *(encode(Verb.ADD_CITY, i) for i in range(SLATE_K)))
            if a is not None:
                return a

        # 3. Flights, with a preset that matches what the client actually said.
        if core.timeline.outbound is None or core.timeline.inbound is None:
            a = _first(info, *(encode(Verb.SELECT_FLIGHT, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            preset = (
                FlightPreset.MOST_RELIABLE if rel_w > 0.6
                else FlightPreset.CHEAPEST if cost_w > 0.5
                else FlightPreset.FASTEST
            )
            # Try each preset ONCE. Repeating a search that surfaced nothing affordable is
            # not a move; a real agent widens the net instead of asking the same question.
            for p in (preset, FlightPreset.CHEAPEST, FlightPreset.FASTEST, FlightPreset.FLEXIBLE):
                a = encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(p))
                if mask[a] and a not in self.tried:
                    self.tried.add(a)
                    return a

        # 4. Intercity legs for every hop the route implies.
        if core._next_hop() is not None:
            a = _first(info, *(encode(Verb.SELECT_GROUND, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            for p in (GroundPreset.ROBUST if rel_w > 0.6 else GroundPreset.CHEAPEST,
                      GroundPreset.CHEAPEST, GroundPreset.FASTEST, GroundPreset.OVERNIGHT):
                a = encode(Verb.SEARCH_GROUND, list(GroundPreset).index(p))
                if mask[a] and a not in self.tried:
                    self.tried.add(a)
                    return a

        # 5. A bed in every route city. Re-search cheaper rather than standing there, which
        #    is what a competent agent does when a slate has nothing affordable on it.
        if core._pending_city() is not None:
            a = _first(info, *(encode(Verb.SELECT_HOTEL, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            sort = HotelSort.BEST_RATED if comfort_w > 0.55 else HotelSort.CHEAPEST
            for s in (sort, HotelSort.CHEAPEST, HotelSort.MOST_CENTRAL):
                for f in (LodgingFilter.ANY, LodgingFilter.BUDGET, LodgingFilter.WHOLE_UNIT):
                    a = encode_hotel_search(s, f)
                    if mask[a] and (a, core._pending_city()) not in self.tried:
                        self.tried.add((a, core._pending_city()))
                        return a

        # 6. Activities are upside, then confirm, commit and close out, in that order.
        a = _first(info, *(encode(Verb.SELECT_ACTIVITY, i) for i in range(SLATE_K)))
        if a is not None and len(core.timeline.activities) < 4:
            return a
        if len(core.timeline.activities) < 4:
            a = _first(info, *(encode(Verb.SEARCH_ACTIVITIES, i) for i in range(7)))
            if a is not None:
                return a
        for verb in (Verb.PROPOSE, Verb.BOOK, Verb.FINALIZE):
            a = _first(info, encode(verb))
            if a is not None:
                return a
        return int(_legal(info)[0])


class ExpertPolicy(HeuristicPolicy):
    """The heuristic plus the four constraint questions.

    The gap between this and `heuristic` is the measured value of interviewing a client
    properly — the single thing v2 was built to make expensive to skip.
    """

    name = "expert"
    ask_constraints = True


ALL_POLICIES: dict[str, type] = {
    "null": NullPolicy,
    "random": RandomPolicy,
    "greedy_cheapest": GreedyCheapestPolicy,
    "greedy_quality": GreedyQualityPolicy,
    "heuristic": HeuristicPolicy,
    "expert": ExpertPolicy,
}
