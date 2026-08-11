"""Baseline policies.

These are a design instrument, not just a scoreboard. If `random` scores the same as
`heuristic`, the reward is broken — and no amount of reading the code would reveal that.
The `null` baseline exists because tau-bench's own reward gives a do-nothing agent 38%
success on their benchmark (arXiv:2507.02825); an agent returning empty responses
"outperforms a GPT-4o-based agent" there. We assert ours scores ~0.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from travel_env.actions import SLATE_K, FlightPreset, HotelSort, Verb, encode
from travel_env.contracts import PreferenceAxis


class Policy(Protocol):
    name: str

    def act(self, obs: dict, info: dict, env) -> int: ...
    def reset(self) -> None: ...


def _legal(info: dict) -> np.ndarray:
    return np.flatnonzero(info["action_mask"])


def _first_legal(info: dict, *candidates: int) -> int | None:
    mask = info["action_mask"]
    for c in candidates:
        if mask[c]:
            return c
    return None


class NullPolicy:
    """Does the least it can. Should score ~0 — asserted in tests."""

    name = "null"

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        # The mask guarantees ASK is always available outside a terminal state, so the
        # laziest legal action is a single repeated question.
        return int(_legal(info)[0])


class RandomPolicy:
    """Uniform over legal actions. The reward-scale reference point."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        return int(self.rng.choice(_legal(info)))


class GreedyCheapestPolicy:
    """Books the cheapest thing available, never asks. The 'always book cheapest' exploit,
    implemented so we can measure whether the environment actually defeats it."""

    name = "greedy_cheapest"

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        mask = info["action_mask"]
        plan = [
            encode(Verb.SELECT_FLIGHT, 0),
            encode(Verb.SELECT_HOTEL, 0),
            encode(Verb.PROPOSE),
            encode(Verb.BOOK),
            encode(Verb.FINALIZE),
            encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(FlightPreset.CHEAPEST)),
            encode(Verb.SEARCH_HOTELS, 0 * 3 + list(HotelSort).index(HotelSort.CHEAPEST)),
        ]
        for a in plan:
            if mask[a]:
                return a
        return int(_legal(info)[0])


class GreedyQualityPolicy:
    """Books the best-rated thing available. The other named exploit."""

    name = "greedy_quality"

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env) -> int:
        mask = info["action_mask"]
        plan = [
            encode(Verb.SELECT_FLIGHT, 0),
            encode(Verb.SELECT_HOTEL, 0),
            encode(Verb.PROPOSE),
            encode(Verb.BOOK),
            encode(Verb.FINALIZE),
            encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(FlightPreset.MOST_RELIABLE)),
            encode(Verb.SEARCH_HOTELS, 0 * 3 + list(HotelSort).index(HotelSort.BEST_RATED)),
        ]
        for a in plan:
            if mask[a]:
                return a
        return int(_legal(info)[0])


class HeuristicPolicy:
    """What a competent human travel agent does: interview, then search against what you
    learned, then book, then confirm.

    If this does not clearly beat `greedy_*` and `random`, the environment does not reward
    the behaviour it claims to reward, and that is a defect in the environment rather than
    in the policy. This policy is the falsifier.
    """

    name = "heuristic"

    def __init__(self, ask_budget: int = 5) -> None:
        self.ask_budget = ask_budget
        self.asked: list[int] = []

    def reset(self) -> None:
        self.asked = []

    def act(self, obs: dict, info: dict, env) -> int:
        mask = info["action_mask"]

        # 1. Interview first — the hidden weights are the whole game.
        if len(self.asked) < self.ask_budget:
            confidence = obs["belief"][:, 1]
            for axis_i in np.argsort(confidence):
                a = encode(Verb.ASK, int(axis_i))
                if mask[a] and int(axis_i) not in self.asked:
                    self.asked.append(int(axis_i))
                    return a

        belief = obs["belief"]
        cost_w = belief[list(PreferenceAxis).index(PreferenceAxis.COST), 0]
        rel_w = belief[list(PreferenceAxis).index(PreferenceAxis.RELIABILITY), 0]
        comfort_w = belief[list(PreferenceAxis).index(PreferenceAxis.COMFORT), 0]

        # 2. Search with a preset that matches what the client actually said.
        if not env.unwrapped.flight_slate:
            preset = (
                FlightPreset.MOST_RELIABLE if rel_w > 0.6
                else FlightPreset.CHEAPEST if cost_w > 0.5
                else FlightPreset.FASTEST
            )
            a = encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(preset))
            if mask[a]:
                return a

        need_flight = (
            env.unwrapped.itinerary.outbound is None
            or env.unwrapped.itinerary.inbound is None
        )
        if need_flight:
            a = _first_legal(info, *(encode(Verb.SELECT_FLIGHT, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            # Nothing affordable on this slate — try a cheaper filter before giving up.
            for preset in (FlightPreset.CHEAPEST, FlightPreset.FASTEST):
                a = encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(preset))
                if mask[a]:
                    return a

        # Hotels. If the current slate has nothing we can take, re-search cheaper rather
        # than standing there — which is what a competent agent does.
        hotel_pick = _first_legal(
            info, *(encode(Verb.SELECT_HOTEL, i) for i in range(SLATE_K))
        )
        if env.unwrapped.itinerary.nights_covered < env.unwrapped.nights:
            if hotel_pick is not None:
                return hotel_pick
            for sort in (
                HotelSort.BEST_RATED if comfort_w > 0.55 else HotelSort.CHEAPEST,
                HotelSort.CHEAPEST,
                HotelSort.MOST_CENTRAL,
            ):
                for city_i in range(3):
                    a = encode(Verb.SEARCH_HOTELS, city_i * 3 + list(HotelSort).index(sort))
                    if mask[a]:
                        return a

        # 3. Confirm with the client, then commit, then close out. Order matters and
        # each step happens once — proposing twice is not progress.
        for verb in (Verb.PROPOSE, Verb.BOOK, Verb.FINALIZE):
            a = encode(verb)
            if mask[a]:
                return a

        return int(_legal(info)[0])


ALL_POLICIES: dict[str, type] = {
    "null": NullPolicy,
    "random": RandomPolicy,
    "greedy_cheapest": GreedyCheapestPolicy,
    "greedy_quality": GreedyQualityPolicy,
    "heuristic": HeuristicPolicy,
}
