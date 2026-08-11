"""LLM-agent vs LLM-client evaluation on the v2 environment, over a frozen instance set.

Three things this harness insists on:

* **A frozen seed list.** The eval set is `FROZEN_SEEDS[:episodes]`, always the same worlds
  and the same clients. Numbers from two runs are only comparable if the instances are, and
  "we sampled 20 random seeds" is how benchmark deltas turn into sampling noise.
* **pass^k, not pass@k.** `eval.harness.pass_hat_k` is the unbiased combinatorial estimator
  `mean_task C(c,k)/C(n,k)` — the probability that *all* k attempts at the same task succeed.
  Nobody re-rolls a booking, so reliability is the honest metric and it collapses with k
  where pass@k grows.
* **The client model is a reported hyperparameter.** Swapping the user simulator moves
  tau-bench scores by up to ~9 points (arXiv:2506.07982), which is larger than most agent
  deltas anyone reports. A solve rate without the client model beside it is not a result, so
  it is written into every JSON and printed in every table.

The 2x2 of `--agent {llm,heuristic,random}` x `--client {llm,scripted}` exists so the client
simulator's contribution is measurable rather than assumed. `--dry-run` is the offline cell:
scripted client, heuristic agent, zero network calls, zero dollars.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict

import travel_env.env_v2  # noqa: F401 — registers travel_agent/TravelAgent-v2
from eval.harness import iqm, pass_hat_k
from travel_env.actions_v2 import (
    SLATE_K,
    FlightPreset,
    HotelSort,
    LodgingFilter,
    Verb,
    encode,
    encode_hotel_search,
)
from travel_env.client_sim import LLMClientSim, ScriptedClientSim, question_for_action
from travel_env.constraints import ASKABLE
from travel_env.contracts import ActivityCategory, PreferenceAxis
from travel_env.llm import LLMClient
from travel_env.llm_agent import LLMAgent

# The eval instance set. Frozen on purpose — see the module docstring.
FROZEN_SEEDS: tuple[int, ...] = tuple(range(100))
DIFFICULTY_CYCLE: tuple[float, ...] = (0.2, 0.5, 0.8)

DEFAULT_AGENT_MODEL = "openai/gpt-5-mini"
DEFAULT_CLIENT_MODEL = "openai/gpt-4.1-mini"
"""Different families for agent and client, deliberately: with one model on both sides the
agent's prior over 'what would I say next' is a near-perfect model of the client, and the
elicitation problem the environment is built around quietly disappears."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class EpisodeRecord(FrozenModel):
    seed: int
    difficulty: float
    ret: float
    steps: int
    complete: bool
    walked: bool
    solved: bool
    constraint_failures: int
    spent_cents: int
    budget_cents: int
    axes_elicited: float
    constraints_elicited: float
    client_turns: int


# ---- policies ---------------------------------------------------------------------------


class RandomV2Policy:
    """Uniform over legal actions. The reward-scale reference point for the v2 space."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None: ...

    def act(self, obs: dict, info: dict, env: Any) -> int:
        return int(self.rng.choice(np.flatnonzero(info["action_mask"])))


class HeuristicV2Policy:
    """What a competent human travel agent does, in the v2 action space.

    Books a single-city route on purpose: with one city there are no intercity hops to get
    wrong, so this measures 'interview then book carefully' without also measuring multi-leg
    scheduling. It is the falsifier for the reward, not an attempt at the ceiling.
    """

    name = "heuristic"

    def __init__(self, *, ask_axes: int = 5, want_activities: int = 3) -> None:
        self.ask_axes = ask_axes
        self.want_activities = want_activities
        self.asked_axes: set[int] = set()
        self.empty_flight_searches = 0
        self.empty_hotel_searches = 0

    def reset(self) -> None:
        self.asked_axes = set()
        self.empty_flight_searches = 0
        self.empty_hotel_searches = 0

    def act(self, obs: dict, info: dict, env: Any) -> int:
        mask = info["action_mask"]
        core = env.unwrapped

        def first_legal(*candidates: int) -> int | None:
            return next((c for c in candidates if mask[c]), None)

        # 1. Hard constraints first: asking is what switches on filtering in every later search.
        for i, klass in enumerate(ASKABLE):
            if klass not in core.known:
                a = encode(Verb.ASK_CONSTRAINTS, i)
                if mask[a]:
                    return a

        # 2. Then the preference axes we know least about.
        if len(self.asked_axes) < self.ask_axes:
            for axis_i in np.argsort(obs["belief"][:, 1]):
                a = encode(Verb.ASK, int(axis_i))
                if mask[a] and int(axis_i) not in self.asked_axes:
                    self.asked_axes.add(int(axis_i))
                    return a

        # 3. A route that uses every night, in one city that flights actually reach.
        # Which cities have an airport is public information in any travel search, and a
        # route anchored on a city with no gateway is unbookable however good the rest is.
        allocated = sum(n for _, n in core.route)
        if allocated < core.nights:
            if not core.destination_slate:
                a = encode(Verb.SEARCH_DESTINATIONS)
                if mask[a]:
                    return a
            gateways = {airport.city for airport in core.world.airports}
            preferred = [
                encode(Verb.ADD_CITY, i)
                for i, city in enumerate(core.destination_slate[:SLATE_K])
                if city in gateways
            ]
            a = first_legal(*preferred) or first_legal(
                *(encode(Verb.ADD_CITY, i) for i in range(SLATE_K))
            )
            if a is not None:
                return a

        belief = obs["belief"]
        cost_w = float(belief[list(PreferenceAxis).index(PreferenceAxis.COST), 0])
        comfort_w = float(belief[list(PreferenceAxis).index(PreferenceAxis.COMFORT), 0])
        reliability_w = float(belief[list(PreferenceAxis).index(PreferenceAxis.RELIABILITY), 0])

        # 4. Flights out and back, filtered by what the client actually said.
        if core.timeline.outbound is None or core.timeline.inbound is None:
            a = first_legal(*(encode(Verb.SELECT_FLIGHT, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            preset = (
                FlightPreset.MOST_RELIABLE
                if reliability_w > 0.6
                else FlightPreset.CHEAPEST
                if cost_w > 0.5
                else FlightPreset.FASTEST
            )
            a = first_legal(
                encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(preset)),
                encode(Verb.SEARCH_FLIGHTS, list(FlightPreset).index(FlightPreset.CHEAPEST)),
            )
            if a is not None:
                return a

        # 5. Ground legs, if the route happens to have a hop.
        a = first_legal(*(encode(Verb.SELECT_GROUND, i) for i in range(SLATE_K)))
        if a is not None:
            return a

        # 6. Lodging for every city on the route.
        if core._pending_city() is not None:
            a = first_legal(*(encode(Verb.SELECT_HOTEL, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            sort = HotelSort.BEST_RATED if comfort_w > 0.55 else HotelSort.CHEAPEST
            lodging = LodgingFilter.WHOLE_UNIT if core.needs.size >= 3 else LodgingFilter.ANY
            a = first_legal(
                encode_hotel_search(sort, lodging),
                encode_hotel_search(HotelSort.CHEAPEST, LodgingFilter.ANY),
                encode_hotel_search(HotelSort.CHEAPEST, LodgingFilter.BUDGET),
            )
            if a is not None:
                return a

        # 7. A few activities on the client's strongest axis.
        if len(core.timeline.activities) < self.want_activities:
            a = first_legal(*(encode(Verb.SELECT_ACTIVITY, i) for i in range(SLATE_K)))
            if a is not None:
                return a
            a = first_legal(
                *(
                    encode(Verb.SEARCH_ACTIVITIES, list(ActivityCategory).index(c))
                    for c in _preferred_categories(belief)
                )
            )
            if a is not None:
                return a

        # 8. Confirm, commit, close — in that order, each exactly once.
        a = first_legal(encode(Verb.PROPOSE), encode(Verb.BOOK), encode(Verb.FINALIZE))
        if a is not None:
            return a

        a = first_legal(*(encode(Verb.REBOOK, i) for i in range(SLATE_K)))
        if a is not None:
            return a
        return int(np.flatnonzero(mask)[0])


def _preferred_categories(belief: np.ndarray) -> list[ActivityCategory]:
    """Activity categories ordered by the axis the client cares most about."""
    axis_to_category = {
        PreferenceAxis.FOOD: ActivityCategory.FOOD,
        PreferenceAxis.CULTURE: ActivityCategory.HISTORY,
        PreferenceAxis.NOVELTY: ActivityCategory.NATURE,
    }
    ranked = sorted(
        axis_to_category,
        key=lambda axis: -float(belief[list(PreferenceAxis).index(axis), 0]),
    )
    return [axis_to_category[a] for a in ranked] + [
        ActivityCategory.MUSEUM,
        ActivityCategory.FAMILY,
        ActivityCategory.SHOPPING,
        ActivityCategory.NIGHTLIFE,
    ]


# ---- the loop ---------------------------------------------------------------------------


def run_episode(policy: Any, *, seed: int, difficulty: float, client_kind: str, client_llm) -> EpisodeRecord:
    env = gym.make("travel_agent/TravelAgent-v2", difficulty=difficulty)
    obs, info = env.reset(seed=seed)
    core = env.unwrapped
    policy.reset()

    if client_kind == "llm":
        client_llm.begin_episode()
        client = LLMClientSim(core.persona, core.transcript[0][1], client_llm)
    else:
        client = ScriptedClientSim(core.persona)

    ret, steps, client_turns = 0.0, 0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs, info, env)
        # The utterance the client hears: whatever the LLM agent wrote, or the canonical
        # rendering of the ASK action for policies that do not write.
        spoken = getattr(policy, "last_utterance", None) or question_for_action(action)
        legal = bool(info["action_mask"][action])
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        steps += 1
        if spoken and legal:
            core.transcript.append(("agent", spoken))
            core.transcript.append(("client", client.reply(spoken)))
            client_turns += 1

    env.close()
    complete = bool(info["itinerary_complete"])
    walked = bool(info["client_walked"])
    return EpisodeRecord(
        seed=seed,
        difficulty=difficulty,
        ret=ret,
        steps=steps,
        complete=complete,
        walked=walked,
        solved=complete and not walked and ret > 0.0,
        constraint_failures=int(info["constraint_failures"]),
        spent_cents=int(info["spent_cents"]),
        budget_cents=int(info["budget_cents"]),
        axes_elicited=float(info["axes_elicited"]),
        constraints_elicited=float(info["constraints_elicited"]),
        client_turns=client_turns,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    seeds = FROZEN_SEEDS[: args.episodes]

    agent_llm = client_llm = None
    if args.agent == "llm":
        agent_llm = LLMClient(
            args.agent_model,
            base_url=args.base_url,
            rate_per_min=args.rate_per_min,
            max_calls_per_episode=args.max_agent_calls,
        )
    if args.client == "llm":
        client_llm = LLMClient(
            args.client_model,
            base_url=args.base_url,
            rate_per_min=args.rate_per_min,
            max_calls_per_episode=args.max_client_calls,
        )

    policy = _build_policy(args, agent_llm)
    records: list[EpisodeRecord] = []
    successes_per_task: list[int] = []

    for i, seed in enumerate(seeds):
        difficulty = DIFFICULTY_CYCLE[i % len(DIFFICULTY_CYCLE)]
        wins = 0
        for trial in range(args.trials):
            if args.agent == "random":
                policy = RandomV2Policy(seed=seed * 97 + trial)
            record = run_episode(
                policy,
                seed=seed * 1000 + trial,
                difficulty=difficulty,
                client_kind=args.client,
                client_llm=client_llm,
            )
            records.append(record)
            wins += int(record.solved)
            print(
                f"  seed {seed:>3} d={difficulty} trial {trial}: "
                f"return {record.ret:+7.3f} steps {record.steps:>3} "
                f"complete={record.complete} walked={record.walked} "
                f"solved={record.solved} cfail={record.constraint_failures}",
                flush=True,
            )
        successes_per_task.append(wins)

    return _report(args, records, successes_per_task, agent_llm, client_llm, policy)


def _build_policy(args: argparse.Namespace, agent_llm: LLMClient | None) -> Any:
    match args.agent:
        case "llm":
            assert agent_llm is not None
            return LLMAgent(agent_llm)
        case "heuristic":
            return HeuristicV2Policy()
        case _:
            return RandomV2Policy()


def _report(
    args: argparse.Namespace,
    records: list[EpisodeRecord],
    successes_per_task: list[int],
    agent_llm: LLMClient | None,
    client_llm: LLMClient | None,
    policy: Any,
) -> dict[str, Any]:
    returns = [r.ret for r in records]
    by_stratum: dict[float, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        by_stratum[r.difficulty].append(r)

    tokens = {
        "agent_prompt": agent_llm.prompt_tokens if agent_llm else 0,
        "agent_completion": agent_llm.completion_tokens if agent_llm else 0,
        "client_prompt": client_llm.prompt_tokens if client_llm else 0,
        "client_completion": client_llm.completion_tokens if client_llm else 0,
    }
    cost = (agent_llm.cost_usd if agent_llm else 0.0) + (client_llm.cost_usd if client_llm else 0.0)
    steps = sum(r.steps for r in records)
    parse_failures = getattr(policy, "parse_failures", 0)

    return {
        "run_utc": datetime.now(tz=UTC).isoformat(),
        # Hyperparameters, not decoration: the client model is part of the result.
        "agent": args.agent,
        "client": args.client,
        "agent_model": args.agent_model if args.agent == "llm" else None,
        "client_model": args.client_model if args.client == "llm" else None,
        "endpoint": (agent_llm or client_llm).base_url if (agent_llm or client_llm) else None,
        "episodes": len(records),
        "seeds": list(FROZEN_SEEDS[: args.episodes]),
        "trials_per_seed": args.trials,
        "solve_rate": round(float(np.mean([r.solved for r in records])), 4),
        "complete_rate": round(float(np.mean([r.complete for r in records])), 4),
        "walk_rate": round(float(np.mean([r.walked for r in records])), 4),
        "return_mean": round(float(np.mean(returns)), 4),
        "return_iqm": round(iqm(returns), 4),
        "return_std": round(float(np.std(returns)), 4),
        "mean_steps": round(float(np.mean([r.steps for r in records])), 1),
        "constraint_failure_rate": round(
            float(np.mean([r.constraint_failures > 0 for r in records])), 4
        ),
        "mean_constraint_failures": round(
            float(np.mean([r.constraint_failures for r in records])), 3
        ),
        "budget_used": round(
            float(np.mean([r.spent_cents / max(1, r.budget_cents) for r in records])), 3
        ),
        "axes_elicited": round(float(np.mean([r.axes_elicited for r in records])), 3),
        "constraints_elicited": round(float(np.mean([r.constraints_elicited for r in records])), 3),
        "client_turns_mean": round(float(np.mean([r.client_turns for r in records])), 1),
        "pass^1": round(pass_hat_k(successes_per_task, args.trials, 1), 4),
        "pass^k": (
            round(pass_hat_k(successes_per_task, args.trials, args.trials), 4)
            if args.trials > 1
            else None
        ),
        "k": args.trials,
        "solve_by_stratum": {
            str(d): round(float(np.mean([r.solved for r in rs])), 3)
            for d, rs in sorted(by_stratum.items())
        },
        "parse_failures": int(parse_failures),
        "parse_failure_rate": round(parse_failures / max(1, steps), 4),
        "llm_call_failures": int(getattr(policy, "call_failures", 0)),
        "tokens": tokens,
        "cost_usd": round(cost, 6),
        "cost_usd_per_episode": round(cost / max(1, len(records)), 6),
        "cost_source": "gateway-reported"
        if all(c.cost_is_reported for c in (agent_llm, client_llm) if c)
        else "local-price-table",
    }


def format_report(report: dict[str, Any]) -> str:
    keys = [
        "agent", "agent_model", "client", "client_model", "endpoint", "episodes",
        "solve_rate", "return_iqm", "return_mean", "complete_rate", "walk_rate",
        "mean_steps", "constraint_failure_rate", "axes_elicited", "constraints_elicited",
        "client_turns_mean", "pass^1", "pass^k", "solve_by_stratum",
        "parse_failure_rate", "llm_call_failures", "tokens", "cost_usd",
        "cost_usd_per_episode", "cost_source",
    ]
    width = max(len(k) for k in keys)
    lines = ["", "=" * 72]
    lines += [f"{k:<{width}}  {report[k]}" for k in keys]
    lines.append("=" * 72)
    lines.append(
        "client model is a hyperparameter: swapping the user simulator moves tau-bench "
        "results by up to ~9 points."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="how many frozen seeds to run")
    parser.add_argument("--trials", type=int, default=1, help="attempts per seed; k for pass^k")
    parser.add_argument("--agent", choices=("llm", "heuristic", "random"), default="heuristic")
    parser.add_argument("--client", choices=("llm", "scripted"), default="scripted")
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--client-model", default=DEFAULT_CLIENT_MODEL)
    parser.add_argument("--base-url", default=None, help="override; defaults to LLM_BASE_URL")
    parser.add_argument("--rate-per-min", type=float, default=120.0)
    parser.add_argument("--max-agent-calls", type=int, default=90)
    parser.add_argument("--max-client-calls", type=int, default=40)
    parser.add_argument("--out", default="logs/llm_eval.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scripted client + heuristic agent, zero network calls",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.agent, args.client = "heuristic", "scripted"

    report = evaluate(args)
    print(format_report(report))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
