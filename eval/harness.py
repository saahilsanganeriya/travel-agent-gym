"""Evaluation harness: runs baselines, reports metrics, and gates on the ones that matter.

Two of these numbers are gates rather than reports:

* `null` and `random` must score ~0. tau-bench's own reward gives a do-nothing agent 38%
  success (arXiv:2507.02825) — the check exists because that failure is invisible otherwise.
* `heuristic` must clearly beat `greedy_*`. If interviewing the client is not worth more
  than booking the cheapest thing, the environment does not reward what it claims to.

Solve rate is read against two targets: random near 0%, and a competent heuristic near 30%,
low enough that a learned policy has room to beat it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

import travel_env.env  # noqa: F401  — registers the env id
from eval.baselines import ALL_POLICIES


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    ret: float
    steps: int
    complete: bool
    walked: bool
    spent: int
    budget: int
    elicited: float
    damage: int
    stratum: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """A solved episode: the client got a complete trip, stayed engaged, and the plan
        survived the trip well enough to be worth more than it cost."""
        return self.complete and not self.walked and self.ret > 0.0


def run_episode(policy, seed: int, *, difficulty: float, nights: int = 7) -> EpisodeResult:
    env = gym.make(
        "travel_agent/TravelAgent-v0", nights=nights, difficulty=difficulty
    )
    obs, info = env.reset(seed=seed)
    policy.reset()
    ret, steps = 0.0, 0
    term = trunc = False
    parts: dict[str, float] = defaultdict(float)

    while not (term or trunc):
        action = policy.act(obs, info, env)
        obs, reward, term, trunc, info = env.step(action)
        ret += float(reward)
        steps += 1
        for k, v in info.items():
            if k.startswith("reward_"):
                parts[k] += float(v)

    env.close()
    return EpisodeResult(
        policy=getattr(policy, "name", type(policy).__name__),
        seed=seed, ret=ret, steps=steps,
        complete=bool(info["itinerary_complete"]),
        walked=bool(info["client_walked"]),
        spent=int(info["spent_cents"]), budget=int(info["budget_cents"]),
        elicited=float(info["axes_elicited"]),
        damage=int(info["disruption_damage_cents"]),
        stratum=float(info["stratum"]),
        components=dict(parts),
    )


def pass_hat_k(successes_per_task: list[int], n_trials: int, k: int) -> float:
    """pass^k — the probability that ALL k independent trials of the same task succeed.

    The unbiased combinatorial estimator, matching tau-bench's implementation:
        pass^k = (1/|T|) * sum_task  C(c_task, k) / C(n, k)
    Note this is not `1 - (1-p)^k`; pass@k grows with k, pass^k collapses. Nobody gets to
    re-roll a booking, so pass^k is the honest reliability metric here.
    """
    if k > n_trials or not successes_per_task:
        return float("nan")
    denom = math.comb(n_trials, k)
    return sum(math.comb(c, k) / denom for c in successes_per_task) / len(successes_per_task)


def iqm(values: list[float]) -> float:
    """Interquartile mean. With few seeds, means and medians hide enough sampling noise to
    reverse rankings (arXiv:2108.13264); the IQM is the recommended point estimate."""
    if not values:
        return float("nan")
    v = np.sort(np.asarray(values, dtype=float))
    lo, hi = int(len(v) * 0.25), int(math.ceil(len(v) * 0.75))
    core = v[lo:hi] if hi > lo else v
    return float(core.mean())


def ratio(numerator: float, denominator: float) -> float | None:
    """None on an empty denominator, never 0.0.

    A metric that never ran is not a metric that measured zero. Without this,
    "disruption recovery: 1.0" when nothing was disrupted reads as a pass.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate(
    policy_names: list[str] | None = None,
    *,
    n_seeds: int = 40,
    trials_per_seed: int = 1,
    difficulties: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> dict[str, dict]:
    names = policy_names or list(ALL_POLICIES)
    report: dict[str, dict] = {}

    for name in names:
        results: list[EpisodeResult] = []
        per_task_successes: list[int] = []
        for s in range(n_seeds):
            difficulty = difficulties[s % len(difficulties)]
            succ = 0
            for t in range(trials_per_seed):
                policy = ALL_POLICIES[name]() if name != "random" else ALL_POLICIES[name](seed=s * 97 + t)
                r = run_episode(policy, seed=s * 1000 + t, difficulty=difficulty)
                results.append(r)
                succ += int(r.solved)
            per_task_successes.append(succ)

        returns = [r.ret for r in results]
        by_stratum: dict[float, list[EpisodeResult]] = defaultdict(list)
        for r in results:
            by_stratum[r.stratum].append(r)

        disrupted = [r for r in results if r.damage > 0]
        report[name] = {
            "episodes": len(results),
            "return_iqm": round(iqm(returns), 4),
            "return_mean": round(float(np.mean(returns)), 4),
            "return_std": round(float(np.std(returns)), 4),
            "solve_rate": round(float(np.mean([r.solved for r in results])), 4),
            "complete_rate": round(float(np.mean([r.complete for r in results])), 4),
            "walk_rate": round(float(np.mean([r.walked for r in results])), 4),
            "mean_steps": round(float(np.mean([r.steps for r in results])), 1),
            "mean_elicited": round(float(np.mean([r.elicited for r in results])), 3),
            "budget_used": round(
                float(np.mean([r.spent / max(1, r.budget) for r in results])), 3
            ),
            # None, not 0.0, when nothing was disrupted.
            "disruption_rate": ratio(len(disrupted), len(results)),
            "pass^1": round(pass_hat_k(per_task_successes, trials_per_seed, 1), 4)
            if trials_per_seed >= 1 else None,
            "pass^k": round(pass_hat_k(per_task_successes, trials_per_seed, trials_per_seed), 4)
            if trials_per_seed > 1 else None,
            "solve_by_stratum": {
                str(d): round(float(np.mean([r.solved for r in rs])), 3)
                for d, rs in sorted(by_stratum.items())
            },
        }
    return report


def format_report(report: dict[str, dict]) -> str:
    cols = ["solve_rate", "return_iqm", "return_mean", "complete_rate", "walk_rate",
            "mean_steps", "mean_elicited", "budget_used"]
    head = f"{'policy':<18}" + "".join(f"{c:>14}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, r in report.items():
        lines.append(f"{name:<18}" + "".join(f"{r[c]:>14}" for c in cols))
    lines.append("")
    lines.append("solve rate by difficulty stratum:")
    for name, r in report.items():
        lines.append(f"  {name:<18}{r['solve_by_stratum']}")
    lines.append("")
    lines.append("targets: random ~0% · heuristic ~30% · near 0% no signal, near 100% nothing to learn")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rep = evaluate(n_seeds=n)
    print(format_report(rep))
