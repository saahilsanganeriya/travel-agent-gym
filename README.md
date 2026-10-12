# travel-agent-gym

A client writes in: "family of four, Japan in spring, around $4k." They do not mention that one
of them uses a wheelchair, or that food matters to them more than anything else. A good travel
agent finds that out before booking. A bad one books the cheapest flight and the best-rated
hotel, and the trip falls apart in week two.

This repository is a reinforcement-learning environment built around that job. An agent
interviews a client whose preferences are hidden, searches inventory, builds an itinerary that
has to fit a real clock, books it against a budget, and then the trip actually happens and a
disruption engine fires. The reward is the trip the client actually got, not the plan.

It is a [Gymnasium](https://gymnasium.farama.org/) environment, so a standard RL algorithm
(MaskablePPO here) can train on it, and the same action set is rendered as tool calls so an
LLM agent can play it too.

**Status, plainly:** the first world (v1) turned out too easy. The second world (v2) fixed the
part that mattered and then overshot: today it is too hard for the reasons explained in
[Where v2 overshot](#where-v2-overshot). Both worlds run. The 98 tests cover v1; v2 has its
baselines and a world-coherence check. The numbers below are re-measured from this code.

## What happens in one episode

```
 Act I: plan and book                                   Act II: the trip
 ─────────────────────────────────────────────────      ──────────────────────────────
 client's opening line (reveals 1-3 of 8 preferences)
        │
        ▼
 INTERVIEW ──► ask about preferences and the party      disruption engine fires, but
        │      (patience drops every turn)              only on what was booked:
        ▼
 SEARCH ─────► slates of up to 8 options                  flight delayed
        │      (flights, rail, ferries, hotels,             └► missed connection
        │       activities; filters, not sort orders)           └► late arrival
        ▼                                                          └► hotel no-show
 ITINERARY ──► every segment must fit the clock:
        │      a.end + transfer(a, b) <= b.start        a cheap non-refundable fare with
        ▼                                               a 45-minute connection now costs
 BOOK ───────► integer-cent ledger; overspending        what it really costs
        │      is not penalised, it is impossible
        ▼                                                         │
 FINALIZE ──────────────────────────────────────────────►  terminal reward
```

| Part | What it does | Where |
|---|---|---|
| Interview | The client has eight hidden preference weights (cost, comfort, location, pacing, food, culture, novelty, reliability) and, in v2, a hidden party: each traveller's age, passport, access needs and diet. Patience runs down every turn, and the client can walk out. | `travel_env/personas.py`, `personas_v2.py`, `client_sim.py` |
| Search | Each search returns a fixed-width slate of candidates, and the agent selects by index. Presets are different candidate sets, not different sort orders. | `travel_env/actions.py`, `actions_v2.py` |
| Itinerary | One inequality on a timeline replaces a pile of special cases: over-packed days, moving cities without a booked leg, a tour that starts before the flight lands. | `travel_env/timeline.py`, `itinerary.py`, `clock.py` |
| Booking | An authoritative ledger in integer cents. Illegal actions (over budget, wrong city, museum closed that day) are removed by the action mask, never punished. | `travel_env/ledger.py`, `env_v2.py` |
| Disruption engine | Act II runs the booked trip. Delays, cancellations, overbooking, weather and strikes fire conditionally on what was booked, and a delay cascades through the timeline as far as it really reaches. | `travel_env/disruptions.py`, `env_v2.py` |

## Two worlds

| | v1 (`travel_agent/TravelAgent-v0`) | v2 (`travel_agent/TravelAgent-v2`, the default) |
|---|---|---|
| Geography | Tokyo, Kyoto, Osaka; 14 hand-set districts | Japan, Italy, Thailand; 13 cities, 43 districts; one or two regions per episode (two means an open-jaw route) |
| Transport | flights and one rail corridor | flight, rail, bus, ferry, car hire on local timetables; Thailand's ferries cancel in the monsoon |
| Lodging | hotels | 7 types (hotel, ryokan, hostel, apartment, capsule, resort, guesthouse) with real occupancy rules |
| Time | had three silent timezone bugs | UTC offsets from `zoneinfo`, arrivals derived from distance; `clock.assert_world_coherent` checks every instant, leg and duration (it passes on 50 generated worlds, but the tests do not run it yet) |
| Travel times | a haversine rule | 108 real Google Distance Matrix pairs, fetched once into `data/transfer_matrix.json`; nothing calls the network inside `step()` |
| Hard constraints | none | accessibility, capacity and documents: masked once the agent has asked, unmasked (and fatal at travel time) if it has not |
| Actions | `Discrete(63)` | `Discrete(89)` |

v1 is kept, unchanged, because its trained checkpoint (`models/v1_ppo/`) is the evidence that
the reward carries a learnable signal.

The synthetic world is correlated on purpose. Price tracks a hotel's true quality plus noise.
Displayed ratings are noisier when a hotel has few reviews, so sorting by rating picks lucky
small-sample hotels and overpays (measured over 200 worlds in `data/world_stats.md`). The
cheap, well-located seats and rooms are often the ones already gone. A sample world and ten sample clients,
hidden weights included, are in `data/`.

## Reward

```
per turn:   Φ(s_t) − Φ(s_{t−1})  −  friction
terminal:   realised client utility + budget efficiency − disruption damage − patience forfeit
```

The shaping term is a potential difference, with Φ = 0 at termination and at truncation, so it
cannot change which policy is optimal. There is no LLM anywhere in the reward: it is integer
arithmetic over the ledger, and the same seed gives the same reward. Every component is
reported separately in `info`.

The two obvious exploits score zero without a penalty term written against either:

| Exploit | Why it loses |
|---|---|
| Always book the cheapest flight | Act II makes cheap and fragile lose in expectation |
| Always pick the highest-rated hotel | Ratings from few reviews are noisy, and price tracks real quality |
| Book without asking | The weights are hidden, so an unasked booking is a bet on a prior |
| Ask forever | Patience runs out and the client walks |

## Baselines

Six scripted policies. `null` does nothing. `random` picks uniformly among legal actions.
`greedy_cheapest` and `greedy_quality` are the two exploits. `heuristic` interviews the client
on preferences, then searches and books. `expert` (v2 only) is the heuristic plus four
questions about the party's hard constraints.

**v1**, 24 seeds, from [`logs/published/baselines_v1.txt`](logs/published/baselines_v1.txt)
(`python -m eval.harness 24`):

```
policy             solve   return_iqm   complete   walk_rate   steps
null               0.00      -1.0097        0.00       0.67     50.5
random             0.125     -1.1405        0.25       0.83     30.6
greedy_cheapest    0.00      -1.3002        0.00       0.96     26.4
greedy_quality     0.00      -1.3115        0.00       0.96     26.4
heuristic          0.75      +0.1100        0.75       0.25     47.0
```

The committed MaskablePPO checkpoint solves 88% of 24 episodes (seeds 5000–5023) with mean
return +0.126 ([`logs/published/v1_ppo_checkpoint_eval.txt`](logs/published/v1_ppo_checkpoint_eval.txt)).
Its training run went from −1.02 to +0.133 mean episode reward over 516k steps, with a best
evaluation of 0.36 ± 0.11 at 400k steps (`logs/published/v1_ppo_train.log`, lines 6, 3736 and
2901).

That is the problem with v1. Random solved one episode in eight and completed one in four,
so the task could be finished by accident. A hand-written heuristic solved three in four, and
a learned policy nearly nine in ten. My targets were random near zero and the heuristic near
30%: low enough that a learned policy has something to find, high enough that success is
not so rare a learner never sees it. v1 missed both.

**v2**, 60 seeds, from [`logs/published/baselines_v2.txt`](logs/published/baselines_v2.txt)
(`python eval/run_v2.py 60`):

```
policy             solve   ret_iqm   complete   walk   steps   prefs   hard_q   cfail
null                0.00   -0.3600       0.00   0.00    90.0    0.33     0.00    0.00
random              0.00   -0.3782       0.00   0.13    89.5    0.91     0.99    0.00
greedy_cheapest     0.00   -1.7207       0.00   0.63    82.1    0.25     0.00    0.00
greedy_quality      0.00   -1.7230       0.00   0.63    82.1    0.25     0.00    0.00
heuristic           0.05   -1.3175       0.08   0.47    83.7    0.57     0.00    0.02
expert              0.02   -1.1827       0.03   0.42    85.0    0.57     1.00    0.00
```

`prefs` is the share of preference axes the policy found out; `hard_q` is the share of
hard-constraint classes it asked about; `cfail` is the share of trips that broke a constraint
at travel time.

## Where v2 overshot

```
                 v1      v2      target
random          12.5%    0%      ~0%        fixed: no more solving by accident
heuristic       75%      5%      ~30%       overshot, and for the wrong reason
```

Random going to zero is the real win. A coherent multi-region route with a correct
night-by-night allocation is not something a uniform policy stumbles into.

The heuristic at 5% is a failure, and not because the task became appropriately hard. Its
`complete` is 0.08: it is failing to finish trips, not making bad choices. Steps sit at 82–90
against a 90-turn cap. Two causes were found and fixed during development (no budget reserve
for the return leg, so one outbound flight could eat 97% of the budget, and a heuristic that
repeated the same empty search). Completion is still low. Raising the turn cap from 90 to 150
made it worse, because patience then ran out first (the measurement is recorded next to
`MAX_TURNS` in `travel_env/env_v2.py`).

Two consequences follow, and both are open:

- **Interviewing does not yet pay.** `expert` asks about every hard constraint and scores
  inside noise of `heuristic` (2% against 5%), because neither finishes. v2's central idea is
  built, and this measurement does not yet show it working.
- **Doing nothing is the best policy.** `null` scores −0.360 and `heuristic` −1.318: a policy
  that books a trip absorbs disruption damage and constraint failures, and one that stalls
  pays only the patience forfeit. PPO found this. Its training reward converged to −0.00005
  with episodes running to 89.9 of 90 turns, so it never books
  (`logs/published/v2_ppo_train.log`, lines 4326–4327; the deterministic evaluation at 500k
  steps is −0.68 ± 0.65, line 4304). The curve goes up. What it climbs toward is inaction.

The fix order matters. Completion comes first: at 8% completion no reward shape saves this.
Then the terminal needs rebalancing, because a `PATIENCE_FORFEIT` of 1.0 is cheap next to a
`DISRUPTION_SCALE` of 0.9 applied to damage that can reach 2.0 (`travel_env/rewards.py`).
Walking away should be the worst outcome, not a floor to retreat to.

**A note on the history.** Commit messages and earlier versions of this README record a v1
heuristic at 96% and a v2 heuristic at 7%. Re-running the committed code, on any seed count I
tried and with the original library versions, gives 67–83% for v1 and 5% for v2; the "random
solves 25%" figure came from the test suite's 12-seed sample. The tables above are the re-run
(Linux x86-64, Python 3.12, NumPy 2.5, Gymnasium 1.4). The conclusions do not change.

## The LLM path

`travel_env/llm_agent.py` turns the observation and action mask into tool schemas, so an LLM
agent is only ever offered legal actions. `travel_env/client_sim.py` plays the client with an
LLM whose context holds the persona as prose and the transcript, never the world, the ledger
or the weights as numbers. Agent and client are independent switches, so all four
combinations run, and `--dry-run` makes no network calls
([`logs/published/llm_eval_dryrun.json`](logs/published/llm_eval_dryrun.json)).

One short live run (two episodes, `gpt-5-mini` as the agent, `gpt-4.1-mini` as the client)
found 91% of preference axes and asked about 100% of hard-constraint classes, against the
heuristic's 57% and 0%, and still completed no trips. Its raw output was not saved; the
numbers are in the commit "Fix the LLM eval call-budget bug; record the first working live
run". An LLM doing the right thing and hitting the same wall points at the environment, not
the agents.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[train,dev]"

PYTHONPATH=. .venv/bin/python scripts/smoke.py            # one rendered episode
PYTHONPATH=. .venv/bin/python -m eval.harness 24          # v1 baseline table
PYTHONPATH=. .venv/bin/python eval/run_v2.py 60           # v2 baseline table
PYTHONPATH=. .venv/bin/python -m pytest tests             # 98 invariant tests (v1)
PYTHONPATH=. .venv/bin/python scripts/play_trained.py     # watch the v1 checkpoint play
PYTHONPATH=. .venv/bin/python scripts/train.py --steps 400000 --n-envs 10
```

Everything above runs offline with no keys. Two optional paths need one: the LLM agent and
client (`LLM_API_KEY` and `LLM_BASE_URL` for any OpenAI-compatible gateway, or
`OPENROUTER_API_KEY`), and the one-off rebuild of the travel-time matrix
(`GOOGLE_MAPS_API_KEY`). Copy `.env.example` to `.env` to set them.

## Layout

| Path | Contents |
|---|---|
| `travel_env/` | the environments, world generators, contracts, ledger, timeline, reward, disruption engine, LLM adapters |
| `eval/` | baselines, the evaluation harness (solve rate, IQM return, pass^k), the LLM evaluation |
| `scripts/` | smoke test, training, checkpoint playback, data dumps, the travel-time fetcher |
| `tests/` | 98 invariant tests for v1 that drive the real environment, no mocks |
| `data/` | the travel-time matrix, a sample world, sample clients, measured world statistics |
| `logs/published/`, `models/v1_ppo/` | training logs, baseline outputs, the v1 checkpoint |

## Licence

MIT. See [LICENSE](LICENSE).
