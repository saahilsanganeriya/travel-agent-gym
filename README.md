# AI Travel Agent — an RL training environment

A Gymnasium environment in which an agent runs an end-to-end client engagement at a
mid-size travel agency: interview the client, search inventory, compose an itinerary, get it
approved, book it — and then **the trip actually happens**, and things go wrong.

```
policy            solve_rate  return_iqm  complete  walk_rate  steps  elicited  budget_used
null                    0.00     -1.0097      0.00       0.67   50.5     0.359        0.000
random                  0.04     -1.1373      0.21       0.83   33.2     0.523        0.064
greedy_cheapest         0.00     -1.3002      0.00       0.96   26.4     0.250        0.000
greedy_quality          0.00     -1.3123      0.00       0.96   26.4     0.250        0.000
heuristic               0.96     +0.1048      0.96       0.04   58.4     0.593        0.723
```

The two obvious exploits — *"always book the cheapest flight"* and *"always pick the
highest-rated hotel"* — score **zero**, and neither has a penalty term written against it.
They lose because the environment makes them bad bets. That is the thesis of this repository.

---

## How this was built

I (Saahil) set the direction; Claude Code did the implementation and the research sweeps.
Worth saying plainly, because the design decisions below were mine and they are the part
that matters:

- **I chose the shape of the world.** Three cities on the Tokyo–Kyoto–Osaka corridor, medium
  depth, rather than a wide shallow global inventory — because I wanted "fly or take the
  train" to be a real decision, and a flat distribution produces a flat agent.
- **I chose two acts.** Score the trip that actually happened, not the plan that was
  proposed. This was the call that makes robustness pay for itself.
- **I insisted the client be LLM-driven.** The first draft used a scripted client because
  PPO needs 10⁵ episodes. That was the training constraint dictating the design, and I
  rejected it: τ-bench, τ²-bench and τ³-bench all use LLM user simulators, and real clients
  do not talk like state machines.
- **I pushed hard on causality and realism,** repeatedly, and the honest gaps that remain
  are listed in "What I'd do next" rather than smoothed over.
- **I made the difficulty and data-generation calls** — hidden preference weights on a
  10-point grid, demographics correlated with behaviour, and a generator as the source of
  truth rather than a committed dataset.

Where the environment is thin, I say so below. I would rather be judged on an accurate
account than a flattering one.

---

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[train,dev]"

PYTHONPATH=. .venv/bin/python scripts/smoke.py       # one rendered episode
PYTHONPATH=. .venv/bin/python -m eval.harness 24     # the baseline table above
PYTHONPATH=. .venv/bin/python scripts/train.py --steps 600000 --n-envs 12
tensorboard --logdir logs
```

---

## 1. Why Gymnasium

A bespoke interface was an option. I chose Gymnasium, with a caveat worth
stating because it cuts against the choice.

**For:** `reset(seed, options) → (obs, info)` and `step(action) → (obs, reward, terminated,
truncated, info)` are the de-facto standard. Adopting them costs nothing and buys seeding
and reproducibility semantics, the wrapper stack (`TimeLimit`, `RecordEpisodeStatistics`),
vectorization, `check_env`, and instant legibility to a reader.

**Against, honestly:** every major LLM-agent environment framework — OpenEnv, GEM,
SkyRL-Gym, NVIDIA NeMo Gym — kept the `reset`/`step` vocabulary but **none of them subclass
`gymnasium.Env` or use `gymnasium.spaces`**. The universally discarded piece is `spaces`,
and the reason is concrete: `spaces.Text`'s default charset is 62 alphanumerics with no
space character, so `Text(50).contains("hello world")` is `False`. Composite spaces cannot
cleanly hold a chat transcript.

**Why I stayed anyway:** those frameworks target LLM agents *only*. I wanted this
environment to be trainable by a standard RL algorithm **and** drivable by an LLM. For that,
Gymnasium is right — and we sidestep the `spaces` problem entirely by never putting language
in a space. Language lives in `info` and in the adapter; the space carries numbers. Farama's
own docs concede `Env` is *"not a perfect reconstruction"* of an MDP, which is the licence to
layer an adapter on top rather than invent an interface.

The proof this was the right call is in `logs/` — a MaskablePPO policy trains on this
environment today, and the same masked action set renders as a tool schema for an LLM.

---

## 2. The world

Fourteen hand-set districts across Tokyo, Kyoto and Osaka, each with real coordinates and a
five-axis character vector (walkability, nightlife, dining, culture, quiet) plus a price
index. **Geography is authored and frozen across seeds; everything else is generated.** So
an agent can learn that Ginza is expensive and Arashiyama is quiet — real transferable
knowledge — but cannot memorise which hotel is cheap.

**Kyoto has no airport.** You fly to Tokyo or Osaka and take rail. That is a structural
constraint, not trivia: an agent that treats cities as interchangeable nodes produces
infeasible plans.

Travel time is computed, never stored:

```python
def transfer_minutes(a, b):
    km = haversine_km(a, b)
    if km < 1.2:  return max(5, round(km * 13))          # walking
    return round(8 + (km * 2.2) / 22.0 * 60)             # 8 min overhead, 2.2 detour, 22 km/h
```

This is the feasibility gate for scheduling: two activities cannot be booked closer together
than the journey between their districts.

### Data generation

`generate_world(seed)` runs a seven-stage pipeline off a single `np.random.default_rng`:
calendar/season → frozen geography → supply → depletion → persona → gate → disruption seeds.
Determinism is absolute — `generate_world(42)` is byte-identical every time.

Three modelling choices do the heavy lifting:

**Price correlates with quality, plus noise.** `price = base(quality) × district × season ×
lead_time × N(1, 0.13)`. Deterministic correlation makes quality a free win; zero
correlation makes it unlearnable. Noise makes it a judgement call.

**Rating confidence scales with sample size.** `rating_sd = clip(2.4/√review_count, .06,
1.5)`. A 9.8 from 11 reviews is weak evidence; an 8.6 from 2,400 is strong. Since price
tracks *stars* (true quality) rather than *displayed rating*, the lucky small-sample hotel
is cheap and looks excellent — so sorting by rating systematically overpays. **This is how
the "always pick the highest-rated hotel" exploit is defeated by the data distribution
rather than by a penalty.**

**Availability is skewed against the good options.** Seats deplete on a Poisson draw scaled
by fare flexibility, so the well-priced options are disproportionately gone — which is what
creates genuine search pressure instead of a lookup.

Fare brands (BASIC / STANDARD / FLEX) and rate plans (non-refundable / free-cancellation)
are the bridge between the two acts. At booking time they look like the same seat at
different prices. In act II they are not.

---

## 3. Reward

```
per turn:   Φ(s_t) − Φ(s_{t−1})  −  friction
terminal:   realized_client_utility + budget_efficiency − disruption_damage − patience_forfeit
```

Every component is emitted separately into `info`, so the shaping analysis needs no rerun.

**The client's preferences are hidden.** Eight axes on a 10-point scale; the opening request
reveals one to three of them. The reward is a dot product against the hidden weights, so an
agent that books without interviewing is betting on its prior — and that bet is low-value
without any penalty term saying so. This is what makes question-asking *instrumentally*
necessary rather than instructed.

**The shaping term is a potential difference, and Φ is zero at terminal *and* truncation.**
Ng, Harada & Russell (1999) prove `γΦ(s′) − Φ(s)` is necessary and sufficient for policy
invariance. Less well known: Grześ (AAMAS 2017) shows the trajectory picks up a `γ^N·Φ(s_N)`
term that depends on which terminal you reach and can flip the optimal policy — and a step
cap manufactures exactly that situation. Both conditions are enforced.

**No exploration bonus,** deliberately. Rewarding exploration tends to buy turns that look
exploratory and do nothing: questions asked for the bonus rather than for the answer.

**No LLM anywhere in the reward path.** Reward is integer arithmetic over a transaction
ledger. Same seed, same reward, every time.

### The exploit table

| Cheapest exploit | Structural defence |
|---|---|
| Always book the cheapest flight | Act II makes cheap+fragile lose in expectation; reliability is priced |
| Always pick the highest-rated hotel | Rating variance shrinks with review count; price tracks stars, not rating |
| Finish in one step, never ask | Hidden weights ⇒ an unelicited booking has low expected utility |
| Never book anything | Zero itinerary ⇒ zero utility; the potential telescopes to zero |
| Spend the whole budget to look thorough | Efficiency bonus is **gated** on hard needs met |
| Ask forever | Patience depletes; the client walks |
| Overspend | Not penalised — the action is **masked out** |

**Denominator audit.** Agents game ratios by shrinking the denominator, not growing the
numerator. This reward contains exactly one ratio — budget efficiency — and it is gated on
hard-needs-met, so booking less to spend less fails the gate and scores zero.

---

## 4. Gates, not penalties

Illegal actions are removed from the action space by `action_masks()`, never punished.
Nine of the environment's twenty-one invariants are enforced this way: you cannot book past
your budget, book a hotel in a city you are not in, schedule a museum on its closed day, or
book two activities closer together than the journey between them.

The evidence is unusually strong. Huang & Ontañón (arXiv:2006.14171) found that across every
penalty weight tried, the agent *"struggles to even find the very first reward"* on any map
≥10×10, while masking stayed scale-invariant — and the invalid-action policy gradient is
provably the gradient of the masked policy, with the gradient w.r.t. an invalid logit
exactly zero.

Both conventions are exposed: `action_masks()` for `MaskablePPO`, and `info["action_mask"]`
for Gymnasium-native consumers.

---

## 5. Evaluation

`eval/harness.py` runs five baselines and reports solve rate, IQM return, completion, walk
rate, elicitation, budget use, and `pass^k` — using the unbiased combinatorial estimator
`(1/|T|)·Σ C(c,k)/C(n,k)`, not `1−(1−p)^k`, because nobody gets to re-roll a booking.

Two numbers are **gates, not reports**:
- `null` and `random` must score ≈ 0. tau-bench's own reward gives a do-nothing agent 38%
  success on their benchmark; ours gives 0% and 4%.
- `heuristic` must clearly beat `greedy_*`. It does, by ~1.4 reward.

Metrics distinguish `None` from `0.0`. A disruption-recovery rate of 1.0 when nothing was
disrupted is not a pass, it is an unmeasured metric.

---

## 6. What is honestly wrong with this

- **The heuristic solves 96%**, which leaves a learned policy almost nothing to discover.
  Difficulty needs tightening; the knob exists
  (`difficulty` scales budget, patience and revealed axes) but is not yet calibrated.
- **The client is not yet LLM-driven.** The interface and model choice are settled
  (`openai/gpt-4.1-mini` on an OpenAI-compatible gateway, with `claude-haiku-4.5` for the simulator-swap
  sensitivity test) but `client.py` is unwritten. This is the top of the list.
- **There are no tests.** Everything above rests on code verified by running it, not by a
  suite. This is the largest risk in the repo.
- **Causality gaps I know about:** inventory depletes once at world-build rather than during
  the episode, so dithering is free; booking a late flight does not invalidate a morning
  activity; the client reacts to an axis score rather than to a specific item; hotel check-in
  time versus flight arrival is unmodelled.
- **The oracle and regret metric are designed but unbuilt**, so returns are not normalised
  against achievable optimum.
- **Priors are self-certified.** I have no real booking data. Price levels, the 2.2 detour
  factor and the depletion rates are calibrated against plausibility, not measurement.
  Internal consistency is testable; the gap to reality is not, from inside.

Two bugs found late and worth recording, because both were invisible to `check_env`:
the return leg was being drawn from the outbound slate (you could fly home westbound), and
patience exhaustion after booking left the episode with **no reachable terminal state** —
`check_env` cannot catch that, because it only ever takes one step.

---

## 7. What I'd do next

1. Tests — determinism rollout, ledger reconciliation, mask correctness, `Φ(terminal)=0`.
2. The LLM client, then an LLM agent evaluated on the same masked action set.
3. Tighten difficulty into the 10–80% band.
4. Real geography via a Google Maps transfer matrix fetched once and committed offline.
5. Accessibility as a hard constraint — step-free access, dietary needs — masking actions
   rather than reducing utility. Real travel-agent work, and a clean gate.
6. Deepen the cascade: cross-booking temporal invalidation, in-episode price movement.

## 8. If this were connected to real APIs

The data model is already shaped for it. `Flight` is a flattened projection of an Amadeus
Flight Offers Search result (`itineraries[].segments[]`, `fareDetailsBySegment[].cabin` and
`brandedFare`, `numberOfBookableSeats`); lodging keeps the real property → room type → rate
plan split with board-basis codes.

Three things would change. **Search becomes slow and rate-limited**, so the slate mechanic
stops being a convenience and becomes a necessity, with caching and a real per-turn latency
cost. **Fare rules move behind a second call** — we flatten refundability onto the offer;
Amadeus does not. And **`on_time_rate` has no API**: it would come from a separate
historical-performance dataset, which is a data-engineering problem rather than an
integration one.

The reward and the ledger would not change at all. That is the point of making the ledger
authoritative — it does not care whether the inventory behind it is simulated or real.
