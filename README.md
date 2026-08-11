# AI Travel Agent — an RL training environment

A Gymnasium environment in which an agent runs an end-to-end client engagement at a
mid-size travel agency: interview the client, decide where the trip actually goes, search
inventory, compose an itinerary that survives contact with a clock, get it approved, book
it — and then **the trip happens**, and things go wrong.

Two environments are registered, and both work:

| id | what it is | why it still exists |
|---|---|---|
| `travel_agent/TravelAgent-v2` | **the environment.** 3 regions, 13 cities, 5 transport modes, hard constraints, a real timeline | the default everywhere |
| `travel_agent/TravelAgent-v0` | frozen v1: one rail corridor, 3 cities | its trained checkpoint is the evidence the reward carries learnable signal |

## The measurement that drove v2

v1 produced this:

```
policy            solve_rate  return_iqm  complete  walk_rate  steps  elicited
null                    0.00     -1.0097      0.00       0.67   50.5     0.359
random                  0.25     -1.1373      0.21       0.83   33.2     0.523
greedy_cheapest         0.00     -1.3002      0.00       0.96   26.4     0.250
greedy_quality          0.00     -1.3123      0.00       0.96   26.4     0.250
heuristic               0.96     +0.1048      0.96       0.04   58.4     0.593
```

The good news is real and survives into v2: **both obvious exploits — "always book
the cheapest flight" and "always pick the highest-rated hotel" — score zero, and neither has
a penalty term written against it.** They lose because the environment makes them bad bets.

The bad news is the reason v2 exists. A uniformly random policy solved **one episode in
four**, and a hand-written heuristic solved **96%**. The targets were random near zero and
the heuristic near 30%, low enough to leave a learned policy room to beat it. Both missed.

Random solving 25% is the more damning number, because it says the task can be completed by
accident — the space of legal plans was too small to get lost in.

## The principle behind v2: harder ≠ noisier

| change | effect | why |
|---|---|---|
| more price noise, wider distributions, random failures | **unfair** | outcome variance swamps skill; the reward stops discriminating |
| more places, modes, and interacting rules | **hard** | there is a right answer and finding it takes work |

Every addition below had to pass one test: *would a better travel agent reliably beat a
worse one at it?* If yes it is complexity. If it only widens the outcome distribution for
everybody, it is noise, and it was not added.

---

## What v2 added

**A world with structure, not just more of the same.** Three regions that differ in kind,
not degree:

| | Japan | Italy | Thailand |
|---|---|---|---|
| intercity | superb rail | one fast spine, thin off it | poor rail; ferries and domestic air |
| price level | high | high | **0.45×** — a budget goes twice as far |
| peak | sakura, kōyō | high summer | cool-dry, Dec–Feb |
| weather risk | typhoon Aug–Oct | autumn rain | **monsoon Jun–Oct, and it cancels ferries** |
| visas | free for some passports, weeks of lead for others | Schengen | free or on arrival for most |

An episode draws one region or two; two means an open-jaw, which a random policy cannot
stumble into. Thailand's cheapest months are its wettest, so the budget option carries the
ferry risk — seasonality is coupled to reliability rather than being a flat multiplier.

**A clock that is actually correct.** v1 had three silent time bugs, all invisible to
`check_env`:

- `local_departure_hour = (dep.hour + 9) % 24` applied *Tokyo's* offset to a *San Francisco*
  departure — every long-haul leg was wrong by 16 hours.
- Activity opening hours are local, but the scheduler compared them against a UTC day base:
  a nine-hour shift that made evening venues look like morning ones.
- Every booking's `service_date` was `trip_start.date()`, so a day-five hotel night recorded
  as day zero and every date-keyed rule fired on the wrong day.

v2 resolves UTC offsets from real `zoneinfo` data once and freezes them onto the city,
derives every arrival as departure-plus-duration, and derives duration from real distance.
`clock.assert_world_coherent` then proves it: instants inside the window, arrival after
departure, declared duration matching the clock, and no leg faster or slower than its mode
can travel. **It caught a generator fault on its first run** — 96 SFO→SFO flights, because
the origin airport had been declared as one of Tokyo's.

**A world graph instead of a pile of checks.** One inequality:

```
for consecutive segments (a, b):
    a.end_utc + transfer(a.place_out, b.place_in) <= b.start_utc
```

That subsumes over-packed days, an activity in Kyoto after an activity in Tokyo with no
booked leg between them, a 09:00 tour after a flight landing at 11:00, hotel check-in
cutoffs, and an overnight bus substituting for a hotel night. **Act II needs no separate
cascade logic at all**: a delay shifts one arrival forward and everything downstream that no
longer satisfies the inequality is simply missed. The cascade is therefore arbitrarily deep
and always consistent with what was actually booked — which is the fix for v1's cascade
being one level deep and partly random.

**Hard constraints, and the mechanic that makes interviewing pay.** The party is now a
manifest of people — each with an age, a passport, access needs and dietary needs — and it
is hidden exactly like the utility weights. The opening request says "family of four", not
"one of us uses a wheelchair".

> A constraint the agent has **asked about** is enforced by the action mask, so a
> well-informed agent cannot fumble. A constraint it has **not** asked about is not masked —
> the agent can book the inaccessible hotel — and it fails at travel time.

So the cost of not interviewing the client is not a penalty term. It is a trip that does not
work. Accessibility is also distributed *geographically* (Venice 0.16–0.20 step-free, Kyoto's
Higashiyama 0.22, Marunouchi 0.92), so it interacts with the location preference instead of
sitting beside it: a client who wants Gion **and** needs step-free access is a genuinely
constrained problem rather than a filtered list.

**Inventory that moves while you dither.** Seats and rooms deplete on a clock, and the
falling number is *visible* in the observation — so losing an option is a consequence of
your own slowness, not a trap. Fares drift too, and the slate you searched ten turns ago is
stale; re-searching costs a turn. That asymmetry (depletion visible, drift hidden) is a
deliberate fairness line.

**Geography that is measured, not modelled.** `data/transfer_matrix.json` holds 108 real
Google Distance Matrix pairs, fetched **once** by `scripts/fetch_transfer_matrix.py` and
never called at runtime — a network call inside `step()` would destroy determinism and add
tail latency to a function that runs millions of times. v1's single haversine rule was about
twice as pessimistic as reality (Gion→Arashiyama: 69 min modelled, **37 measured**), which
made feasible days look infeasible. Where the matrix has no entry the fallback is a per-city
transit profile, because one global detour factor is what made Venice look like Tokyo.

---

## Raising versus scoring

A rule worth stating because it is easy to get backwards:

- **Nothing the agent does raises.** A plan that does not fit in time is not an error, it is
  a worse trip: the party cannot be there, so the thing does not happen and its value
  evaporates. `Timeline.violations()` returns a list and never raises.
- **Generator faults raise.** A flight that arrives before it departs is a bug in this
  repository, not a decision. Scoring the agent for it would be grading it for someone
  else's mistake. `assert_world_coherent` and `Ledger.assert_reconciled` both raise.

---

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[train,dev]"

PYTHONPATH=. .venv/bin/python scripts/smoke.py       # one rendered episode
PYTHONPATH=. .venv/bin/python -m eval.harness 24     # the baseline table
PYTHONPATH=. .venv/bin/python scripts/train.py --steps 400000 --n-envs 10
tensorboard --logdir logs

# optional, one-off, needs GOOGLE_MAPS_API_KEY in .env (see .env.example)
PYTHONPATH=. .venv/bin/python scripts/fetch_transfer_matrix.py
```

---

## 1. Why Gymnasium

A bespoke interface was an option. Gymnasium was chosen, with a caveat worth
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

**Why Gymnasium anyway:** those frameworks target LLM agents *only*. This environment has to
be trainable by a standard RL algorithm **and** drivable by an LLM. The `spaces` problem is
sidestepped entirely by never putting language in a space — language lives in `info` and in
the adapter, and the space carries numbers. Farama's own docs concede `Env` is *"not a
perfect reconstruction"* of an MDP, which is the licence to layer an adapter on top rather
than invent an interface.

The proof this was the right call is that a MaskablePPO policy trains on this environment
today, and the same masked action set renders as a tool schema for an LLM agent.

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
- **Known causality gaps:** inventory depletes once at world-build rather than during
  the episode, so dithering is free; booking a late flight does not invalidate a morning
  activity; the client reacts to an axis score rather than to a specific item; hotel check-in
  time versus flight arrival is unmodelled.
- **The oracle and regret metric are designed but unbuilt**, so returns are not normalised
  against achievable optimum.
- **Priors are self-certified.** There is no real booking data behind them. Price levels, the detour
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
cost. **Fare rules move behind a second call** — refundability is flattened onto the offer here;
Amadeus does not. And **`on_time_rate` has no API**: it would come from a separate
historical-performance dataset, which is a data-engineering problem rather than an
integration one.

The reward and the ledger would not change at all. That is the point of making the ledger
authoritative — it does not care whether the inventory behind it is simulated or real.
