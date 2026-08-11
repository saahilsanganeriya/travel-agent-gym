"""TravelAgentEnv — a Gymnasium environment for end-to-end trip planning.

The class is a thin orchestrator. All world logic lives in plain objects that know nothing
about RL (`world`, `personas`, `itinerary`, `ledger`); this file only maps them onto the
Gymnasium contract. That separation is what lets the same world serve both a masked
`Discrete` policy and an LLM tool-calling adapter.

Two conventions worth stating because they are easy to get wrong:

* **Observation bounds are wider than the termination thresholds.** We terminate when the
  budget goes negative, so the budget `Box` low is below zero — otherwise the terminal
  observation falls outside its own space and `check_env` fails on it. CartPole does the
  same thing, terminating at ±12° while declaring the space at ±24°.
* **`truncated` is always False here.** The turn cap is `TimeLimit`'s job, applied by
  `gym.make` from `max_episode_steps`. Running out of *client patience*, by contrast, is
  part of the MDP — the client really did walk away — so that is `terminated`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from travel_env.actions import (
    N_ACTIONS,
    SLATE_K,
    FlightPreset,
    HotelSort,
    Verb,
    decode,
    describe,
    empty_mask,
    verb_slice,
)
from travel_env.contracts import (
    Activity,
    ActivityCategory,
    BookingKind,
    CabinClass,
    CityCode,
    FareBrand,
    Flight,
    Hotel,
    PreferenceAxis,
    RatePlan,
    RoomOffer,
)
from travel_env.itinerary import Itinerary
from travel_env.ledger import Ledger
from travel_env.personas import draw_persona, opening_request
from travel_env.rewards import (
    RewardInputs,
    potential,
    step_reward,
    terminal_reward,
    zero_components,
)
from travel_env.world import generate_world, generate_world_cached

MAX_TURNS = 60
N_AXES = len(PreferenceAxis)
_CITY_ORDER: tuple[CityCode, ...] = (CityCode.TOKYO, CityCode.KYOTO, CityCode.OSAKA)
_CATEGORIES: tuple[ActivityCategory, ...] = tuple(ActivityCategory)


def _stratified(pool: list, *, key, k: int) -> list:
    """Take the top few by the requested sort, then spread the rest across the price range.

    A search that returns only the top-k by one criterion is a dead end: sort by rating and
    every result is unaffordable, sort by price and every result is grim. Real search
    surfaces a spread, and the spread is what makes selection a decision.
    """
    if len(pool) <= k:
        return list(pool)
    head = pool[: k // 2]
    rest = sorted(pool[k // 2 :], key=key)
    if not rest:
        return head
    step = max(1, len(rest) // (k - len(head)))
    tail = rest[::step][: k - len(head)]
    return head + tail


def _min_lodging_cost(world, nights: int) -> int:
    """Cheapest possible lodging for the whole trip. Reserved out of the flight budget so
    an agent cannot book two flights and then be unable to sleep anywhere."""
    if not world.room_offers:
        return 0
    return min(o.price_per_night for o in world.room_offers) * nights


class TravelAgentEnv(gym.Env):
    """One client engagement: interview, search, compose, book, then live the trip."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        nights: int = 7,
        difficulty: float = 0.5,
        enable_act_two: bool = True,
        exact_worlds: bool = True,
        client=None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.nights = nights
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.enable_act_two = enable_act_two
        self.exact_worlds = exact_worlds
        self.client = client  # None → scripted behaviour, set by the harness for LLM runs

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                # low < 0 on purpose: the terminal observation must be inside the space.
                "budget": spaces.Box(-2.0, 4.0, shape=(3,), dtype=np.float32),
                "clock": spaces.Box(-1.0, 2.0, shape=(4,), dtype=np.float32),
                "belief": spaces.Box(0.0, 1.0, shape=(N_AXES, 2), dtype=np.float32),
                "flight_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 9), dtype=np.float32),
                "hotel_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 9), dtype=np.float32),
                "activity_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 7), dtype=np.float32),
                "itinerary": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                "phase": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
            }
        )

    # ---- lifecycle -------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)  # MUST be first: this seeds self.np_random
        opts = options or {}
        difficulty = float(opts.get("difficulty", self.difficulty))

        world_seed = int(self.np_random.integers(0, 2**31 - 1))
        trip_start = datetime(2027, 4, 3, tzinfo=UTC) + timedelta(
            days=int(self.np_random.integers(-60, 240))
        )
        # Exact by default: world pooling measured SLOWER than regeneration (see
        # `generate_world_cached`). Kept as an opt-in so the experiment is reproducible.
        build = generate_world if self.exact_worlds else generate_world_cached
        self.world = build(world_seed, trip_start=trip_start, nights=self.nights)
        self.districts = self.world.district_index()
        self.persona = draw_persona(self.np_random, difficulty=difficulty)

        self.ledger = Ledger(self.persona.budget_cents)
        self.itinerary = Itinerary(
            trip_start=trip_start,
            nights=self.nights,
            party_size=self.persona.party_size,
            districts=self.districts,
        )

        # Belief: what the agent has elicited. Channel 0 = value, 1 = confidence.
        self.belief = np.zeros((N_AXES, 2), dtype=np.float32)
        for axis in self.persona.stated_axes:
            i = list(PreferenceAxis).index(axis)
            self.belief[i, 0] = self.persona.weights[axis] / 9.0
            self.belief[i, 1] = 1.0

        self.turn = 0
        self.patience = float(self.persona.initial_patience)
        self.phase = 0  # 0 = planning, 1 = trip running, 2 = done
        self.trip_day = 0
        self.proposed = False
        self._proposed_signature: tuple | None = None
        self.client_walked = False
        self.disruption_damage_cents = 0
        self.forfeited_activity_value = 0.0
        self.seen_searches: set[int] = set()
        self.transcript: list[tuple[str, str]] = [
            ("client", opening_request(self.persona, self.nights))
        ]

        self.flight_slate: list[Flight] = []
        self.hotel_slate: list[tuple[Hotel, RoomOffer]] = []
        self.activity_slate: list[Activity] = []
        self.rebook_slate: list[Flight] = []

        self._prev_potential = self._potential()
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        act = decode(int(action))
        mask = self.action_masks()
        if not mask[int(action)]:
            # Unreachable when the caller honours the mask. A check a second code path can
            # bypass is not a check, so we raise rather than silently no-op.
            raise ValueError(
                f"illegal action {int(action)} ({describe(int(action))}); "
                "call action_masks() and sample within it"
            )

        self.turn += 1
        redundant = False
        components = zero_components()

        match act.verb:
            case Verb.ASK:
                self._do_ask(act.arg)
            case Verb.SEARCH_FLIGHTS:
                redundant = self._mark_search(int(action))
                self._search_flights(list(FlightPreset)[act.arg])
            case Verb.SEARCH_HOTELS:
                redundant = self._mark_search(int(action))
                self._search_hotels(
                    _CITY_ORDER[act.city_index], list(HotelSort)[act.sort_index]
                )
            case Verb.SEARCH_ACTIVITIES:
                redundant = self._mark_search(int(action))
                self._search_activities(_CATEGORIES[act.arg])
            case Verb.SELECT_FLIGHT:
                self._select_flight(act.arg)
            case Verb.SELECT_HOTEL:
                self._select_hotel(act.arg)
            case Verb.SELECT_ACTIVITY:
                self._select_activity(act.arg)
            case Verb.PROPOSE:
                self._do_propose()
            case Verb.BOOK:
                self._do_book()
            case Verb.FINALIZE:
                self._begin_act_two()
            case Verb.REBOOK:
                self._do_rebook(act.arg)

        # Patience depletes, but not uniformly: talking to the client is engagement,
        # silently grinding through searches is dead air. A real client tolerates being
        # interviewed far better than being left waiting.
        drain = self.persona.patience_decay_per_turn
        if act.verb is Verb.ASK:
            drain *= 0.45
        elif act.verb in (Verb.PROPOSE, Verb.BOOK, Verb.FINALIZE):
            drain *= 0.25  # visible progress buys goodwill
        self.patience -= drain
        if self.patience <= 0 and self.phase == 0:
            if self.ledger.active_bookings():
                # Money is committed, so the client cannot simply walk — but they are out
                # of patience, so the engagement force-closes and the trip runs as booked.
                # Without this branch the episode has no reachable terminal state at all:
                # `client_walked` is gated off and `phase` never advances. `check_env`
                # cannot catch that, because it only ever takes one step.
                self._begin_act_two()
            else:
                self.client_walked = True

        terminated = self.client_walked or self.phase == 2
        new_potential = self._potential(is_terminal=terminated)
        reward, step_parts = step_reward(
            prev_potential=self._prev_potential,
            new_potential=new_potential,
            redundant_search=redundant,
        )
        components.update(step_parts)
        self._prev_potential = new_potential

        if terminated:
            term_reward, term_parts = terminal_reward(self._reward_inputs())
            reward += term_reward
            components.update(term_parts)
            self.ledger.assert_reconciled()

        info = self._info()
        info.update(components)
        # truncated is False: TimeLimit owns the turn cap (see module docstring).
        return self._obs(), float(reward), bool(terminated), False, info

    # ---- action handlers ---------------------------------------------------------------

    def _mark_search(self, action: int) -> bool:
        redundant = action in self.seen_searches
        self.seen_searches.add(action)
        return redundant

    def _do_ask(self, axis_index: int) -> None:
        """Elicit one preference axis. This is the only way hidden weights become visible.

        A real client answers imperfectly: `reply_verbosity` controls how much signal comes
        back, so a terse client yields a noisy reading and the agent may need to probe
        twice. That is the behaviour, not a bug.
        """
        axis = list(PreferenceAxis)[axis_index]
        true_value = self.persona.weights[axis] / 9.0
        clarity = 0.35 + 0.65 * (self.persona.reply_verbosity / 9.0)
        noise = float(self.np_random.normal(0.0, 0.18 * (1.0 - clarity)))
        observed = float(np.clip(true_value + noise, 0.0, 1.0))

        i = axis_index
        prior_conf = self.belief[i, 1]
        # Bayesian-flavoured update: repeated asks converge, with diminishing returns.
        new_conf = min(1.0, prior_conf + clarity * (1.0 - prior_conf))
        if new_conf > 0:
            self.belief[i, 0] = (
                self.belief[i, 0] * prior_conf + observed * (new_conf - prior_conf)
            ) / new_conf
        self.belief[i, 1] = new_conf
        self.transcript.append(("agent", f"asks about {axis.value}"))

    def _search_flights(self, preset: FlightPreset) -> None:
        need_outbound = self.itinerary.outbound is None
        origin, dest = ("SFO", "HND") if need_outbound else ("HND", "SFO")
        pool = [
            f for f in self.world.flights
            if f.seats_remaining >= self.persona.party_size
            and (f.origin == origin if need_outbound else f.destination == dest)
            and (f.destination in ("HND", "NRT") if need_outbound else f.origin in ("HND", "NRT", "KIX"))
        ]
        match preset:
            case FlightPreset.CHEAPEST:
                pool.sort(key=lambda f: f.price)
            case FlightPreset.FASTEST:
                pool.sort(key=lambda f: f.duration_minutes)
            case FlightPreset.MOST_RELIABLE:
                pool.sort(key=lambda f: -f.on_time_rate)
            case FlightPreset.FLEXIBLE:
                pool.sort(key=lambda f: (not f.refundable, f.price))
        self.flight_slate = _stratified(pool, key=lambda f: f.price, k=SLATE_K)

    def _search_hotels(self, city: CityCode, sort: HotelSort) -> None:
        by_key = {h.key: h for h in self.world.hotels if h.city is city}
        pairs = [
            (by_key[o.hotel], o)
            for o in self.world.room_offers
            if o.hotel in by_key
            and o.rooms_remaining > 0
            and o.max_occupancy >= self.persona.party_size
        ]
        match sort:
            case HotelSort.CHEAPEST:
                pairs.sort(key=lambda p: p[1].price_per_night)
            case HotelSort.BEST_RATED:
                pairs.sort(key=lambda p: -p[0].guest_rating)
            case HotelSort.MOST_CENTRAL:
                pairs.sort(key=lambda p: p[0].minutes_to_station)
        self.hotel_slate = _stratified(pairs, key=lambda p: p[1].price_per_night, k=SLATE_K)

    def _search_activities(self, category: ActivityCategory) -> None:
        cities = set(self.itinerary.cities_visited()) or {CityCode.TOKYO}
        pool = [
            a for a in self.world.activities
            if a.category is category and a.city in cities and a.daily_capacity > 0
        ]
        pool.sort(key=lambda a: -a.rating)
        self.activity_slate = pool[:SLATE_K]

    def _select_flight(self, index: int) -> None:
        flight = self.flight_slate[index]
        if self.itinerary.outbound is None:
            self.itinerary.outbound = flight
        else:
            self.itinerary.inbound = flight
        # The slate held outbound candidates; the return leg is a different market and has
        # to be searched for. Without this an agent can "fly home" on a westbound flight.
        self.flight_slate = []

    def _select_hotel(self, index: int) -> None:
        hotel, offer = self.hotel_slate[index]
        remaining = self.nights - self.itinerary.nights_covered
        self.itinerary.lodging[hotel.city] = (hotel, offer, remaining)

    def _select_activity(self, index: int) -> None:
        activity = self.activity_slate[index]
        for day in range(self.nights):
            if self.itinerary.can_add_activity(activity, day) is None:
                self.itinerary.add_activity(activity, day)
                return

    def _do_propose(self) -> None:
        """Show the client the plan. They accept, or push back on their weakest axis.

        Pushback is what makes PROPOSE more than a formality: it reveals information, at
        the cost of a turn.
        """
        self.proposed = True
        self._proposed_signature = self._plan_signature()
        feats = self.itinerary.realized_features()
        worst = min(PreferenceAxis, key=lambda a: feats.get(a, 0.0) * self.persona.weights[a] / 9.0)
        i = list(PreferenceAxis).index(worst)
        # The complaint itself is information — a free, if narrow, elicitation.
        self.belief[i, 0] = self.persona.weights[worst] / 9.0
        self.belief[i, 1] = 1.0
        self.transcript.append(("client", f"pushes back on {worst.value}"))
        # Being made to look at a weak plan costs goodwill.
        if feats.get(worst, 0.0) < 0.3:
            self.patience -= 1.0

    def _plan_signature(self) -> tuple:
        """Identity of the current plan. Two plans with the same signature are the same
        proposal as far as the client is concerned."""
        return (
            self.itinerary.outbound.key if self.itinerary.outbound else None,
            self.itinerary.inbound.key if self.itinerary.inbound else None,
            tuple(sorted(o.key for _, o, _ in self.itinerary.lodging.values())),
            tuple(sorted(s.activity.key for s in self.itinerary.activities)),
        )

    def _do_book(self) -> None:
        """Commit the whole itinerary at once. Every charge hits the ledger here."""
        party = self.persona.party_size
        base_date = self.itinerary.trip_start.date()

        for label, flight in (("out", self.itinerary.outbound), ("ret", self.itinerary.inbound)):
            if flight is None:
                continue
            key = f"bk_flight_{label}"
            if key in self.ledger.bookings:
                continue
            self.ledger.book(
                turn=self.turn, key=key, kind=BookingKind.FLIGHT,
                inventory_key=flight.key, price=flight.price * party,
                refundable=flight.refundable, travellers=party, service_date=base_date,
            )
        for city, (hotel, offer, nights) in self.itinerary.lodging.items():
            key = f"bk_hotel_{city.value}"
            if key in self.ledger.bookings:
                continue
            self.ledger.book(
                turn=self.turn, key=key, kind=BookingKind.HOTEL,
                inventory_key=offer.key, price=offer.price_per_night * nights,
                refundable=offer.rate_plan is RatePlan.FREE_CANCELLATION,
                travellers=party, service_date=base_date,
                depends_on=("bk_flight_out",),
            )
        for n, slot in enumerate(self.itinerary.activities):
            key = f"bk_act_{n}"
            if key in self.ledger.bookings:
                continue
            self.ledger.book(
                turn=self.turn, key=key, kind=BookingKind.ACTIVITY,
                inventory_key=slot.activity.key,
                price=slot.activity.price_per_person * party,
                refundable=not slot.activity.booking_required, travellers=party,
                service_date=(self.itinerary.trip_start + timedelta(days=slot.trip_day)).date(),
                depends_on=("bk_flight_out",),
            )

    def _begin_act_two(self) -> None:
        """Lock the plan and run the trip. Scoring is on what actually happens."""
        if not self.enable_act_two:
            self.phase = 2
            return
        self.phase = 1
        self._run_trip()
        self.phase = 2

    def _run_trip(self) -> None:
        """Fast-forward the trip, firing conditional disruptions.

        The agent is not asked to emit no-ops on uneventful days. Each disruption is
        conditioned on what was actually booked, which is what turns act I's fare-brand and
        rate-plan choices into consequences rather than flavour.
        """
        from travel_env.disruptions import run_cascade

        damage, forfeited = run_cascade(
            rng=self.np_random,
            itinerary=self.itinerary,
            ledger=self.ledger,
            turn=self.turn,
            party_size=self.persona.party_size,
        )
        self.disruption_damage_cents += damage
        self.forfeited_activity_value += forfeited

    def _do_rebook(self, index: int) -> None:
        flight = self.rebook_slate[index]
        self.ledger.charge_change_fee(
            turn=self.turn, key="bk_flight_out", fee=flight.price // 4,
            reason="rebooking after disruption",
        )

    # ---- masks --------------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Which actions are legal right now.

        Named exactly `action_masks` because sb3-contrib's MaskablePPO looks that string up
        (`EXPECTED_METHOD_NAME`). The same array also goes into `info["action_mask"]`, which
        is Gymnasium's own convention (Taxi-v3) and is what `action_space.sample(mask)`
        consumes — so both ecosystems can drive this env without an adapter.
        """
        mask = empty_mask()
        if self.phase == 2 or self.client_walked:
            return mask  # terminal: nothing is legal

        if self.phase == 0:
            mask[verb_slice(Verb.ASK)] = 1
            mask[verb_slice(Verb.SEARCH_FLIGHTS)] = 1
            mask[verb_slice(Verb.SEARCH_ACTIVITIES)] = 1

            # Hotels only in cities you can actually reach and have not yet booked.
            hs = verb_slice(Verb.SEARCH_HOTELS)
            for ci, city in enumerate(_CITY_ORDER):
                if city not in self.itinerary.lodging:
                    for si in range(3):
                        mask[hs.start + ci * 3 + si] = 1

            party = self.persona.party_size
            committed = self.itinerary.committed_cost(party)
            headroom = self.persona.budget_cents - committed

            fs = verb_slice(Verb.SELECT_FLIGHT)
            both_legs = (
                self.itinerary.outbound is not None and self.itinerary.inbound is not None
            )
            for i, f in enumerate(self.flight_slate[:SLATE_K]):
                # Affordability is measured against the WHOLE plan, not this item alone.
                # A return leg also has to leave room for lodging, or the plan is a trap.
                reserve = 0 if self.itinerary.inbound is not None else _min_lodging_cost(
                    self.world, self.nights
                )
                if not both_legs and f.price * party <= headroom - reserve:
                    mask[fs.start + i] = 1

            hsel = verb_slice(Verb.SELECT_HOTEL)
            for i, (hotel, offer) in enumerate(self.hotel_slate[:SLATE_K]):
                nights = self.nights - self.itinerary.nights_covered
                if nights <= 0:
                    break
                if self.itinerary.can_add_lodging(hotel, offer, nights) is None and (
                    offer.price_per_night * nights <= headroom
                ):
                    mask[hsel.start + i] = 1

            asel = verb_slice(Verb.SELECT_ACTIVITY)
            for i, activity in enumerate(self.activity_slate[:SLATE_K]):
                feasible = any(
                    self.itinerary.can_add_activity(activity, d) is None
                    for d in range(self.nights)
                )
                if feasible and activity.price_per_person * party <= headroom:
                    mask[asel.start + i] = 1

            if self.itinerary.is_complete():
                # Only propose a plan the client has not already seen. Showing the same
                # itinerary twice is not a move; it is wasting the client's time.
                if self._plan_signature() != self._proposed_signature:
                    mask[verb_slice(Verb.PROPOSE)] = 1
                if self.ledger.can_afford(committed - self.ledger.spent):
                    mask[verb_slice(Verb.BOOK)] = 1
            # FINALIZE requires a booked, client-approved plan — you cannot skip to the end.
            if self.proposed and self.ledger.active_bookings():
                mask[verb_slice(Verb.FINALIZE)] = 1

        elif self.phase == 1:
            rb = verb_slice(Verb.REBOOK)
            for i in range(min(len(self.rebook_slate), SLATE_K)):
                mask[rb.start + i] = 1

        if not mask.any():
            # Never hand back an all-zero mask outside a terminal state: MaskablePPO would
            # have nothing to sample. Asking the client something is always available.
            mask[verb_slice(Verb.ASK)] = 1
        return mask

    # ---- observation --------------------------------------------------------------------

    def _potential(self, *, is_terminal: bool = False) -> float:
        required = 2 + 1 + 3  # two flights, lodging coverage, three activities is "full"
        filled = (
            (1 if self.itinerary.outbound else 0)
            + (1 if self.itinerary.inbound else 0)
            + min(1.0, self.itinerary.nights_covered / max(1, self.nights))
            + min(3, len(self.itinerary.activities))
        )
        confidence = float(self.belief[:, 1].mean())
        return potential(
            slots_filled=filled / required,
            belief_confidence=confidence,
            is_terminal=is_terminal,
        )

    def _obs(self) -> dict[str, np.ndarray]:
        budget = max(1, self.persona.budget_cents)
        obs = {
            "budget": np.array(
                [self.ledger.balance / budget, self.ledger.spent / budget, budget / 1_000_000],
                dtype=np.float32,
            ),
            "clock": np.array(
                [
                    self.turn / MAX_TURNS,
                    max(-1.0, self.patience / max(1, self.persona.initial_patience)),
                    self.nights / 14.0,
                    self.persona.party_size / 6.0,
                ],
                dtype=np.float32,
            ),
            "belief": self.belief.copy(),
            "flight_slate": self._flight_rows(),
            "hotel_slate": self._hotel_rows(),
            "activity_slate": self._activity_rows(),
            "itinerary": np.array(
                [
                    1.0 if self.itinerary.outbound else 0.0,
                    1.0 if self.itinerary.inbound else 0.0,
                    self.itinerary.nights_covered / max(1, self.nights),
                    len(self.itinerary.activities) / 6.0,
                    1.0 if self.proposed else 0.0,
                    len(self.ledger.active_bookings()) / 8.0,
                ],
                dtype=np.float32,
            ),
            "phase": np.eye(3, dtype=np.float32)[self.phase],
        }
        # Clip each key to ITS OWN declared bounds. A global clip is how an observation
        # silently escapes its space: a flight costing more than the whole budget produces
        # a ratio above the slate's high of 1.0, and at step() that is only a warning.
        # float64 promotion is the other half — np.can_cast(float64, float32) is False, so
        # a numerically perfect observation fails `contains` on dtype alone.
        out = {}
        for key, value in obs.items():
            box = self.observation_space[key]
            out[key] = np.clip(value, box.low, box.high).astype(np.float32, copy=False)
        return out

    def _flight_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 9), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        for i, f in enumerate(self.flight_slate[:SLATE_K]):
            rows[i] = [
                f.price * self.persona.party_size / budget,
                f.duration_minutes / 900.0,
                f.stops / 2.0,
                f.on_time_rate,
                1.0 if f.refundable else 0.0,
                {CabinClass.ECONOMY: 0.0, CabinClass.PREMIUM_ECONOMY: 0.5, CabinClass.BUSINESS: 1.0}[f.cabin],
                {FareBrand.BASIC: 0.0, FareBrand.STANDARD: 0.5, FareBrand.FLEX: 1.0}[f.fare_brand],
                min(1.0, f.seats_remaining / 10.0),
                (f.connection_buffer_minutes or 180) / 180.0,
            ]
        return rows

    def _hotel_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 9), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        for i, (h, o) in enumerate(self.hotel_slate[:SLATE_K]):
            d = self.districts[h.district]
            rows[i] = [
                o.price_per_night * self.nights / budget,
                h.star_rating / 5.0,
                h.guest_rating / 10.0,
                # Review count is shown so the agent *can* learn to discount a 9.8 from 11
                # reviews. Whether it does is the interesting question.
                min(1.0, h.review_count / 2000.0),
                d.walkability / 9.0,
                d.quiet / 9.0,
                min(1.0, h.minutes_to_station / 25.0),
                1.0 if o.rate_plan is RatePlan.FREE_CANCELLATION else 0.0,
                min(1.0, o.rooms_remaining / 8.0),
            ]
        return rows

    def _activity_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 7), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        for i, a in enumerate(self.activity_slate[:SLATE_K]):
            d = self.districts[a.district]
            rows[i] = [
                a.price_per_person * self.persona.party_size / budget,
                a.duration_minutes / 240.0,
                a.rating / 10.0,
                _CATEGORIES.index(a.category) / max(1, len(_CATEGORIES) - 1),
                a.physical_intensity / 9.0,
                d.cultural_density / 9.0,
                1.0 if a.booking_required else 0.0,
            ]
        return rows

    def _reward_inputs(self) -> RewardInputs:
        feats = self.itinerary.realized_features()
        return RewardInputs(
            persona=self.persona,
            realized_features=feats,
            spent_cents=self.ledger.spent,
            budget_cents=self.persona.budget_cents,
            hard_needs_met=self.itinerary.is_complete() and self.ledger.balance >= 0,
            itinerary_complete=self.itinerary.is_complete(),
            disruption_damage_cents=self.disruption_damage_cents,
            forfeited_activity_value=self.forfeited_activity_value,
            client_walked=self.client_walked,
            turns_used=self.turn,
            redundant_searches=0,
        )

    def _info(self) -> dict[str, Any]:
        """Stable key set with stable dtypes on every step.

        Under vectorisation Gymnasium fixes each key's dtype from the first sub-env to emit
        it and zero-fills absences, so a key that appears only sometimes yields silently
        wrong batched logs.
        """
        return {
            "action_mask": self.action_masks(),
            "turn": int(self.turn),
            "patience": float(self.patience),
            "spent_cents": int(self.ledger.spent),
            "budget_cents": int(self.persona.budget_cents),
            "itinerary_complete": bool(self.itinerary.is_complete()),
            "client_walked": bool(self.client_walked),
            "axes_elicited": float(self.belief[:, 1].mean()),
            "disruption_damage_cents": int(self.disruption_damage_cents),
            "phase": int(self.phase),
            "stratum": float(self.difficulty),
        }

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines = [
            f"turn {self.turn:>2}  phase {self.phase}  patience {self.patience:5.1f}",
            f"budget {self.ledger.balance/100:>9,.0f} / {self.persona.budget_cents/100:,.0f}",
            f"party {self.persona.party_size} · {self.persona.party_type.value} · "
            f"{self.persona.age_band.value} · {self.persona.communication_style.value}",
            f"elicited {self.belief[:,1].mean():.0%}  "
            f"flights {int(self.itinerary.outbound is not None)}/"
            f"{int(self.itinerary.inbound is not None)}  "
            f"nights {self.itinerary.nights_covered}/{self.nights}  "
            f"activities {len(self.itinerary.activities)}",
        ]
        return "\n".join(lines)


gym.register(
    id="travel_agent/TravelAgent-v0",
    entry_point="travel_env.env:TravelAgentEnv",
    max_episode_steps=MAX_TURNS,  # TimeLimit owns truncation
)
