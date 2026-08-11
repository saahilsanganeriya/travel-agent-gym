"""TravelAgentEnvV2 — the multi-region environment, and the default one.

v1 (`env.py`, registered as `TravelAgent-v0`) is left untouched and still loads its trained
checkpoint; it stays as the evidence that the reward carries learnable signal. This is what
everything new points at.

What an episode is now: interview a client whose preferences *and whose party* are hidden,
build a route across one or two regions, move between cities on real timetables, put everyone
somewhere they can actually sleep, fill the days without breaking the clock, and then live
the trip.

Three mechanics carry the difficulty, and none of them is noise:

* **The timeline is a graph** (`timeline.py`). Anything that cannot be reached in time is not
  offered; anything that *becomes* unreachable in act II is simply missed.
* **Hard constraints are masked only once elicited** (`constraints.py`). Ask about
  accessibility and inaccessible rooms disappear from the slate. Do not ask, and you can book
  them — and the party cannot use them, which surfaces at travel time.
* **Inventory moves while you dither.** Seats and rooms deplete on a visible clock, so losing
  an option is a consequence of your own slowness. Fares drift too, and the slate you searched
  ten turns ago is stale — which is exactly how a real saved search behaves.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from travel_env.actions_v2 import (
    MAX_ROUTE_CITIES,
    N_ACTIONS,
    SLATE_K,
    FlightPreset,
    GroundPreset,
    HotelSort,
    LodgingFilter,
    Verb,
    decode,
    describe,
    empty_mask,
    verb_slice,
)
from travel_env.clock import local_date, to_local
from travel_env.constraints import (
    ASKABLE,
    ConstraintClass,
    PartyNeeds,
    activity_blocked,
    ancillary_cost,
    audit,
    dietary_shortfall,
    documents_blocked,
    flight_blocked,
    ground_blocked,
    lodging_blocked,
    mode_reliability_bonus,
    region_visa_summary,
    units_needed,
)
from travel_env.contracts import (
    Activity,
    ActivityCategory,
    BookingKind,
    CabinClass,
    CityCode,
    FareBrand,
    Flight,
    GroundLeg,
    Hotel,
    LodgingType,
    PreferenceAxis,
    RatePlan,
    RoomOffer,
    TransportMode,
)
from travel_env.ledger import Ledger
from travel_env.personas_v2 import draw_persona, opening_request
from travel_env.rewards import (
    RewardInputs,
    potential,
    step_reward,
    terminal_reward,
    zero_components,
)
from travel_env.timeline import PlaceResolver, Segment, SegmentKind, Stay, Timeline
from travel_env.world_v2 import generate_world, weather_risk

MAX_TURNS = 90
"""A v2 trajectory is genuinely longer than v1's: a route has to be built city by city, four
constraint classes elicited, and intercity legs booked. Calibrated so a competent agent
finishes in roughly 30-45 turns and a wasteful one runs out."""

N_AXES = len(PreferenceAxis)
_CATEGORIES: tuple[ActivityCategory, ...] = tuple(ActivityCategory)
_WHOLE_UNIT_TYPES = (LodgingType.APARTMENT, LodgingType.RESORT, LodgingType.RYOKAN)
_BUDGET_TYPES = (LodgingType.HOSTEL, LodgingType.CAPSULE, LodgingType.GUESTHOUSE)


def _stratified(pool: list, *, key, k: int) -> list:
    """Top few by the requested sort, then a spread across the price range.

    A search returning only the top-k by one criterion is a dead end: sort by rating and
    everything is unaffordable, sort by price and everything is grim. Real search surfaces a
    spread, and the spread is what makes selection a decision.
    """
    if len(pool) <= k:
        return list(pool)
    head = pool[: k // 2]
    rest = sorted(pool[k // 2 :], key=key)
    if not rest:
        return head
    step = max(1, len(rest) // (k - len(head)))
    return head + rest[::step][: k - len(head)]


def _hash01(*parts: object) -> float:
    """A stable [0, 1) draw from a key. Deterministic across processes.

    Python's `hash()` on strings is salted per process, which would make two workers in the
    same vectorised run disagree about how fast an item sells. This is the same discipline as
    using a shared seeded RNG rather than a per-table one.
    """
    value = 2_166_136_261
    for part in parts:
        for byte in str(part).encode():
            value = ((value ^ byte) * 16_777_619) & 0xFFFF_FFFF
    return value / 0x1_0000_0000


class TravelAgentEnvV2(gym.Env):
    """One client engagement across one or two regions."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        nights: int = 7,
        difficulty: float = 0.5,
        enable_act_two: bool = True,
        live_inventory: bool = True,
        client=None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.nights = nights
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.enable_act_two = enable_act_two
        self.live_inventory = live_inventory
        self.client = client  # None -> scripted. An LLMClient makes this the tau-bench shape.

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "budget": spaces.Box(-2.0, 4.0, shape=(3,), dtype=np.float32),
                "clock": spaces.Box(-1.0, 2.0, shape=(5,), dtype=np.float32),
                "belief": spaces.Box(0.0, 1.0, shape=(N_AXES, 2), dtype=np.float32),
                "constraints": spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32),
                "route": spaces.Box(-1.0, 1.0, shape=(MAX_ROUTE_CITIES, 5), dtype=np.float32),
                "destination_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 6), dtype=np.float32),
                "flight_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 11), dtype=np.float32),
                "ground_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 8), dtype=np.float32),
                "hotel_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 12), dtype=np.float32),
                "activity_slate": spaces.Box(-1.0, 1.0, shape=(SLATE_K, 9), dtype=np.float32),
                "itinerary": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
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
        self.trip_start = datetime(2027, 4, 3, tzinfo=UTC) + timedelta(
            days=int(self.np_random.integers(-60, 240))
        )
        self.world = generate_world(
            world_seed, trip_start=self.trip_start, nights=self.nights, difficulty=difficulty
        )
        self.districts = self.world.district_index()
        self.cities = self.world.city_index()
        self.airports = self.world.airport_index()
        self.resolver = PlaceResolver(self.districts, self.airports, self.cities)

        self.persona = draw_persona(
            self.np_random, self.world, nights=self.nights,
            trip_start=self.trip_start, difficulty=difficulty,
        )
        self.needs = PartyNeeds(self.persona)
        self.ledger = Ledger(self.persona.budget_cents)
        self.timeline = Timeline(
            resolver=self.resolver,
            trip_nights=self.nights,
            origin_airport=self.world.origin_airport,
            depart_home_after_utc=self.trip_start - timedelta(days=2),
            commitments=self.persona.fixed_commitments,
        )

        self.belief = np.zeros((N_AXES, 2), dtype=np.float32)
        for axis in self.persona.stated_axes:
            i = list(PreferenceAxis).index(axis)
            self.belief[i, 0] = self.persona.weights[axis] / 9.0
            self.belief[i, 1] = 1.0

        # Which hard-constraint classes the agent has actually asked about. The mask enforces
        # only these; the rest bite in act II. This is the elicitation incentive.
        self.known: set[ConstraintClass] = set()

        self.turn = 0
        self.patience = float(self.persona.initial_patience)
        self.phase = 0
        self.proposed = False
        self._proposed_signature: tuple | None = None
        self.client_walked = False
        self.disruption_damage_cents = 0
        self.forfeited_value = 0.0
        self.constraint_failures = 0
        self.seen_searches: set[int] = set()
        self.illegal_attempts = 0
        self.events: list[str] = []
        self.route: list[list] = []  # [CityCode, nights]

        self.destination_slate: list[CityCode] = []
        self.flight_slate: list[Flight] = []
        self.ground_slate: list[GroundLeg] = []
        self.hotel_slate: list[tuple[Hotel, RoomOffer]] = []
        self.activity_slate: list[Activity] = []
        self.rebook_slate: list[Flight] = []
        self._quoted: dict[str, int] = {}  # price shown on the slate, which may be stale

        self.transcript: list[tuple[str, str]] = [
            ("client", opening_request(self.persona, self.world, self.nights))
        ]
        self._prev_potential = self._potential()
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        act = decode(int(action))
        mask = self.action_masks()
        self.turn += 1
        components = zero_components()
        redundant = False

        if not mask[int(action)]:
            # An illegal action is a WASTED TURN — not an exception, not a reward penalty.
            # `check_env` samples the space without honouring the mask, so an env that raises
            # fails standard validation; and a masked policy never reaches this branch.
            self.illegal_attempts += 1
            self._drain(1.0)
        else:
            redundant = self._dispatch(act, int(action))

        terminated = self._settle()
        new_potential = self._potential(is_terminal=terminated)
        reward, parts = step_reward(
            prev_potential=self._prev_potential,
            new_potential=new_potential,
            redundant_search=redundant,
        )
        components.update(parts)
        self._prev_potential = new_potential

        if terminated:
            term_reward, term_parts = terminal_reward(self._reward_inputs())
            reward += term_reward
            components.update(term_parts)
            self.ledger.assert_reconciled()

        info = self._info()
        info.update(components)
        return self._obs(), float(reward), bool(terminated), False, info

    def _dispatch(self, act, raw: int) -> bool:
        redundant = False
        match act.verb:
            case Verb.ASK:
                self._do_ask(act.arg)
                self._drain(0.45)
                return False
            case Verb.ASK_CONSTRAINTS:
                self._do_ask_constraints(ASKABLE[act.arg])
                self._drain(0.45)
                return False
            case Verb.SEARCH_DESTINATIONS:
                redundant = self._mark(raw)
                self.destination_slate = self._search_destinations()
            case Verb.ADD_CITY:
                self._add_city(act.arg)
            case Verb.DROP_CITY:
                self._drop_city()
            case Verb.SEARCH_FLIGHTS:
                redundant = self._mark(raw)
                self._search_flights(list(FlightPreset)[act.arg])
            case Verb.SEARCH_GROUND:
                redundant = self._mark(raw)
                self._search_ground(list(GroundPreset)[act.arg])
            case Verb.SEARCH_HOTELS:
                redundant = self._mark(raw)
                self._search_hotels(
                    list(HotelSort)[act.sort_index], list(LodgingFilter)[act.filter_index]
                )
            case Verb.SEARCH_ACTIVITIES:
                redundant = self._mark(raw)
                self._search_activities(_CATEGORIES[act.arg])
            case Verb.SELECT_FLIGHT:
                self._select_flight(act.arg)
            case Verb.SELECT_GROUND:
                self._select_ground(act.arg)
            case Verb.SELECT_HOTEL:
                self._select_hotel(act.arg)
            case Verb.SELECT_ACTIVITY:
                self._select_activity(act.arg)
            case Verb.PROPOSE:
                self._do_propose()
                self._drain(0.25)
                return redundant
            case Verb.BOOK:
                self._do_book()
                self._drain(0.25)
                return redundant
            case Verb.FINALIZE:
                self._begin_act_two()
                self._drain(0.25)
                return redundant
            case Verb.REBOOK:
                self._do_rebook(act.arg)
        self._drain(1.0)
        return redundant

    def _drain(self, scale: float) -> None:
        """Patience depletes, but not uniformly.

        Talking to the client is engagement; grinding silently through searches is dead air.
        A real client tolerates being interviewed far better than being left waiting, and
        that asymmetry is what stops the optimal policy from being "never ask".
        """
        self.patience -= self.persona.patience_decay_per_turn * scale

    def _settle(self) -> bool:
        if self.patience <= 0 and self.phase == 0:
            if self.ledger.active_bookings():
                # Money is committed, so the client cannot simply walk — the engagement
                # force-closes and the trip runs as booked. Without this branch a patience
                # wipeout after booking leaves no reachable terminal state at all.
                self._begin_act_two()
            else:
                self.client_walked = True
        return bool(self.client_walked or self.phase == 2)

    def _mark(self, action: int) -> bool:
        redundant = action in self.seen_searches
        self.seen_searches.add(action)
        return redundant

    # ---- live inventory ----------------------------------------------------------------

    def _sold(self, key: str, base: int, desirability: float) -> int:
        """How many units other customers have taken by now. VISIBLE to the agent.

        Depletion is derived from the turn rather than sampled, so it is deterministic and —
        crucially — the falling number is in the observation. Losing an option is then a
        consequence of the agent's own slowness rather than a trap, which is the difference
        between difficulty and unfairness. Desirable-and-cheap goes first, as it does in life.
        """
        if not self.live_inventory:
            return 0
        rate = (0.06 + 0.24 * desirability) * (0.5 + _hash01("demand", key))
        return int(rate * self.turn)

    def _seats(self, flight: Flight) -> int:
        desirability = float(np.clip(1.2 - flight.price / max(1, self.persona.budget_cents), 0.0, 1.0))
        return max(0, flight.seats_remaining - self._sold(flight.key, flight.seats_remaining, desirability))

    def _rooms(self, offer: RoomOffer) -> int:
        desirability = float(np.clip(1.1 - offer.price_per_night * 6 / max(1, self.persona.budget_cents), 0.0, 1.0))
        return max(0, offer.rooms_remaining - self._sold(offer.key, offer.rooms_remaining, desirability))

    def _ground_seats(self, leg: GroundLeg) -> int:
        return max(0, leg.seats_remaining - self._sold(leg.key, leg.seats_remaining, 0.35))

    def _price_now(self, key: str, base: int) -> int:
        """The fare right now. HIDDEN, in the sense that the slate shows what it showed.

        Revenue management raises prices as inventory sells, and a saved search goes stale.
        The mask and the ledger both use this value so an affordable-looking selection can
        never become an unpayable booking; the agent just does not get told that the number
        moved unless it searches again. Re-searching costs a turn — that is the tradeoff.
        """
        if not self.live_inventory:
            return base
        drift = 0.35 * _hash01("drift", key) * (self.turn / MAX_TURNS)
        return int(base * (1.0 + drift))

    # ---- action handlers ---------------------------------------------------------------

    def _do_ask(self, axis_index: int) -> None:
        """Elicit one preference axis. The only way hidden weights become visible."""
        axis = list(PreferenceAxis)[axis_index]
        true_value = self.persona.weights[axis] / 9.0
        clarity = 0.35 + 0.65 * (self.persona.reply_verbosity / 9.0)
        observed = float(
            np.clip(true_value + self.np_random.normal(0.0, 0.18 * (1.0 - clarity)), 0.0, 1.0)
        )
        prior = self.belief[axis_index, 1]
        new_conf = min(1.0, prior + clarity * (1.0 - prior))
        if new_conf > 0:
            self.belief[axis_index, 0] = (
                self.belief[axis_index, 0] * prior + observed * (new_conf - prior)
            ) / new_conf
        self.belief[axis_index, 1] = new_conf
        self.transcript.append(("agent", f"asks about {axis.value}"))

    def _do_ask_constraints(self, klass: ConstraintClass) -> None:
        """Ask about a whole class of hard constraints.

        This is the highest-value action in the environment and it looks like the least
        interesting one. Until it is taken, the mask cannot filter on that class, so the
        agent is free to book something the party cannot use — and will.
        """
        self.known.add(klass)
        self.transcript.append(("agent", f"asks about {klass.value}"))

    def _search_destinations(self) -> list[CityCode]:
        placed = {city for city, _ in self.route}
        remaining = self.nights - sum(n for _, n in self.route)
        pool = [c.code for c in self.world.cities if c.code not in placed or remaining > 0]
        # Cities the client has a commitment in come first: they are not optional.
        must = {c.city for c in self.persona.fixed_commitments}
        pool.sort(key=lambda c: (c not in must, c.value))
        return pool[:SLATE_K]

    def _add_city(self, index: int) -> None:
        """Append a city, or extend the current last stay by one night."""
        city = self.destination_slate[index]
        if self.route and self.route[-1][0] is city:
            self.route[-1][1] += 1
        else:
            self.route.append([city, 1])
        self._rebuild_stays()

    def _drop_city(self) -> None:
        if not self.route:
            return
        self.route[-1][1] -= 1
        if self.route[-1][1] <= 0:
            self.route.pop()
        self._rebuild_stays()

    def _rebuild_stays(self) -> None:
        """Re-seat booked lodging against the route's night allocation.

        Stays are keyed by city, so changing the route length re-slots them rather than
        discarding them; a stay whose city has left the route is dropped.
        """
        by_city = {stay.hotel.city: stay for stay in self.timeline.stays}
        rebuilt: list[Stay] = []
        cursor = 0
        for city, nights in self.route:
            stay = by_city.get(city)
            if stay is not None:
                stay.first_night = cursor
                stay.nights = nights
                rebuilt.append(stay)
            cursor += nights
        self.timeline.stays = rebuilt

    def _search_flights(self, preset: FlightPreset) -> None:
        need_outbound = self.timeline.outbound is None
        origin = self.world.origin_airport
        if need_outbound:
            targets = self._airports_of(self.route[0][0]) if self.route else self._all_gateways()
            pool = [f for f in self.world.flights if f.origin == origin and f.destination in targets]
        else:
            sources = self._airports_of(self.route[-1][0]) if self.route else self._all_gateways()
            pool = [f for f in self.world.flights if f.destination == origin and f.origin in sources]

        pool = [f for f in pool if self._seats(f) >= self.needs.size]
        if ConstraintClass.ACCESSIBILITY in self.known or ConstraintClass.CAPACITY in self.known:
            pool = [f for f in pool if flight_blocked(self.needs, f) is None]

        match preset:
            case FlightPreset.CHEAPEST:
                pool.sort(key=lambda f: self._price_now(f.key, f.price))
            case FlightPreset.FASTEST:
                pool.sort(key=lambda f: f.duration_minutes)
            case FlightPreset.MOST_RELIABLE:
                pool.sort(key=lambda f: (f.is_self_transfer, -f.on_time_rate))
            case FlightPreset.FLEXIBLE:
                pool.sort(key=lambda f: (not f.refundable, f.price))
        self.flight_slate = _stratified(pool, key=lambda f: f.price, k=SLATE_K)
        for f in self.flight_slate:
            self._quoted[f.key] = self._price_now(f.key, f.price)

    def _all_gateways(self) -> set[str]:
        return {a.key for a in self.world.airports if a.key != self.world.origin_airport}

    def _airports_of(self, city: CityCode) -> set[str]:
        return {a.key for a in self.world.airports if a.city is city}

    def _next_hop(self) -> tuple[CityCode, CityCode] | None:
        """The first route hop with no booked leg yet. Ground search targets exactly this."""
        booked = {(leg.origin_city, leg.destination_city) for leg in self.timeline.ground}
        for (a, _), (b, _) in zip(self.route, self.route[1:], strict=False):
            if a is not b and (a, b) not in booked:
                return a, b
        return None

    def _search_ground(self, preset: GroundPreset) -> None:
        hop = self._next_hop()
        if hop is None:
            self.ground_slate = []
            return
        origin, destination = hop
        pool = [
            leg for leg in self.world.ground
            if leg.origin_city is origin
            and leg.destination_city is destination
            and self._ground_seats(leg) >= self.needs.size
        ]
        if self.known & {ConstraintClass.ACCESSIBILITY, ConstraintClass.CAPACITY}:
            pool = [leg for leg in pool if ground_blocked(self.needs, leg) is None]
        # Only offer legs the plan can actually accommodate: this is the timeline predicate
        # doing the masking, so an unreachable departure is never on the slate at all.
        pool = [leg for leg in pool if self.timeline.would_break(self._ground_segment(leg)) is None]

        match preset:
            case GroundPreset.CHEAPEST:
                pool.sort(key=lambda leg: self._price_now(leg.key, leg.price))
            case GroundPreset.FASTEST:
                pool.sort(key=lambda leg: leg.duration_minutes)
            case GroundPreset.ROBUST:
                pool = [leg for leg in pool if not leg.weather_sensitive]
                pool.sort(key=lambda leg: -leg.on_time_rate)
            case GroundPreset.OVERNIGHT:
                pool = [leg for leg in pool if leg.duration_minutes >= 300]
                pool.sort(key=lambda leg: self._price_now(leg.key, leg.price))
        self.ground_slate = _stratified(pool, key=lambda leg: leg.price, k=SLATE_K)

    def _ground_segment(self, leg: GroundLeg, index: int | None = None) -> Segment:
        n = len(self.timeline.ground) if index is None else index
        return Segment(
            key=f"seg_ground_{n}",
            kind=SegmentKind.TRANSPORT,
            start_utc=leg.departure_utc,
            end_utc=leg.arrival_utc,
            place_in=self.resolver.station(leg.origin_city),
            place_out=self.resolver.station(leg.destination_city),
            label=f"{leg.operator} {leg.mode.value}",
            ref=leg,
        )

    def _pending_city(self) -> CityCode | None:
        """First route city without lodging. Hotel search always targets it, so the agent
        never has to name a city — the route already did."""
        housed = {stay.hotel.city for stay in self.timeline.stays}
        for city, _ in self.route:
            if city not in housed:
                return city
        return None

    def _search_hotels(self, sort: HotelSort, lodging: LodgingFilter) -> None:
        city = self._pending_city()
        if city is None:
            self.hotel_slate = []
            return
        by_key = {h.key: h for h in self.world.hotels if h.city is city}
        if lodging is LodgingFilter.WHOLE_UNIT:
            by_key = {k: h for k, h in by_key.items() if h.lodging_type in _WHOLE_UNIT_TYPES}
        elif lodging is LodgingFilter.BUDGET:
            by_key = {k: h for k, h in by_key.items() if h.lodging_type in _BUDGET_TYPES}

        pairs = [
            (by_key[o.hotel], o)
            for o in self.world.room_offers
            if o.hotel in by_key and self._rooms(o) > 0
        ]
        if ConstraintClass.ACCESSIBILITY in self.known or ConstraintClass.CAPACITY in self.known:
            pairs = [
                (h, o) for h, o in pairs
                if lodging_blocked(self.needs, h, o, units_needed(self.needs, o)) is None
            ]
        match sort:
            case HotelSort.CHEAPEST:
                pairs.sort(key=lambda p: self._unit_cost(p[1]))
            case HotelSort.BEST_RATED:
                pairs.sort(key=lambda p: -p[0].guest_rating)
            case HotelSort.MOST_CENTRAL:
                pairs.sort(key=lambda p: p[0].minutes_to_station)
        self.hotel_slate = _stratified(pairs, key=lambda p: self._unit_cost(p[1]), k=SLATE_K)

    def _unit_cost(self, offer: RoomOffer) -> int:
        """Nightly cost for the whole party, which is what actually matters.

        An apartment at twice the nightly rate that sleeps the whole family in one unit is
        cheaper than three hotel rooms, and sorting on the headline rate hides that.
        """
        return self._price_now(offer.key, offer.price_per_night) * units_needed(self.needs, offer)

    def _search_activities(self, category: ActivityCategory) -> None:
        cities = {city for city, _ in self.route}
        pool = [
            a for a in self.world.activities
            if a.category is category and a.city in cities and a.daily_capacity > 0
        ]
        if ConstraintClass.ACCESSIBILITY in self.known:
            day = local_date(self.trip_start, 0)
            pool = [a for a in pool if activity_blocked(self.needs, a, day) is None]
        pool.sort(key=lambda a: -a.rating)
        self.activity_slate = pool[:SLATE_K]

    def _select_flight(self, index: int) -> None:
        flight = self.flight_slate[index]
        if self.timeline.outbound is None:
            self.timeline.outbound = flight
        else:
            self.timeline.inbound = flight
        # The slate held one market; the other leg is a different one and has to be searched
        # for. Without this an agent can "fly home" on an inbound flight.
        self.flight_slate = []

    def _select_ground(self, index: int) -> None:
        self.timeline.ground.append(self.ground_slate[index])
        self.timeline.ground.sort(key=lambda leg: leg.departure_utc)
        self.ground_slate = []

    def _select_hotel(self, index: int) -> None:
        hotel, offer = self.hotel_slate[index]
        city = hotel.city
        first = 0
        for route_city, nights in self.route:
            if route_city is city:
                self.timeline.stays.append(
                    Stay(
                        hotel=hotel,
                        offer=offer,
                        units=units_needed(self.needs, offer),
                        first_night=first,
                        nights=nights,
                        with_cot=bool(offer.cot_available and self.needs.has_minor),
                    )
                )
                break
            first += nights
        self.hotel_slate = []

    def _select_activity(self, index: int) -> None:
        activity = self.activity_slate[index]
        slot = self._first_feasible_slot(activity)
        if slot is not None:
            self.timeline.activities.append((activity, slot))
            self.timeline.activities.sort(key=lambda pair: pair[1])

    def _first_feasible_slot(self, activity: Activity) -> datetime | None:
        """Earliest start that satisfies opening hours, the calendar and every transfer."""
        zero = self.timeline.day_zero()
        if zero is None:
            return None
        offset = self.resolver.offset(activity.city)
        for day in range(self.nights):
            local_day = zero + timedelta(days=day)
            if local_day.isoweekday() in activity.closed_weekdays:
                continue
            if local_day.month not in activity.season_months:
                continue
            base = datetime(local_day.year, local_day.month, local_day.day)
            opens = (base + timedelta(hours=activity.opens_hour) - timedelta(minutes=offset)).replace(tzinfo=UTC)
            closes = (base + timedelta(hours=activity.closes_hour) - timedelta(minutes=offset)).replace(tzinfo=UTC)
            start = self.timeline.earliest_start_for(activity, day, opens, closes)
            if start is None:
                continue
            candidate = Segment(
                key=f"seg_act_{len(self.timeline.activities)}",
                kind=SegmentKind.ACTIVITY,
                start_utc=start,
                end_utc=start + timedelta(minutes=activity.duration_minutes),
                place_in=self.resolver.district(activity.district),
                place_out=self.resolver.district(activity.district),
                label=activity.name,
                ref=activity,
            )
            if self.timeline.would_break(candidate) is None:
                return start
        return None

    def _do_propose(self) -> None:
        """Show the client the plan. They accept, or push back on their weakest axis."""
        self.proposed = True
        self._proposed_signature = self._plan_signature()
        feats = self._realized_features()
        worst = min(PreferenceAxis, key=lambda a: feats.get(a, 0.0) * self.persona.weights[a] / 9.0)
        i = list(PreferenceAxis).index(worst)
        self.belief[i, 0] = self.persona.weights[worst] / 9.0
        self.belief[i, 1] = 1.0
        self.transcript.append(("client", f"pushes back on {worst.value}"))
        if feats.get(worst, 0.0) < 0.3:
            self.patience -= 1.0

    def _plan_signature(self) -> tuple:
        return (
            self.timeline.outbound.key if self.timeline.outbound else None,
            self.timeline.inbound.key if self.timeline.inbound else None,
            tuple(leg.key for leg in self.timeline.ground),
            tuple(sorted(s.offer.key for s in self.timeline.stays)),
            tuple(sorted(a.key for a, _ in self.timeline.activities)),
        )

    def _committed_cost(self) -> int:
        party = self.needs.size
        total = 0
        for flight in (self.timeline.outbound, self.timeline.inbound):
            if flight is not None:
                total += self._price_now(flight.key, flight.price) * party
        for leg in self.timeline.ground:
            total += self._price_now(leg.key, leg.price) * party
        for stay in self.timeline.stays:
            total += self._price_now(stay.offer.key, stay.offer.price_per_night) * stay.nights * stay.units
        for activity, _ in self.timeline.activities:
            total += activity.price_per_person * party
        return total + ancillary_cost(self.needs, self.timeline, self.world)

    def _do_book(self) -> None:
        """Commit the whole plan at once. Every charge hits the ledger here."""
        party = self.needs.size
        base_date = self.trip_start.date()

        for label, flight in (("out", self.timeline.outbound), ("ret", self.timeline.inbound)):
            if flight is None or f"bk_flight_{label}" in self.ledger.bookings:
                continue
            self.ledger.book(
                turn=self.turn, key=f"bk_flight_{label}", kind=BookingKind.FLIGHT,
                inventory_key=flight.key,
                price=self._price_now(flight.key, flight.price) * party,
                refundable=flight.refundable, travellers=party,
                # The LOCAL date where it happens, not the trip start. v1 stamped every
                # booking with day zero, so every date-keyed rule fired on the wrong day.
                service_date=local_date(
                    flight.departure_utc, self.airports[flight.origin].utc_offset_minutes
                ),
            )
        for n, leg in enumerate(self.timeline.ground):
            key = f"bk_ground_{n}"
            if key in self.ledger.bookings:
                continue
            self.ledger.book(
                turn=self.turn, key=key, kind=BookingKind.GROUND, inventory_key=leg.key,
                price=self._price_now(leg.key, leg.price) * party,
                refundable=leg.refundable, travellers=party,
                service_date=local_date(leg.departure_utc, self.resolver.offset(leg.origin_city)),
                depends_on=("bk_flight_out",),
            )
        for stay in self.timeline.stays:
            key = f"bk_stay_{stay.hotel.key}"
            if key in self.ledger.bookings:
                continue
            nightly = self._price_now(stay.offer.key, stay.offer.price_per_night) * stay.units
            cot = stay.offer.cot_price_per_night * stay.units if stay.with_cot else 0
            self.ledger.book(
                turn=self.turn, key=key, kind=BookingKind.HOTEL, inventory_key=stay.offer.key,
                price=(nightly + cot) * stay.nights,
                refundable=stay.offer.rate_plan is RatePlan.FREE_CANCELLATION,
                travellers=party, service_date=base_date, depends_on=("bk_flight_out",),
            )
        for n, (activity, start) in enumerate(self.timeline.activities):
            key = f"bk_act_{n}"
            if key in self.ledger.bookings:
                continue
            self.ledger.book(
                turn=self.turn, key=key, kind=BookingKind.ACTIVITY, inventory_key=activity.key,
                price=activity.price_per_person * party,
                refundable=not activity.booking_required, travellers=party,
                service_date=local_date(start, self.resolver.offset(activity.city)),
                depends_on=("bk_flight_out",),
            )

    def _begin_act_two(self) -> None:
        if not self.enable_act_two:
            self.phase = 2
            return
        self.phase = 1
        self._run_trip()
        self.phase = 2

    def _run_trip(self) -> None:
        """Live the trip. Delays are sampled; everything downstream is DERIVED.

        The only random draws here are whether each booked leg runs late and whether a
        weather-sensitive leg is cancelled outright. Every consequence after that — the missed
        tour, the lost room, the second missed tour in the next city — comes out of the
        timeline no longer being satisfiable. That is why the cascade is arbitrarily deep and
        always consistent with what the agent actually bought.
        """
        rng = self.np_random
        day_of_year = self.trip_start.timetuple().tm_yday

        for flight, key in ((self.timeline.outbound, "seg_out"), (self.timeline.inbound, "seg_ret")):
            if flight is None:
                continue
            if rng.random() < (1.0 - flight.on_time_rate):
                delay = int(np.clip(rng.pareto(1.4) * 45, 15, 600))
                self.timeline.apply_delay(key, delay)
                self.events.append(f"{flight.carrier}{flight.flight_number} delayed {delay}m")
                buffer_minutes = flight.connection_buffer_minutes
                if flight.stops and buffer_minutes is not None and delay > buffer_minutes:
                    if flight.is_self_transfer:
                        # No protection on two tickets: the replacement is a walk-up fare.
                        cost = int(flight.price * 1.6) * self.needs.size
                        self.ledger.charge_change_fee(
                            turn=self.turn, key=f"bk_flight_{'out' if key == 'seg_out' else 'ret'}",
                            fee=cost, reason="self-transfer misconnection, no protection",
                        )
                        self.disruption_damage_cents += cost
                        self.events.append("SELF-TRANSFER STRANDED — walk-up fare bought")
                    else:
                        self.timeline.apply_delay(key, int(rng.integers(120, 420)))
                        self.events.append("missed connection, re-accommodated")

        for n, leg in enumerate(self.timeline.ground):
            risk = 1.0 - leg.on_time_rate
            if leg.weather_sensitive:
                risk += weather_risk(self.cities[leg.origin_city].region, day_of_year) * 0.6
            if rng.random() < risk:
                cancelled = leg.weather_sensitive and rng.random() < 0.5
                delay = 24 * 60 if cancelled else int(np.clip(rng.pareto(1.5) * 40, 20, 480))
                self.timeline.apply_delay(f"seg_ground_{n}", delay)
                self.events.append(
                    f"{leg.operator} {leg.mode.value} {'cancelled' if cancelled else f'delayed {delay}m'}"
                )

        self._resolve_outcome()

    def _resolve_outcome(self) -> None:
        """Price what actually happened: timeline breakage plus unmet hard constraints."""
        broken = self.timeline.violations()
        unmet = audit(
            self.persona, self.needs, self.timeline, self.world,
            departure=self.timeline.outbound.departure_utc if self.timeline.outbound else self.trip_start,
            return_date=(self.trip_start + timedelta(days=self.nights)).date(),
        )
        self.constraint_failures = len(unmet)
        lost_keys = {v.segment_key for v in broken} | {v.segment_key for v in unmet}

        total_activities = max(1, len(self.timeline.activities))
        for n, _ in enumerate(self.timeline.activities):
            if f"seg_act_{n}" not in lost_keys:
                continue
            booking_key = f"bk_act_{n}"
            if booking_key in self.ledger.bookings:
                booking = self.ledger.bookings[booking_key]
                if booking.refundable:
                    self.ledger.cancel(turn=self.turn, key=booking_key, agent_initiated=False)
                else:
                    self.disruption_damage_cents += booking.price_paid
                    self.ledger.mark_disrupted(booking_key)
            self.forfeited_value += 1.0 / total_activities

        for stay in self.timeline.stays:
            if f"stay_{stay.hotel.key}" not in lost_keys:
                continue
            booking_key = f"bk_stay_{stay.hotel.key}"
            if booking_key in self.ledger.bookings:
                booking = self.ledger.bookings[booking_key]
                if booking.refundable:
                    refund = self.ledger.cancel(turn=self.turn, key=booking_key, agent_initiated=False)
                    # A same-night replacement costs a premium over what was refunded.
                    self.disruption_damage_cents += max(0, int(booking.price_paid * 1.4) - refund)
                else:
                    self.disruption_damage_cents += booking.price_paid
                    self.ledger.mark_disrupted(booking_key)
            self.forfeited_value += 0.35

        if any(v.segment_key == "documents" for v in unmet):
            # Denied boarding: the trip does not happen at all. Nothing is refundable at the
            # gate, and this is entirely preventable by asking about passports.
            self.forfeited_value += 1.0
            self.events.append("DENIED BOARDING — documents")

        for violation in broken + unmet:
            self.events.append(str(violation))
        self.forfeited_value = min(2.0, self.forfeited_value)

        for booking in list(self.ledger.active_bookings()):
            self.ledger.mark_completed(booking.key)

    def _do_rebook(self, index: int) -> None:
        flight = self.rebook_slate[index]
        self.ledger.charge_change_fee(
            turn=self.turn, key="bk_flight_out", fee=flight.price // 4,
            reason="rebooking after disruption",
        )

    # ---- masks --------------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Which actions are legal right now.

        Named exactly `action_masks` because that is the string sb3-contrib looks up. The same
        array goes into `info["action_mask"]`, which is Gymnasium's own Taxi-v3 convention.
        """
        mask = empty_mask()
        if self.phase == 2 or self.client_walked:
            return mask

        if self.phase == 1:
            rb = verb_slice(Verb.REBOOK)
            for i in range(min(len(self.rebook_slate), SLATE_K)):
                mask[rb.start + i] = 1
            if not mask.any():
                mask[verb_slice(Verb.ASK)] = 1
            return mask

        mask[verb_slice(Verb.ASK)] = 1
        # Asking about a class you already know is not a move; it is wasting the client's time.
        ac = verb_slice(Verb.ASK_CONSTRAINTS)
        for i, klass in enumerate(ASKABLE):
            if klass not in self.known:
                mask[ac.start + i] = 1
        mask[verb_slice(Verb.SEARCH_DESTINATIONS)] = 1

        allocated = sum(n for _, n in self.route)
        if allocated < self.nights and len(self.route) <= MAX_ROUTE_CITIES:
            add = verb_slice(Verb.ADD_CITY)
            for i, city in enumerate(self.destination_slate[:SLATE_K]):
                # A new city needs a route slot; extending the last one always fits.
                extending = bool(self.route and self.route[-1][0] is city)
                if extending or len(self.route) < MAX_ROUTE_CITIES:
                    mask[add.start + i] = 1
        if self.route:
            mask[verb_slice(Verb.DROP_CITY)] = 1

        route_ready = bool(self.route) and allocated == self.nights
        if route_ready:
            mask[verb_slice(Verb.SEARCH_FLIGHTS)] = 1
            if self._next_hop() is not None:
                mask[verb_slice(Verb.SEARCH_GROUND)] = 1
            if self._pending_city() is not None:
                mask[verb_slice(Verb.SEARCH_HOTELS)] = 1
            mask[verb_slice(Verb.SEARCH_ACTIVITIES)] = 1

        headroom = self.persona.budget_cents - self._committed_cost()
        party = self.needs.size

        fs = verb_slice(Verb.SELECT_FLIGHT)
        both = self.timeline.outbound is not None and self.timeline.inbound is not None
        for i, f in enumerate(self.flight_slate[:SLATE_K]):
            if both:
                break
            if self._price_now(f.key, f.price) * party <= headroom:
                mask[fs.start + i] = 1

        gs = verb_slice(Verb.SELECT_GROUND)
        for i, leg in enumerate(self.ground_slate[:SLATE_K]):
            if self._price_now(leg.key, leg.price) * party <= headroom:
                mask[gs.start + i] = 1

        hs = verb_slice(Verb.SELECT_HOTEL)
        pending = self._pending_city()
        if pending is not None:
            nights = next((n for c, n in self.route if c is pending), 0)
            for i, (hotel, offer) in enumerate(self.hotel_slate[:SLATE_K]):
                cost = self._unit_cost(offer) * nights
                if cost <= headroom and self._rooms(offer) >= units_needed(self.needs, offer):
                    mask[hs.start + i] = 1

        acts = verb_slice(Verb.SELECT_ACTIVITY)
        for i, activity in enumerate(self.activity_slate[:SLATE_K]):
            if activity.price_per_person * party > headroom:
                continue
            if self._first_feasible_slot(activity) is not None:
                mask[acts.start + i] = 1

        if self.timeline.is_complete():
            if self._plan_signature() != self._proposed_signature:
                mask[verb_slice(Verb.PROPOSE)] = 1
            if self.ledger.can_afford(self._committed_cost() - self.ledger.spent):
                mask[verb_slice(Verb.BOOK)] = 1
        if self.proposed and self.ledger.active_bookings():
            mask[verb_slice(Verb.FINALIZE)] = 1

        if not mask.any():
            # Never hand back an all-zero mask outside a terminal state: MaskablePPO would
            # have nothing to sample from.
            mask[verb_slice(Verb.ASK)] = 1
        return mask

    # ---- observation ----------------------------------------------------------------------

    def _potential(self, *, is_terminal: bool = False) -> float:
        required = 2 + 1 + 1 + 3  # two flights, lodging, ground hops, three activities
        allocated = sum(n for _, n in self.route)
        hops = max(1, len(self.route) - 1)
        filled = (
            (1 if self.timeline.outbound else 0)
            + (1 if self.timeline.inbound else 0)
            + min(1.0, allocated / max(1, self.nights))
            + min(1.0, len(self.timeline.ground) / hops)
            + min(3, len(self.timeline.activities))
        )
        # Knowing the hard constraints counts as progress the same way knowing the soft ones
        # does — both are information the plan depends on.
        confidence = 0.7 * float(self.belief[:, 1].mean()) + 0.3 * (len(self.known) / len(ASKABLE))
        return potential(
            slots_filled=filled / required, belief_confidence=confidence, is_terminal=is_terminal
        )

    def _realized_features(self) -> dict[PreferenceAxis, float]:
        feats = dict.fromkeys(PreferenceAxis, 0.0)
        stays = self.timeline.stays
        if stays:
            weight = sum(s.nights for s in stays) or 1
            feats[PreferenceAxis.COMFORT] = sum((s.hotel.star_rating / 5.0) * s.nights for s in stays) / weight
            feats[PreferenceAxis.LOCATION] = sum(
                (self.districts[s.hotel.district].walkability / 9.0) * s.nights for s in stays
            ) / weight

        cats = [a.category.value for a, _ in self.timeline.activities]
        food_share = min(1.0, cats.count("food") / 2.0)
        # A food lover whose restaurants cannot feed them did not get a food trip.
        feats[PreferenceAxis.FOOD] = food_share * (1.0 - dietary_shortfall(self.needs, self.timeline))
        feats[PreferenceAxis.CULTURE] = min(1.0, (cats.count("history") + cats.count("museum")) / 3.0)
        feats[PreferenceAxis.NOVELTY] = min(
            1.0,
            (len({a.district for a, _ in self.timeline.activities}) + 2 * max(0, len(self.route) - 1)) / 6.0,
        )
        per_day = len(self.timeline.activities) / max(1, self.nights)
        feats[PreferenceAxis.PACING] = max(0.0, 1.0 - abs(per_day - 1.5) / 1.5)

        legs = [f for f in (self.timeline.outbound, self.timeline.inbound) if f is not None]
        if legs:
            base = sum(f.on_time_rate for f in legs) / len(legs)
            feats[PreferenceAxis.RELIABILITY] = float(
                np.clip(base + mode_reliability_bonus(self.timeline), 0.0, 1.0)
            )
        return feats

    def _obs(self) -> dict[str, np.ndarray]:
        budget = max(1, self.persona.budget_cents)
        visas, lead = region_visa_summary(self.needs, self.world)
        allocated = sum(n for _, n in self.route)

        obs = {
            "budget": np.array(
                [self.ledger.balance / budget, self.ledger.spent / budget, budget / 2_000_000],
                dtype=np.float32,
            ),
            "clock": np.array(
                [
                    self.turn / MAX_TURNS,
                    max(-1.0, self.patience / max(1, self.persona.initial_patience)),
                    self.nights / 14.0,
                    self.needs.size / 6.0,
                    len(self.world.regions) / 2.0,
                ],
                dtype=np.float32,
            ),
            "belief": self.belief.copy(),
            "constraints": self._constraint_row(visas, lead),
            "route": self._route_rows(),
            "destination_slate": self._destination_rows(),
            "flight_slate": self._flight_rows(),
            "ground_slate": self._ground_rows(),
            "hotel_slate": self._hotel_rows(),
            "activity_slate": self._activity_rows(),
            "itinerary": np.array(
                [
                    1.0 if self.timeline.outbound else 0.0,
                    1.0 if self.timeline.inbound else 0.0,
                    allocated / max(1, self.nights),
                    len(self.timeline.stays) / max(1, MAX_ROUTE_CITIES),
                    len(self.timeline.ground) / 4.0,
                    len(self.timeline.activities) / 6.0,
                    1.0 if self.proposed else 0.0,
                    len(self.ledger.active_bookings()) / 10.0,
                ],
                dtype=np.float32,
            ),
            "phase": np.eye(3, dtype=np.float32)[self.phase],
        }
        # Clip each key to ITS OWN bounds. A global clip is how an observation silently
        # escapes its space, and float64 promotion is the other half: np.can_cast(float64,
        # float32) is False, so a numerically perfect observation fails `contains` on dtype.
        return {
            key: np.clip(value, self.observation_space[key].low, self.observation_space[key].high)
            .astype(np.float32, copy=False)
            for key, value in obs.items()
        }

    def _constraint_row(self, visas: int, lead: int) -> np.ndarray:
        """What the agent KNOWS about the party's hard constraints.

        Each class contributes a "have I asked" bit and, only once asked, the answer. Before
        asking, the answer channel reads zero — which is indistinguishable from "no needs",
        and that ambiguity is precisely the thing the agent has to resolve.
        """
        asked = {k: (k in self.known) for k in ASKABLE}
        return np.array(
            [
                float(asked[ConstraintClass.ACCESSIBILITY]),
                float(bool(self.needs.access)) if asked[ConstraintClass.ACCESSIBILITY] else 0.0,
                float(self.needs.mobility) if asked[ConstraintClass.ACCESSIBILITY] else 0.0,
                len(self.needs.access) / 5.0 if asked[ConstraintClass.ACCESSIBILITY] else 0.0,
                float(asked[ConstraintClass.CAPACITY]),
                float(self.needs.has_minor) if asked[ConstraintClass.CAPACITY] else 0.0,
                float(self.needs.needs_adjacent) if asked[ConstraintClass.CAPACITY] else 0.0,
                float(asked[ConstraintClass.DOCUMENTS]),
                min(1.0, visas / 2.0) if asked[ConstraintClass.DOCUMENTS] else 0.0,
                min(1.0, lead / 30.0) if asked[ConstraintClass.DOCUMENTS] else 0.0,
                float(asked[ConstraintClass.COMMITMENTS]),
                min(1.0, len(self.persona.fixed_commitments)) if asked[ConstraintClass.COMMITMENTS] else 0.0,
            ],
            dtype=np.float32,
        )

    def _route_rows(self) -> np.ndarray:
        rows = np.full((MAX_ROUTE_CITIES, 5), -1.0, dtype=np.float32)
        housed = {s.hotel.city for s in self.timeline.stays}
        for i, (city, nights) in enumerate(self.route[:MAX_ROUTE_CITIES]):
            record = self.cities[city]
            rows[i] = [
                nights / max(1, self.nights),
                1.0 if city in housed else 0.0,
                sum(1 for a, _ in self.timeline.activities if a.city is city) / 4.0,
                float(np.mean([d.price_index for d in record.districts])) / 2.0,
                1.0 if any(c.city is city for c in self.persona.fixed_commitments) else 0.0,
            ]
        return rows

    def _destination_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 6), -1.0, dtype=np.float32)
        for i, code in enumerate(self.destination_slate[:SLATE_K]):
            city = self.cities[code]
            districts = city.districts
            rows[i] = [
                float(np.mean([d.price_index for d in districts])) / 2.0,
                float(np.mean([d.cultural_density for d in districts])) / 9.0,
                float(np.mean([d.nightlife for d in districts])) / 9.0,
                float(np.mean([d.quiet for d in districts])) / 9.0,
                1.0 if any(c.city is code for c in self.persona.fixed_commitments) else 0.0,
                sum(1 for _, n in self.route if True) / max(1, self.nights),
            ]
        return rows

    def _flight_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 11), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        for i, f in enumerate(self.flight_slate[:SLATE_K]):
            rows[i] = [
                self._quoted.get(f.key, f.price) * self.needs.size / budget,
                f.duration_minutes / 900.0,
                f.stops / 2.0,
                f.on_time_rate,
                1.0 if f.refundable else 0.0,
                {CabinClass.ECONOMY: 0.0, CabinClass.PREMIUM_ECONOMY: 0.5, CabinClass.BUSINESS: 1.0}[f.cabin],
                {FareBrand.BASIC: 0.0, FareBrand.STANDARD: 0.5, FareBrand.FLEX: 1.0}[f.fare_brand],
                min(1.0, self._seats(f) / 10.0),
                (f.connection_buffer_minutes or 180) / 180.0,
                1.0 if f.is_self_transfer else 0.0,
                min(1.0, f.adjacent_seats_available / 6.0),
            ]
        return rows

    def _ground_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 8), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        modes = list(TransportMode)
        for i, leg in enumerate(self.ground_slate[:SLATE_K]):
            rows[i] = [
                self._price_now(leg.key, leg.price) * self.needs.size / budget,
                min(1.0, leg.duration_minutes / 900.0),
                leg.on_time_rate,
                modes.index(leg.mode) / max(1, len(modes) - 1),
                1.0 if leg.step_free else 0.0,
                1.0 if leg.weather_sensitive else 0.0,
                min(1.0, self._ground_seats(leg) / 20.0),
                1.0 if leg.refundable else 0.0,
            ]
        return rows

    def _hotel_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 12), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        types = list(LodgingType)
        for i, (h, o) in enumerate(self.hotel_slate[:SLATE_K]):
            d = self.districts[h.district]
            rows[i] = [
                self._unit_cost(o) * self.nights / budget,
                h.star_rating / 5.0,
                h.guest_rating / 10.0,
                # Shown so the agent CAN learn to discount a 9.8 from 11 reviews.
                min(1.0, h.review_count / 2000.0),
                d.walkability / 9.0,
                d.quiet / 9.0,
                min(1.0, h.minutes_to_station / 25.0),
                1.0 if o.rate_plan is RatePlan.FREE_CANCELLATION else 0.0,
                min(1.0, self._rooms(o) / 8.0),
                types.index(h.lodging_type) / max(1, len(types) - 1),
                min(1.0, units_needed(self.needs, o) / 3.0),
                1.0 if h.check_in_until_hour >= 24 else h.check_in_until_hour / 24.0,
            ]
        return rows

    def _activity_rows(self) -> np.ndarray:
        rows = np.full((SLATE_K, 9), -1.0, dtype=np.float32)
        budget = max(1, self.persona.budget_cents)
        for i, a in enumerate(self.activity_slate[:SLATE_K]):
            d = self.districts[a.district]
            rows[i] = [
                a.price_per_person * self.needs.size / budget,
                a.duration_minutes / 240.0,
                a.rating / 10.0,
                _CATEGORIES.index(a.category) / max(1, len(_CATEGORIES) - 1),
                a.physical_intensity / 9.0,
                d.cultural_density / 9.0,
                1.0 if a.booking_required else 0.0,
                1.0 if a.accessible else 0.0,
                1.0 if a.weather_dependent else 0.0,
            ]
        return rows

    def _reward_inputs(self) -> RewardInputs:
        complete = self.timeline.is_complete()
        return RewardInputs(
            persona=self.persona,
            realized_features=self._realized_features(),
            spent_cents=self.ledger.spent,
            budget_cents=self.persona.budget_cents,
            hard_needs_met=complete and self.ledger.balance >= 0 and self.constraint_failures == 0,
            itinerary_complete=complete,
            disruption_damage_cents=self.disruption_damage_cents,
            forfeited_activity_value=self.forfeited_value,
            client_walked=self.client_walked,
            turns_used=self.turn,
            redundant_searches=0,
        )

    def _info(self) -> dict[str, Any]:
        """Stable key set with stable dtypes on every step.

        Under vectorisation Gymnasium fixes a key's dtype from the first sub-env to emit it
        and zero-fills absences, so a key appearing only sometimes yields wrong batched logs.
        """
        return {
            "action_mask": self.action_masks(),
            "turn": int(self.turn),
            "patience": float(self.patience),
            "spent_cents": int(self.ledger.spent),
            "budget_cents": int(self.persona.budget_cents),
            "itinerary_complete": bool(self.timeline.is_complete()),
            "client_walked": bool(self.client_walked),
            "axes_elicited": float(self.belief[:, 1].mean()),
            "constraints_elicited": float(len(self.known) / len(ASKABLE)),
            "constraint_failures": int(self.constraint_failures),
            "disruption_damage_cents": int(self.disruption_damage_cents),
            "route_cities": int(len(self.route)),
            "regions": int(len(self.world.regions)),
            "phase": int(self.phase),
            "illegal_attempts": int(self.illegal_attempts),
            "stratum": float(self.difficulty),
        }

    # ---- render ---------------------------------------------------------------------------

    def render(self) -> str | None:
        """An ANSI dashboard of the whole engagement.

        Worth more than it looks: the timeline is the hard part of this environment, and a
        table of segments with their transfer gaps is the only way to see at a glance whether
        a plan is coherent. Most of the v1 bugs that survived `check_env` were visible in one
        glance at a rendering.
        """
        if self.render_mode != "ansi":
            return None

        bar = "─" * 78
        regions = " + ".join(r.value for r in self.world.regions)
        pat = self.patience / max(1, self.persona.initial_patience)
        lines = [
            bar,
            f" turn {self.turn:>2}/{MAX_TURNS}   phase {['planning','travelling','done'][self.phase]:<10}"
            f" patience {'█' * int(max(0, pat) * 10):<10} {self.patience:5.1f}",
            f" {regions:<22} {self.nights} nights   party {self.needs.size}"
            f"   spent ${self.ledger.spent/100:,.0f} / ${self.persona.budget_cents/100:,.0f}",
            bar,
            f" CLIENT   {self.transcript[0][1]}",
            f" elicited preferences {self.belief[:,1].mean():.0%}"
            f"   hard constraints {len(self.known)}/{len(ASKABLE)}"
            f" [{', '.join(sorted(k.value for k in self.known)) or 'none asked'}]",
        ]

        hidden = []
        if self.needs.access:
            hidden.append("access: " + "/".join(sorted(n.value for n in self.needs.access)))
        if self.needs.has_minor:
            hidden.append(f"minors (youngest {self.needs.youngest})")
        if self.persona.fixed_commitments:
            hidden.append(f"fixed: {self.persona.fixed_commitments[0].label}")
        lines.append(f" (hidden truth: {'; '.join(hidden) or 'no hard constraints'})")

        route = " → ".join(f"{c.value}×{n}" for c, n in self.route) or "(no route yet)"
        lines += [bar, f" ROUTE    {route}"]

        segments = self.timeline.segments()
        if segments:
            lines.append(f" {'when (local)':<18}{'what':<34}{'where':<14}{'gap':>8}")
            previous: Segment | None = None
            for seg in segments:
                if seg.kind is SegmentKind.SLEEP:
                    continue
                offset = self.resolver.offset(seg.place_in.city)
                when = to_local(seg.start_utc, offset).strftime("%a %d %H:%M")
                gap = ""
                if previous is not None:
                    have = (seg.start_utc - previous.effective_end).total_seconds() / 60
                    need = self.resolver.transfer(previous.place_out, seg.place_in)
                    gap = "UNREACHABLE" if have < need else f"+{have - need:.0f}m"
                label = seg.label[:32]
                lines.append(f" {when:<18}{label:<34}{str(seg.place_in):<14}{gap:>8}")
                previous = seg

        violations = self.timeline.violations()
        if violations:
            lines.append(f" ⚠ {len(violations)} timeline violation(s): {violations[0].detail[:60]}")
        if self.events:
            lines.append(bar)
            for event in self.events[:6]:
                lines.append(f" • {event[:74]}")
        lines.append(bar)
        return "\n".join(lines)


gym.register(
    id="travel_agent/TravelAgent-v2",
    entry_point="travel_env.env_v2:TravelAgentEnvV2",
    max_episode_steps=MAX_TURNS,  # TimeLimit owns truncation
)
