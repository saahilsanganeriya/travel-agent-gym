"""The agent half of the tau-bench shape: an LLM driving the same `Discrete(89)` action space.

**The idea this file exists to carry: the action mask survives into the LLM path.** The tool
list handed to the model is rebuilt every turn from `env.action_masks()`, and an argument's
JSON-schema `enum` contains exactly the legal indices. An illegal action is therefore never
*offered*, so it cannot be chosen, so there is no "invalid tool call" failure mode to
retry-loop on and no illegal-action penalty to tune. It is the same guarantee `MaskablePPO`
gets — renegotiating the affordances every turn rather than validating a free-form answer
after the fact. Most LLM-agent harnesses expose a fixed tool list and rely on the model to
respect preconditions stated in prose; that is strictly weaker.

Second decision: the prompt is **rebuilt from scratch each turn** rather than appended to a
growing chat history. The environment state *is* the state — a 60-turn accumulated history
would pay for the same slate a dozen times and bury the current one in stale ones. What
carries across turns is the transcript with the client, because that is the part the
environment does not put in the observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from travel_env.actions_v2 import (
    SLATE_K,
    FlightPreset,
    GroundPreset,
    HotelSort,
    LodgingFilter,
    Verb,
    decode,
    encode,
    verb_slice,
)
from travel_env.constraints import ASKABLE
from travel_env.contracts import ActivityCategory, PreferenceAxis
from travel_env.llm import LLMClient, LLMError

# One tool per verb. Names are the model-facing API; the mapping back is exact and total.
TOOL_FOR_VERB: dict[Verb, str] = {
    Verb.ASK: "ask_preference",
    Verb.ASK_CONSTRAINTS: "ask_constraints",
    Verb.SEARCH_DESTINATIONS: "search_destinations",
    Verb.ADD_CITY: "add_night_in_city",
    Verb.DROP_CITY: "drop_last_night",
    Verb.SEARCH_FLIGHTS: "search_flights",
    Verb.SEARCH_GROUND: "search_ground_transport",
    Verb.SEARCH_HOTELS: "search_lodging",
    Verb.SEARCH_ACTIVITIES: "search_activities",
    Verb.SELECT_FLIGHT: "select_flight",
    Verb.SELECT_GROUND: "select_ground_leg",
    Verb.SELECT_HOTEL: "select_lodging",
    Verb.SELECT_ACTIVITY: "select_activity",
    Verb.PROPOSE: "propose_plan",
    Verb.BOOK: "book_plan",
    Verb.FINALIZE: "finalize",
    Verb.REBOOK: "rebook_flight",
}
VERB_FOR_TOOL: dict[str, Verb] = {name: verb for verb, name in TOOL_FOR_VERB.items()}

_VERB_HELP: dict[Verb, str] = {
    Verb.ASK: "Ask the client how much one preference axis matters to them.",
    Verb.ASK_CONSTRAINTS: (
        "Ask the client about a class of hard constraints. Until you ask, search results are "
        "NOT filtered for that class and you can book something the party cannot use."
    ),
    Verb.SEARCH_DESTINATIONS: "List candidate cities for the route.",
    Verb.ADD_CITY: "Add one night in a city from the destination slate (repeat to add more nights).",
    Verb.DROP_CITY: "Remove the last night from the route.",
    Verb.SEARCH_FLIGHTS: "Search the next unbooked long-haul leg with a filter preset.",
    Verb.SEARCH_GROUND: "Search surface transport for the next unbooked route hop.",
    Verb.SEARCH_HOTELS: "Search lodging in the first route city that has none.",
    Verb.SEARCH_ACTIVITIES: "Search activities in the route cities by category.",
    Verb.SELECT_FLIGHT: "Take a flight from the current flight slate, by row index.",
    Verb.SELECT_GROUND: "Take a ground leg from the current ground slate, by row index.",
    Verb.SELECT_HOTEL: "Take a lodging offer from the current lodging slate, by row index.",
    Verb.SELECT_ACTIVITY: "Take an activity from the current activity slate, by row index.",
    Verb.PROPOSE: "Show the client the assembled plan and hear their pushback. Do this before booking.",
    Verb.BOOK: "Commit money to the whole plan.",
    Verb.FINALIZE: "Close the engagement and run the trip.",
    Verb.REBOOK: "Recover a disrupted leg from the rebooking slate.",
}

_ARG_LABELS: dict[Verb, list[str]] = {
    Verb.ASK: [a.value for a in PreferenceAxis],
    Verb.ASK_CONSTRAINTS: [c.value for c in ASKABLE],
    Verb.SEARCH_FLIGHTS: [p.value for p in FlightPreset],
    Verb.SEARCH_GROUND: [p.value for p in GroundPreset],
    Verb.SEARCH_HOTELS: [
        f"{sort.value}+{lodging.value}" for sort in HotelSort for lodging in LodgingFilter
    ],
    Verb.SEARCH_ACTIVITIES: [c.value for c in ActivityCategory],
}

SYSTEM_PROMPT = (
    "You are an expert travel agent planning one trip for one client. You are paid on how "
    "well the finished trip matches what this client actually wants — not on how fast you "
    "book.\n\n"
    "What wins:\n"
    "1. INTERVIEW FIRST. The client's preferences and their party's hard constraints "
    "(accessibility, capacity, passports, fixed commitments) are hidden until you ask. Asking "
    "about a constraint class also switches on filtering for it in every later search.\n"
    "2. Build a route that uses every night, then book flights out and back, ground legs "
    "between cities, lodging in every city, and a few activities.\n"
    "3. Propose the plan to the client, then book it, then finalize.\n"
    "The client's patience is finite and searching burns more of it than talking does.\n\n"
    "You will be shown the current state and given exactly the tools that are legal right "
    "now. Call exactly one tool per turn. Never explain instead of acting."
)


def tools_from_mask(env: Any) -> list[dict[str, Any]]:
    """OpenAI tool schemas for exactly the actions that are legal this turn.

    A verb appears only if some action in its slice is legal, and its argument is an `enum`
    of the legal indices — so "select row 4" cannot be emitted when row 4 is sold out or
    unaffordable. This is the action mask, expressed in the only vocabulary a tool-calling
    model can be constrained by.
    """
    mask = np.asarray(env.action_masks() if hasattr(env, "action_masks") else env.unwrapped.action_masks())
    tools: list[dict[str, Any]] = []

    for verb, name in TOOL_FOR_VERB.items():
        span = verb_slice(verb)
        legal = [int(i) for i in np.flatnonzero(mask[span])]
        if not legal:
            continue

        properties: dict[str, Any] = {}
        required: list[str] = []
        description = _VERB_HELP[verb]

        if span.stop - span.start > 1:
            labels = _ARG_LABELS.get(verb)
            if labels:
                description += " Options: " + ", ".join(f"{i}={labels[i]}" for i in legal) + "."
            else:
                description += " Argument is a row index into the slate shown in the state."
            properties["arg"] = {
                "type": "integer",
                "enum": legal,
                "description": "Choose one of the listed legal values.",
            }
            required.append("arg")

        if verb in (Verb.ASK, Verb.ASK_CONSTRAINTS):
            # The agent writes its own utterance; the client simulator answers that text.
            # Without this the dialogue would be a canned rendering of the action id, and the
            # LLM-vs-LLM conversation would not be a conversation.
            properties["message"] = {
                "type": "string",
                "description": "What you actually say to the client, in your own words.",
            }

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def tool_call_to_action(name: str, arguments: dict[str, Any]) -> int:
    """Tool call -> action id. Raises on anything it cannot map exactly."""
    verb = VERB_FOR_TOOL.get(name)
    if verb is None:
        raise ValueError(f"unknown tool {name!r}")
    raw = arguments.get("arg", 0)
    try:
        arg = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-integer arg {raw!r} for {name}") from exc
    return encode(verb, arg)


def observation_to_prompt(env: Any) -> str:
    """The state an agent could actually read: money, patience, route, bookings, slates.

    Slates are rendered as numbered rows with their real attributes, because the tool
    argument is a row index — a model choosing blind on index is choosing at random.
    """
    core = env.unwrapped
    spent, budget = core.ledger.spent, core.persona.budget_cents
    patience_pct = 100.0 * core.patience / max(1, core.persona.initial_patience)
    allocated = sum(n for _, n in core.route)
    route = " -> ".join(f"{c.value} x{n} nights" for c, n in core.route) or "(empty)"

    known = ", ".join(sorted(k.value for k in core.known)) or "none yet"
    beliefs = ", ".join(
        f"{axis.value}={core.belief[i, 0]:.2f}(conf {core.belief[i, 1]:.2f})"
        for i, axis in enumerate(PreferenceAxis)
    )

    timeline = core.timeline
    lines = [
        f"TURN {core.turn} | patience {patience_pct:.0f}% | phase "
        f"{['planning', 'travelling', 'done'][core.phase]}",
        f"BUDGET ${budget / 100:,.0f} | spent ${spent / 100:,.0f} | "
        f"remaining ${(budget - spent) / 100:,.0f} | party of {core.needs.size}",
        f"ROUTE {route}   ({allocated}/{core.nights} nights allocated)",
        f"BOOKED outbound={'yes' if timeline.outbound else 'no'} "
        f"inbound={'yes' if timeline.inbound else 'no'} "
        f"ground_legs={len(timeline.ground)} stays={len(timeline.stays)} "
        f"activities={len(timeline.activities)} complete={timeline.is_complete()}",
        f"CONSTRAINT CLASSES ASKED: {known}",
        f"PREFERENCE BELIEFS (0-1, confidence 0 means you have not asked): {beliefs}",
        "",
        "CONVERSATION SO FAR:",
    ]
    # Recent transcript only: the older turns are already reflected in the belief row above.
    for who, what in core.transcript[-8:]:
        lines.append(f"  {who.upper()}: {what}")

    lines.append("")
    lines.extend(_slates(core))
    return "\n".join(lines)


def _slates(core: Any) -> list[str]:
    out: list[str] = []

    if core.destination_slate:
        out.append("DESTINATION SLATE (arg for add_night_in_city):")
        out += [f"  [{i}] {c.value}" for i, c in enumerate(core.destination_slate[:SLATE_K])]

    if core.flight_slate:
        out.append("FLIGHT SLATE (per-person price):")
        for i, f in enumerate(core.flight_slate[:SLATE_K]):
            out.append(
                f"  [{i}] {f.carrier}{f.flight_number} {f.origin}->{f.destination} "
                f"${f.price / 100:,.0f} {f.duration_minutes}min stops={f.stops} "
                f"on_time={f.on_time_rate:.2f} seats={f.seats_remaining} "
                f"refundable={f.refundable} self_transfer={f.is_self_transfer}"
            )

    if core.ground_slate:
        out.append("GROUND SLATE (per-person price):")
        for i, g in enumerate(core.ground_slate[:SLATE_K]):
            out.append(
                f"  [{i}] {g.mode.value} {g.operator} {g.origin_city.value}->"
                f"{g.destination_city.value} ${g.price / 100:,.0f} {g.duration_minutes}min "
                f"on_time={g.on_time_rate:.2f} step_free={g.step_free} "
                f"weather_sensitive={g.weather_sensitive}"
            )

    if core.hotel_slate:
        out.append("LODGING SLATE (nightly price per unit):")
        for i, (hotel, offer) in enumerate(core.hotel_slate[:SLATE_K]):
            access = "/".join(sorted(n.value for n in hotel.accessibility)) or "none"
            out.append(
                f"  [{i}] {hotel.name} ({hotel.lodging_type.value}, {hotel.city.value}) "
                f"${offer.price_per_night / 100:,.0f}/night sleeps={offer.max_occupancy} "
                f"rating={hotel.guest_rating:.1f} station={hotel.minutes_to_station}min "
                f"board={offer.board.value} access={access} rooms_left={offer.rooms_remaining}"
            )

    if core.activity_slate:
        out.append("ACTIVITY SLATE (per-person price):")
        for i, a in enumerate(core.activity_slate[:SLATE_K]):
            out.append(
                f"  [{i}] {a.name} ({a.category.value}, {a.city.value}) "
                f"${a.price_per_person / 100:,.0f} rating={a.rating:.1f} "
                f"{a.duration_minutes}min intensity={a.physical_intensity}"
            )

    if core.rebook_slate:
        out.append("REBOOKING SLATE:")
        for i, f in enumerate(core.rebook_slate[:SLATE_K]):
            out.append(
                f"  [{i}] {f.carrier}{f.flight_number} ${f.price / 100:,.0f} "
                f"{f.duration_minutes}min on_time={f.on_time_rate:.2f}"
            )
    return out or ["(no slates loaded — search first)"]


class LLMAgent:
    """Policy with the baseline `act(obs, info, env) -> int` signature, backed by an LLM.

    Parse failures are counted and reported, never hidden: an agent that silently falls back
    to the first legal action on half its turns is a heuristic wearing an LLM's price tag.
    """

    name = "llm"

    def __init__(self, llm: LLMClient, *, max_tokens: int = 900) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.steps = 0
        self.parse_failures = 0
        self.call_failures = 0
        self.last_utterance: str | None = None

    def reset(self) -> None:
        self.last_utterance = None
        self.llm.begin_episode()

    def act(self, obs: dict, info: dict, env: Any) -> int:
        self.steps += 1
        self.last_utterance = None
        mask = np.asarray(info["action_mask"])
        legal = np.flatnonzero(mask)
        fallback = int(legal[0]) if legal.size else 0

        tools = tools_from_mask(env)
        if not tools:
            return fallback

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": observation_to_prompt(env)
                + "\n\nCall exactly one tool now. Only legal tools are listed.",
            },
        ]
        try:
            response = self.llm.chat(
                messages, tools=tools, tool_choice="required", max_tokens=self.max_tokens
            )
        except LLMError:
            self.call_failures += 1
            self.parse_failures += 1
            return fallback

        for call in response.tool_calls:
            try:
                action = tool_call_to_action(call.name, call.arguments)
            except (ValueError, KeyError):
                continue
            if not mask[action]:
                continue  # masked out between schema build and reply — treat as a miss
            message = call.arguments.get("message")
            if decode(action).verb in (Verb.ASK, Verb.ASK_CONSTRAINTS):
                self.last_utterance = message if isinstance(message, str) and message else None
            return action

        self.parse_failures += 1
        return fallback
