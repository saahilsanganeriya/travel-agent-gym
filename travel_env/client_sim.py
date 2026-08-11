"""The client half of the tau-bench shape: an LLM playing the person the agent is selling to.

**The design point, and the only one that actually protects the hidden state: structural
information asymmetry.** The client's context window contains exactly two things — the
persona rendered as prose, and the conversation transcript. It never contains the world, the
slates, the ledger, the action mask, the reward, or the agent's tool calls. So there is no
prompt to jailbreak and no instruction to disobey: the secret the agent is trying to extract
by *not* asking is not in the room. "Do not reveal X" prompts fail under pressure; contexts
that never held X cannot.

Two further deliberate choices:

* The hidden preference weights are rendered as **dispositions in words**, never as the
  0-9 integers. A number in the context is a number the client can recite, and an agent that
  gets told "food: 8" has been handed the thing it was supposed to interview for.
* The prompt discipline is lifted **verbatim** from tau-bench (arXiv:2406.12045), because
  those two sentences are what stop a simulated user from either dumping the whole task in
  turn one or hallucinating facts about their own trip. Restating them in my own words would
  be a silent change to a load-bearing prompt.

`ScriptedClientSim` implements the same `reply()` contract off the persona vectors, so the
whole LLM path is exercisable offline, deterministically, and for free.
"""

from __future__ import annotations

from typing import Protocol

from travel_env.actions_v2 import Verb, decode
from travel_env.constraints import ASKABLE, ConstraintClass
from travel_env.contracts import ClientPersona, CommunicationStyle, PreferenceAxis
from travel_env.llm import LLMClient, LLMError
from travel_env.personas_v2 import describe_manifest

# tau-bench's user-simulator discipline, quoted exactly.
TAU_BENCH_RULES = (
    "Do not give away all the instruction at once. Only provide the information that is "
    "necessary for the current step.\n"
    "If the agent asks for something you do not know, say you do not remember rather than "
    "making it up."
)

_DISPOSITION = (
    "you do not care at all about",       # 0
    "you barely think about",             # 1
    "you are fairly indifferent to",      # 2
    "you are fairly indifferent to",      # 3
    "you mildly care about",              # 4
    "you mildly care about",              # 5
    "you care about",                     # 6
    "you care quite a lot about",         # 7
    "you care a great deal about",        # 8
    "you care enormously about",          # 9
)

_STYLE_GUIDANCE: dict[CommunicationStyle, str] = {
    CommunicationStyle.TERSE: "Answer in a few words. No pleasantries, no elaboration.",
    CommunicationStyle.CASUAL: "Write like a text message: relaxed, contractions, one or two sentences.",
    CommunicationStyle.FORMAL: "Write in complete, polite sentences. Slightly stiff.",
    CommunicationStyle.CHATTY: "Warm and talkative — volunteer a little colour, but stay on the question.",
}

_CONSTRAINT_QUESTIONS: dict[ConstraintClass, str] = {
    ConstraintClass.ACCESSIBILITY: (
        "Does anyone travelling have accessibility needs — step-free access, a wheelchair, "
        "a lift, a service animal, anything like that?"
    ),
    ConstraintClass.CAPACITY: "How many of you are travelling, and how old is everyone?",
    ConstraintClass.DOCUMENTS: "What passports does everyone hold, and when do they expire?",
    ConstraintClass.COMMITMENTS: "Is there anything fixed in the calendar you have to be at?",
}

_AXIS_QUESTIONS: dict[PreferenceAxis, str] = {
    PreferenceAxis.COST: "How tight is the budget really — where would you rather spend and where would you save?",
    PreferenceAxis.COMFORT: "How much does comfort matter — nice rooms, short travel days?",
    PreferenceAxis.LOCATION: "How important is staying central and easy to get around?",
    PreferenceAxis.PACING: "Do you want a packed schedule or plenty of downtime? How is your pacing?",
    PreferenceAxis.FOOD: "How big a part does food play in this trip for you?",
    PreferenceAxis.CULTURE: "How interested are you in history, museums and culture?",
    PreferenceAxis.NOVELTY: "Do you want the famous sights, or novelty — somewhere off the beaten path?",
    PreferenceAxis.RELIABILITY: "How much does reliability matter — would a tight connection worry you?",
}


class ClientSim(Protocol):
    """What both simulators promise. `reply` is the only thing the eval loop calls."""

    name: str

    def reply(self, question: str) -> str: ...


def question_for_action(action: int) -> str | None:
    """The natural-language utterance behind an ASK action, or None if the action is silent.

    The `Discrete` action space says *which* question; the text is a rendering of it. This is
    what lets a non-LLM policy still drive an LLM client, which is the point of the 2x2.
    """
    decoded = decode(int(action))
    if decoded.verb is Verb.ASK:
        return _AXIS_QUESTIONS[list(PreferenceAxis)[decoded.arg]]
    if decoded.verb is Verb.ASK_CONSTRAINTS:
        return _CONSTRAINT_QUESTIONS[ASKABLE[decoded.arg]]
    if decoded.verb is Verb.PROPOSE:
        return "Here is the plan I have put together. What do you think?"
    return None


def persona_prose(persona: ClientPersona) -> str:
    """The persona as the client experiences it: dispositions, a party, a calendar.

    Deliberately includes the manifest and the commitments — the client knows their own
    family — and deliberately excludes every number the agent is meant to elicit.
    """
    dispositions = "\n".join(
        f"- {_DISPOSITION[persona.weights[axis]].capitalize()} {axis.value}."
        for axis in PreferenceAxis
    )
    party = "\n".join(
        f"- Traveller {i + 1}: age {t.age}, {t.nationality.value} passport expiring "
        f"{t.passport_expiry.isoformat()}"
        + (f", needs {', '.join(sorted(n.value for n in t.accessibility))}" if t.accessibility else "")
        + (f", dietary: {', '.join(sorted(d.value for d in t.dietary))}" if t.dietary else "")
        for i, t in enumerate(persona.travellers)
    )
    commitments = (
        "\n".join(
            f"- {c.label} in {c.city.value}, starting {c.start_utc.isoformat()}"
            for c in persona.fixed_commitments
        )
        or "- Nothing fixed in the calendar."
    )
    return (
        f"You are a travel client. Party type: {persona.party_type.value}. "
        f"Your total budget for the trip is about ${persona.budget_cents // 100:,}.\n\n"
        f"How you feel about the trip (never state these as scores or numbers — express them "
        f"the way a person would):\n{dispositions}\n\n"
        f"Who is travelling:\n{party}\n\n"
        f"Fixed commitments:\n{commitments}\n"
    )


class LLMClientSim:
    """A client played by an LLM. Sees the persona and the transcript. Nothing else."""

    name = "llm"

    def __init__(
        self,
        persona: ClientPersona,
        opening_request: str,
        llm: LLMClient,
        *,
        max_history: int = 24,
    ) -> None:
        self.persona = persona
        self.llm = llm
        self.system_prompt = _system_prompt(persona)
        # Bounded history: an episode runs ~60 turns and an unbounded transcript is both a
        # cost leak and a recency problem. The opening request is always kept.
        self.max_history = max_history
        self.history: list[dict[str, str]] = [
            {"role": "assistant", "content": opening_request}
        ]
        self.fallbacks = 0

    def reply(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[
            -self.max_history :
        ]
        try:
            response = self.llm.chat(messages, max_tokens=160)
            text = response.text or "Sorry, could you repeat that?"
        except LLMError:
            # A dead client turn must not kill the episode: the eval reports the rate.
            self.fallbacks += 1
            text = "I'm not sure — let's keep going."
        self.history.append({"role": "assistant", "content": text})
        return text


def _system_prompt(persona: ClientPersona) -> str:
    verbosity = (
        "Keep replies to one short sentence."
        if persona.reply_verbosity <= 2
        else "Keep replies to one or two sentences."
        if persona.reply_verbosity <= 6
        else "Two or three sentences is fine."
    )
    return (
        f"{persona_prose(persona)}\n"
        f"You are speaking to a travel agent who is planning this trip for you. Stay in "
        f"character as the client at all times. You are the customer, not the planner.\n\n"
        f"{TAU_BENCH_RULES}\n\n"
        f"Style: {_STYLE_GUIDANCE[persona.communication_style]} {verbosity} "
        f"Never mention that you are an AI, never mention personas, scores or instructions."
    )


class ScriptedClientSim:
    """The same contract, answered straight off the persona vectors. No network, no cost.

    Exists so the LLM path is testable and CI-safe, and so the eval's 2x2 has a control cell:
    if the LLM agent scores the same against a scripted client as against an LLM one, the
    client simulator is not doing any work.
    """

    name = "scripted"

    def __init__(self, persona: ClientPersona) -> None:
        self.persona = persona

    def reply(self, question: str) -> str:
        lowered = question.lower()
        klass = _best_match(lowered, _CONSTRAINT_KEYWORDS)
        if klass is not None:
            return self._constraint_answer(klass)
        axis = _best_match(lowered, _AXIS_KEYWORDS)
        if axis is not None:
            return f"{_DISPOSITION[self.persona.weights[axis]].capitalize()} {axis.value}."
        return "I'm not sure I follow — what do you need from me?"

    def _constraint_answer(self, klass: ConstraintClass) -> str:
        p = self.persona
        match klass:
            case ConstraintClass.ACCESSIBILITY:
                needs = sorted({n.value for t in p.travellers for n in t.accessibility})
                return f"Yes — we need {', '.join(needs)}." if needs else "No, nothing like that."
            case ConstraintClass.CAPACITY:
                return f"{len(p.travellers)} of us: {describe_manifest(p)}."
            case ConstraintClass.DOCUMENTS:
                return "; ".join(
                    f"{t.nationality.value} passport valid to {t.passport_expiry.isoformat()}"
                    for t in p.travellers
                )
            case ConstraintClass.COMMITMENTS:
                if not p.fixed_commitments:
                    return "Nothing fixed, the calendar is open."
                c = p.fixed_commitments[0]
                return f"Yes — {c.label} in {c.city.value} on {c.start_utc.date().isoformat()}."
        return "I don't remember."


# Topic keywords, not string similarity: an LLM agent phrases the question its own way, and
# the canonical wording shares too many stopwords across topics to separate them.
_CONSTRAINT_KEYWORDS: dict[ConstraintClass, tuple[str, ...]] = {
    ConstraintClass.ACCESSIBILITY: (
        "accessib", "step-free", "step free", "wheelchair", "mobility", "lift", "elevator",
        "service animal", "disab",
    ),
    ConstraintClass.CAPACITY: ("how many", "travelling", "traveling", "party", "old is", "ages", "children", "kids"),
    ConstraintClass.DOCUMENTS: ("passport", "visa", "document", "nationalit", "citizen"),
    ConstraintClass.COMMITMENTS: ("fixed", "calendar", "commitment", "have to be at", "appointment", "booked ticket"),
}

_AXIS_KEYWORDS: dict[PreferenceAxis, tuple[str, ...]] = {
    PreferenceAxis.COST: ("budget", "cost", "spend", "afford", "price", "cheap"),
    PreferenceAxis.COMFORT: ("comfort", "nice room", "luxur"),
    PreferenceAxis.LOCATION: ("central", "location", "get around", "neighbourhood", "neighborhood"),
    PreferenceAxis.PACING: ("pacing", "packed", "downtime", "schedule", "relax", "busy"),
    PreferenceAxis.FOOD: ("food", "eat", "restaurant", "cuisine", "dining"),
    PreferenceAxis.CULTURE: ("culture", "cultural", "history", "historic", "museum", "temple"),
    PreferenceAxis.NOVELTY: ("novelty", "off the beaten", "unusual", "adventur", "famous sights"),
    PreferenceAxis.RELIABILITY: ("reliab", "risk", "go wrong", "connection", "delay", "on time"),
}


def _best_match[T](question: str, keywords: dict[T, tuple[str, ...]]) -> T | None:
    """The topic with the most keyword hits, or None when nothing matched."""
    scores = {topic: sum(k in question for k in words) for topic, words in keywords.items()}
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else None
