"""One HTTP client for every LLM call in this repository, and the rate governor in front of it.

Three decisions worth defending:

* **stdlib `urllib`, no new dependency.** The training path must stay installable with nothing
  but `gymnasium`/`numpy`/`torch`. A benchmark adapter is not worth making `requests` a hard
  requirement of an RL environment, and `/chat/completions` is a single POST of JSON.
* **The rate limit lives in a token bucket, not in `sleep()` between calls.** Two simulators
  (agent and client) share one gateway quota; a per-call sleep would either serialise them or
  burst past the limit when they interleave. A bucket with a burst allowance is the standard
  shape for "N per minute, but a short burst is fine", and it is shareable across threads.
* **Cost is measured, not estimated, when the gateway reports it.** The gateway response carries
  `usage.cost` in USD; the per-token table below is only the fallback for gateways that do
  not. An eval that reports a made-up dollar figure is worse than one that reports none.

Nothing here is imported by `env_v2.py` on the `step()` path — the masked-`Discrete`
environment stays fully offline.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from travel_env import config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fallback prices, USD per 1M tokens (prompt, completion). Only consulted when the gateway
# does not return `usage.cost`; a wrong entry here shows up as a wrong dollar figure, so the
# reported number always says which source it came from.
_FALLBACK_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
}
_UNKNOWN_MODEL_PRICE = (0.50, 1.50)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    """Any failure that leaves the caller without a completion."""


class LLMBudgetExhausted(LLMError):
    """The per-episode call budget is spent. Separate type so a policy can degrade rather
    than crash a 100-episode eval on one runaway conversation."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ToolCall(FrozenModel):
    name: str
    arguments: dict[str, Any]


class ChatUsage(FrozenModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_is_reported: bool = False
    """False means `cost_usd` came from the local price table, not from the gateway."""


class ChatResponse(FrozenModel):
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ChatUsage = ChatUsage()


class TokenBucket:
    """Classic token bucket: `burst` tokens of slack, refilled at `rate_per_min`.

    Thread-safe because the agent simulator and the client simulator can share one instance
    and, under a threaded eval, would otherwise both read a stale token count and both go.
    `acquire()` blocks; there is no non-blocking variant on purpose — every caller here wants
    the call to happen eventually, and a dropped LLM turn is a silently corrupted episode.
    """

    def __init__(self, rate_per_min: float, burst: int = 4) -> None:
        if rate_per_min <= 0 or burst < 1:
            raise ValueError("rate_per_min must be > 0 and burst >= 1")
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available. Returns the seconds spent waiting."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate_per_sec
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                # Sleep outside the lock, or a waiter blocks the refill for everyone else.
                deficit = tokens - self._tokens
                delay = deficit / self.rate_per_sec
            time.sleep(delay)
            waited += delay


class LLMClient:
    """An OpenAI-compatible `/chat/completions` caller with retries, budget and cost tracking.

    Defaults to the gateway named by `LLM_BASE_URL` and falls back to OpenRouter only when
    `LLM_API_KEY` is absent.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        rate_per_min: float = 120.0,
        burst: int = 6,
        max_calls_per_episode: int = 120,
        bucket: TokenBucket | None = None,
        temperature: float | None = None,
    ) -> None:
        key, resolved_base, provider = _resolve_credentials(api_key, base_url)
        self.model = model
        self.base_url = resolved_base.rstrip("/")
        self.provider = provider
        self._key = key  # never logged, never rendered
        self.timeout_s = timeout_s
        self.max_calls_per_episode = max_calls_per_episode
        self.temperature = temperature
        self.bucket = bucket or TokenBucket(rate_per_min, burst)

        self.calls = 0
        self.calls_this_episode = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.cost_is_reported = True

    # ---- budget ------------------------------------------------------------------------

    def begin_episode(self) -> None:
        self.calls_this_episode = 0

    # ---- the call ----------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "required", "none"] | None = None,
        max_tokens: int = 512,
    ) -> ChatResponse:
        if self.calls_this_episode >= self.max_calls_per_episode:
            raise LLMBudgetExhausted(
                f"{self.max_calls_per_episode} calls already made this episode"
            )

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        # The gpt-5 family rejects `max_tokens`; everything else predates
        # `max_completion_tokens`. One branch is cheaper than a capability probe per run.
        if self.model.startswith("openai/gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if self.temperature is not None and not self.model.startswith("openai/gpt-5"):
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        body = self._post("/chat/completions", payload)
        self.calls += 1
        self.calls_this_episode += 1
        return self._parse(body)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                # OpenRouter attributes traffic from these; other gateways ignore them.
                "HTTP-Referer": "https://github.com/saahilsanganeriya/travel-agent-gym",
                "X-Title": "travel-agent-env",
            },
            method="POST",
        )

        last: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            self.bucket.acquire()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code not in _RETRYABLE_STATUS or attempt == _MAX_ATTEMPTS - 1:
                    raise LLMError(f"HTTP {exc.code} from {self.provider}: {detail}") from exc
                # Honour Retry-After when the server sets it — guessing a shorter backoff on
                # a 429 is how you get rate-limited for longer.
                after = exc.headers.get("Retry-After") if exc.headers else None
                delay = _parse_retry_after(after)
                last = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise LLMError(f"{type(exc).__name__} calling {self.provider}: {exc}") from exc
                delay = None
                last = exc

            self.failures += 1
            # Jitter, not bare exponential: without it, concurrent simulators retry in
            # lockstep and re-create the burst that triggered the 429.
            backoff = delay if delay is not None else 2.0**attempt
            time.sleep(backoff * (0.6 + 0.8 * random.random()))

        raise LLMError(f"exhausted {_MAX_ATTEMPTS} attempts: {last}")

    def _parse(self, body: dict[str, Any]) -> ChatResponse:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed completion: {str(body)[:200]}") from exc

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or ():
            function = raw.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                # A malformed argument blob is a parse failure for the caller to count, not
                # an exception — the caller has a legal fallback and the rate is a metric.
                arguments = {}
            calls.append(ToolCall(name=name, arguments=arguments if isinstance(arguments, dict) else {}))

        usage = self._account(body.get("usage") or {})
        return ChatResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=tuple(calls),
            usage=usage,
        )

    def _account(self, usage: dict[str, Any]) -> ChatUsage:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        reported = usage.get("cost")
        if isinstance(reported, int | float):
            cost, is_reported = float(reported), True
        else:
            price_in, price_out = _FALLBACK_PRICES.get(self.model, _UNKNOWN_MODEL_PRICE)
            cost = (prompt * price_in + completion * price_out) / 1_000_000
            is_reported = False

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost
        self.cost_is_reported = self.cost_is_reported and is_reported
        return ChatUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
            cost_is_reported=is_reported,
        )


def _resolve_credentials(
    api_key: str | None, base_url: str | None
) -> tuple[str, str, str]:
    """Gateway first, OpenRouter as the fallback. Raises naming `.env.example`, never the key."""
    if api_key:
        return api_key, base_url or config.llm_base_url(), "explicit"

    gateway = config.llm_key()
    if gateway and (base_url is None or base_url == config.llm_base_url()):
        return gateway, base_url or config.llm_base_url(), "gateway"

    openrouter = config.get("OPENROUTER_API_KEY")
    if openrouter:
        return openrouter, base_url or OPENROUTER_BASE_URL, "openrouter"
    if gateway:
        return gateway, base_url or config.llm_base_url(), "gateway"

    raise LLMError(
        "no LLM credential found. Set LLM_API_KEY (or OPENROUTER_API_KEY) in <repo>/.env — "
        "see .env.example for the shape. The scripted client and the heuristic agent run "
        "without any key: `python -m eval.llm_eval --dry-run`."
    )


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; the exponential path is close enough.
