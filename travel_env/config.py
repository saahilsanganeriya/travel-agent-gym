"""One place that knows where credentials live: `<repo>/.env`.

Two rules, both load-bearing:

* **Nothing here is called inside `step()`.** The masked-`Discrete` environment is fully
  offline and deterministic and never reads a key. These are for one-off build scripts
  (`fetch_transfer_matrix.py`) and for the optional LLM configurations.
* **A missing key is not an error.** Every caller has a working fallback — `geo.py` uses
  modelled transit profiles when the Distance Matrix result is absent, and the scripted
  client is what runs when there is no LLM key. Raising here would make the repository
  unrunnable for anyone who clones it, which is the opposite of what a shared repository wants.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


@cache
def _dotenv() -> dict[str, str]:
    """Parse `<repo>/.env`. Cached: the file does not change within a process."""
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            values[key.strip()] = value
    return values


def get(*names: str, default: str | None = None) -> str | None:
    """First of `names` that is set, checking the real environment before the file.

    Several names because the same secret is spelled differently by different providers —
    `GOOGLE_MAPS_KEY` and `GOOGLE_MAPS_API_KEY` are the same key, and a caller should not
    have to care which one the file happens to use.
    """
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    loaded = _dotenv()
    for name in names:
        if loaded.get(name):
            return loaded[name]
    return default


def google_maps_key() -> str | None:
    return get("GOOGLE_MAPS_API_KEY", "GOOGLE_MAPS_KEY")


def llm_key() -> str | None:
    return get("LLM_API_KEY")


def llm_base_url() -> str:
    return get("LLM_BASE_URL", default="https://openrouter.ai/api/v1") or ""


def describe() -> str:
    """Which credentials are present, for a startup banner. Never prints a value."""
    present = [
        name
        for name, value in (
            ("google_maps", google_maps_key()),
            ("llm", llm_key()),
            ("openrouter", get("OPENROUTER_API_KEY")),
        )
        if value
    ]
    return ", ".join(present) if present else "none (offline defaults in use)"
