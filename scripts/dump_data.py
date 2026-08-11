"""Dump a sample world, sample personas, and a statistics report into `data/`.

The quality of the synthetic data is itself a signal. A signal a reader
has to run code to see is not one, so this writes inspectable artifacts:

  data/sample_world.json    one complete world (seed 7), every field, no truncation
  data/sample_personas.json ten clients INCLUDING their hidden weights, so a reader can
                            verify the weights really are hidden from the observation
  data/world_stats.md       the correlations the world is supposed to exhibit, measured
                            over 200 worlds rather than asserted
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from travel_env.personas import draw_persona, normalized_weights, opening_request
from travel_env.world import generate_world

DATA = Path("data")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def dump_world(seed: int = 7) -> dict:
    world = generate_world(seed)
    payload = json.loads(world.model_dump_json())
    DATA.mkdir(exist_ok=True)
    (DATA / "sample_world.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


def dump_personas(n: int = 10, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        difficulty = (i % 5) / 4.0
        p = draw_persona(rng, difficulty=difficulty)
        rows.append(
            {
                "difficulty": round(difficulty, 2),
                "what_the_agent_sees": {
                    "opening_request": opening_request(p, 7),
                    "stated_axes": sorted(a.value for a in p.stated_axes),
                },
                "hidden_from_the_agent": {
                    "weights_0_to_9": {a.value: v for a, v in p.weights.items()},
                    "weights_normalised": {
                        a.value: round(v, 4) for a, v in normalized_weights(p).items()
                    },
                },
                "behavioural": {
                    "age_band": p.age_band.value,
                    "party_type": p.party_type.value,
                    "party_size": p.party_size,
                    "communication_style": p.communication_style.value,
                    "budget_usd": round(p.budget_cents / 100, 2),
                    "budget_flexibility": round(p.budget_flexibility, 3),
                    "initial_patience": p.initial_patience,
                    "patience_decay_per_turn": round(p.patience_decay_per_turn, 3),
                    "tech_savvy": p.tech_savvy,
                    "risk_tolerance": p.risk_tolerance,
                    "reply_verbosity": p.reply_verbosity,
                },
            }
        )
    (DATA / "sample_personas.json").write_text(json.dumps(rows, indent=2))


def dump_stats(n_worlds: int = 200) -> None:
    """Measure the correlations the world is *supposed* to exhibit.

    A world can satisfy every invariant and still be incoherent because its statistics are
    wrong. These are the properties that make the tradeoffs real, so they are measured
    rather than asserted.
    """
    price, stars, guest, reviews, err, pidx = [], [], [], [], [], []
    fprice, ontime, seats = [], [], []

    for s in range(n_worlds):
        w = generate_world(s)
        d = w.district_index()
        hotels = {h.key: h for h in w.hotels}
        for o in w.room_offers[:60]:
            h = hotels[o.hotel]
            price.append(o.price_per_night)
            stars.append(h.star_rating)
            guest.append(h.guest_rating)
            reviews.append(h.review_count)
            err.append(abs(h.guest_rating - (5.6 + 0.72 * h.star_rating)))
            pidx.append(d[h.district].price_index)
        for f in w.flights[:120]:
            fprice.append(f.price)
            ontime.append(f.on_time_rate)
            seats.append(f.seats_remaining)

    lo = [e for e, r in zip(err, reviews, strict=True) if r < 60]
    hi = [e for e, r in zip(err, reviews, strict=True) if r > 800]

    lines = [
        "# Measured world statistics",
        "",
        f"Over **{n_worlds} generated worlds** ({len(price):,} room offers, "
        f"{len(fprice):,} flights). Regenerate with `python scripts/dump_data.py`.",
        "",
        "These are the correlations the world is *designed* to have. A world can satisfy",
        "every structural invariant and still be incoherent if its statistics are wrong,",
        "so they are measured here rather than claimed.",
        "",
        "| relationship | measured | target | why it matters |",
        "|---|---|---|---|",
        f"| price ↔ star_rating | {_pearson(price, stars):+.3f} | 0.4–0.8 | "
        "quality is priced, but not deterministically |",
        f"| price ↔ guest_rating | {_pearson(price, guest):+.3f} | **weaker than stars** | "
        "**the rating trap** — price tracks true quality, not displayed rating |",
        f"| price ↔ district price index | {_pearson(price, pidx):+.3f} | > 0.2 | "
        "location is priced |",
        f"| flight price ↔ on-time rate | {_pearson(fprice, ontime):+.3f} | ~0.3–0.7 | "
        "reliability costs money, imperfectly |",
        f"| flight price ↔ seats remaining | {_pearson(fprice, seats):+.3f} | **> 0** | "
        "expensive flex/business fares sit unsold; the cheap seats are the ones that go |",
        "",
        "## The rating trap, measured",
        "",
        f"- mean \\|rating − star-implied\\| with **< 60 reviews**: **{statistics.fmean(lo):.3f}**",
        f"- mean \\|rating − star-implied\\| with **> 800 reviews**: **{statistics.fmean(hi):.3f}**",
        f"- ratio: **{statistics.fmean(lo) / max(1e-9, statistics.fmean(hi)):.1f}×** noisier "
        "at low review counts",
        "",
        "This is how the *'always pick the highest-rated hotel'* exploit is defeated by the",
        "data distribution rather than by a penalty term. A 9.8 from 11 reviews is weak",
        "evidence; because price tracks stars rather than displayed rating, the lucky",
        "small-sample hotel is cheap **and** looks excellent. Sorting on rating",
        "systematically selects them and systematically overpays.",
        "",
        "## Price distribution",
        "",
        f"- room / night: min ${min(price)/100:,.0f} · median ${statistics.median(price)/100:,.0f} "
        f"· p95 ${sorted(price)[int(len(price)*.95)]/100:,.0f} · max ${max(price)/100:,.0f}",
        f"- flight: min ${min(fprice)/100:,.0f} · median ${statistics.median(fprice)/100:,.0f} "
        f"· p95 ${sorted(fprice)[int(len(fprice)*.95)]/100:,.0f} · max ${max(fprice)/100:,.0f}",
        f"- sold-out share (0 seats): {sum(1 for x in seats if x == 0)/len(seats):.1%}",
    ]
    (DATA / "world_stats.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    w = dump_world()
    dump_personas()
    dump_stats()
    print(
        f"data/sample_world.json  flights={len(w['flights'])} hotels={len(w['hotels'])} "
        f"rooms={len(w['room_offers'])} activities={len(w['activities'])} rail={len(w['rail'])}"
    )
    print("data/sample_personas.json, data/world_stats.md written")
