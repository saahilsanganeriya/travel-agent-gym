# Measured world statistics

Over **200 generated worlds** (12,000 room offers, 24,000 flights). Regenerate with `python scripts/dump_data.py`.

These are the correlations the world is *designed* to have. A world can satisfy
every structural invariant and still be incoherent if its statistics are wrong,
so they are measured here rather than claimed.

| relationship | measured | target | why it matters |
|---|---|---|---|
| price ↔ star_rating | +0.817 | 0.4–0.8 | quality is priced, but not deterministically |
| price ↔ guest_rating | +0.791 | **weaker than stars** | **the rating trap** — price tracks true quality, not displayed rating |
| price ↔ district price index | +0.312 | > 0.2 | location is priced |
| flight price ↔ on-time rate | +0.328 | ~0.3–0.7 | reliability costs money, imperfectly |
| flight price ↔ seats remaining | +0.537 | **> 0** | expensive flex/business fares sit unsold; the cheap seats are the ones that go |

## The rating trap, measured

- mean \|rating − star-implied\| with **< 60 reviews**: **0.372**
- mean \|rating − star-implied\| with **> 800 reviews**: **0.058**
- ratio: **6.5×** noisier at low review counts

This is how the *'always pick the highest-rated hotel'* exploit is defeated by the
data distribution rather than by a penalty term. A 9.8 from 11 reviews is weak
evidence; because price tracks stars rather than displayed rating, the lucky
small-sample hotel is cheap **and** looks excellent. Sorting on rating
systematically selects them and systematically overpays.

## Price distribution

- room / night: min $112 · median $406 · p95 $684 · max $1,099
- flight: min $329 · median $1,851 · p95 $6,159 · max $10,502
- sold-out share (0 seats): 3.6%
