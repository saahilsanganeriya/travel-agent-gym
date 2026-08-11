"""Act II — the trip actually happens, and the plan meets reality.

This is where act I's choices become consequences. Every edge in the cascade fires
*conditionally on what was booked*, which is the difference between a disruption engine and
a random tax:

    FLIGHT_DELAYED(outbound)                p ~ 1 − on_time_rate
      └→ MISSED_CONNECTION                  iff delay > connection_buffer
          └→ LATE_ARRIVAL                   knock-on to the arrival time
              └→ HOTEL_NO_SHOW              iff arrival is past the hold time
                  └→ ACTIVITY_FORFEITED     iff the activity needed you there that morning
      └→ (nothing)                          if you flew nonstop, or bought slack

A FLEX fare rebooks free; a BASIC fare costs its change fee. A FREE_CANCELLATION room
releases cleanly; a NON_REFUNDABLE one is money gone. Rail almost never fires at all. None
of that is priced by the reward function — it is priced by the simulator, which is the
point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from travel_env.contracts import BookingKind, DisruptionKind, RatePlan
from travel_env.itinerary import Itinerary
from travel_env.ledger import Ledger

HOTEL_HOLD_HOUR = 22
"""Local hour after which an unclaimed room is released. Hotels really do this."""

WEATHER_CLOSURE_RATE = 0.06
"""Per outdoor activity. Japan in typhoon season is not hypothetical."""


@dataclass
class DisruptionLog:
    events: list[tuple[str, str]] = field(default_factory=list)
    damage_cents: int = 0
    forfeited_value: float = 0.0

    def record(self, kind: DisruptionKind, detail: str) -> None:
        self.events.append((kind.value, detail))


def run_cascade(
    *,
    rng: np.random.Generator,
    itinerary: Itinerary,
    ledger: Ledger,
    turn: int,
    party_size: int,
) -> tuple[int, float]:
    """Run the trip day by day. Returns (damage_cents, forfeited_activity_value).

    Damage is real money the client lost. Forfeited value is utility that evaporated —
    a paid-for activity that could not be attended is both.
    """
    log = DisruptionLog()

    outbound = itinerary.outbound
    if outbound is None:
        return 0, 0.0

    # ---- Root event: does the outbound leg go wrong? --------------------------------
    delay_minutes = 0
    cancelled = False
    roll = float(rng.random())
    failure_p = 1.0 - outbound.on_time_rate
    if roll < failure_p * 0.25:
        cancelled = True
        log.record(DisruptionKind.FLIGHT_CANCELLED, f"{outbound.carrier}{outbound.flight_number}")
    elif roll < failure_p:
        # Delay magnitude is heavy-tailed: most delays are short, a few are ruinous.
        delay_minutes = int(np.clip(rng.pareto(1.4) * 45, 15, 600))
        log.record(
            DisruptionKind.FLIGHT_DELAYED,
            f"{outbound.carrier}{outbound.flight_number} +{delay_minutes}m",
        )

    if cancelled:
        damage, _ = _handle_cancellation(
            ledger=ledger, turn=turn, booking_key="bk_flight_out",
            change_fee=outbound.change_fee * party_size, log=log,
        )
        log.damage_cents += damage
        # A cancelled outbound costs a whole day of the trip.
        delay_minutes = 24 * 60

    # ---- Edge 1: does the delay break a connection? ----------------------------------
    misconnected = False
    if delay_minutes and outbound.stops > 0:
        buffer_minutes = outbound.connection_buffer_minutes or 90
        if delay_minutes > buffer_minutes:
            misconnected = True
            log.record(
                DisruptionKind.MISSED_CONNECTION,
                f"{delay_minutes}m delay vs {buffer_minutes}m buffer",
            )
            # Re-accommodation on the next available bank: several more hours.
            delay_minutes += int(rng.integers(120, 420))
            log.damage_cents += _rebooking_cost(
                ledger=ledger, turn=turn, booking_key="bk_flight_out",
                fee=outbound.change_fee * party_size, refundable=outbound.refundable,
            )

    # ---- Edge 2: does a late arrival lose the first night's room? ---------------------
    arrival_local_hour = (outbound.local_departure_hour + outbound.duration_minutes // 60) % 24
    effective_hour = (arrival_local_hour + delay_minutes // 60) % 24
    slipped_past_hold = delay_minutes > 0 and (
        effective_hour >= HOTEL_HOLD_HOUR or effective_hour < 5
    )

    first_city = next(iter(itinerary.lodging), None)
    if slipped_past_hold and first_city is not None:
        hotel, offer, _ = itinerary.lodging[first_city]
        key = f"bk_hotel_{first_city.value}"
        if key in ledger.bookings:
            log.record(DisruptionKind.HOTEL_OVERBOOKED, f"{hotel.name} released the room")
            if offer.rate_plan is RatePlan.NON_REFUNDABLE:
                # Money gone. This is the act-I rate-plan decision landing.
                log.damage_cents += offer.price_per_night
                ledger.mark_disrupted(key)
            else:
                refund = ledger.cancel(turn=turn, key=key, agent_initiated=False)
                # Walking to a same-night replacement costs a premium over the original.
                log.damage_cents += max(0, int(offer.price_per_night * 1.4) - refund)

    # ---- Edge 3: do forfeited activities follow? --------------------------------------
    lost_days = delay_minutes // (24 * 60)
    for n, slot in enumerate(itinerary.activities):
        key = f"bk_act_{n}"
        forfeited = False
        reason = ""

        if slot.trip_day <= lost_days and delay_minutes > 0:
            forfeited, reason = True, "unreachable after the delay"
        elif misconnected and slot.trip_day == 0:
            forfeited, reason = True, "missed connection ate day 0"
        elif slot.activity.category.value in ("nature",) and rng.random() < WEATHER_CLOSURE_RATE:
            forfeited, reason = True, "weather closure"
            log.record(DisruptionKind.WEATHER_CLOSURE, slot.activity.name)

        if forfeited:
            log.record(DisruptionKind.ACTIVITY_CLOSED, f"{slot.activity.name}: {reason}")
            if key in ledger.bookings:
                booking = ledger.bookings[key]
                if booking.refundable:
                    ledger.cancel(turn=turn, key=key, agent_initiated=False)
                else:
                    log.damage_cents += booking.price_paid
                    ledger.mark_disrupted(key)
            # The utility of the activity evaporates whether or not the money came back.
            log.forfeited_value += 1.0 / max(1, len(itinerary.activities))

    # ---- Everything that survived completes -------------------------------------------
    for booking in list(ledger.active_bookings()):
        ledger.mark_completed(booking.key)

    return log.damage_cents, min(1.0, log.forfeited_value)


def _handle_cancellation(
    *, ledger: Ledger, turn: int, booking_key: str, change_fee: int, log: DisruptionLog
) -> tuple[int, bool]:
    """Supplier-initiated cancellation. Returns (out-of-pocket cents, was_refunded)."""
    if booking_key not in ledger.bookings:
        return 0, False
    booking = ledger.bookings[booking_key]
    if booking.refundable:
        # FLEX pays off exactly here: money back, rebook at roughly the same fare.
        ledger.cancel(turn=turn, key=booking_key, agent_initiated=False)
        return 0, True
    # BASIC/STANDARD: the fare is not returned and the change fee is charged.
    ledger.charge_change_fee(
        turn=turn, key=booking_key, fee=change_fee, reason="supplier cancellation, non-flex fare"
    )
    ledger.mark_disrupted(booking_key)
    return change_fee, False


def _rebooking_cost(
    *, ledger: Ledger, turn: int, booking_key: str, fee: int, refundable: bool
) -> int:
    if refundable or booking_key not in ledger.bookings:
        return 0
    ledger.charge_change_fee(
        turn=turn, key=booking_key, fee=fee, reason="misconnection rebooking"
    )
    return fee
