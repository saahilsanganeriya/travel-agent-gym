"""The transaction ledger — authoritative state for money and bookings.

Reward is computed from this ledger and never re-derived from the final itinerary. After a
disruption forces a cancel-and-rebook, "what did this trip cost" is not recoverable by
inspecting what the client ended up with: the refunded amount, the change fee and the
replacement fare have all moved. It is only recoverable from the record of what happened.

Every movement is an integer number of cents, signed against the client's wallet: charges
negative, refunds positive. `assert_reconciled()` re-derives the balance from the entry list
and raises if it disagrees with the running balance. A disagreement is a bug in the
environment, not a bad episode, so it raises rather than scoring low.
"""

from __future__ import annotations

from datetime import date

from travel_env.contracts import (
    Booking,
    BookingKind,
    BookingStatus,
    Key,
    LedgerEntry,
    LedgerEntryKind,
    SignedCents,
)


class LedgerError(RuntimeError):
    """A structural fault in the money record. Never raised for a merely bad episode."""


class Ledger:
    """Append-only record of money movements plus the booking table it explains."""

    def __init__(self, initial_budget_cents: int) -> None:
        self.initial_budget = initial_budget_cents
        self.balance: SignedCents = initial_budget_cents
        self.entries: list[LedgerEntry] = []
        self.bookings: dict[Key, Booking] = {}
        self._sequence = 0

    # ---- queries -----------------------------------------------------------------

    @property
    def spent(self) -> int:
        return self.initial_budget - self.balance

    def can_afford(self, amount_cents: int) -> bool:
        """The predicate behind the BOOK action mask. Overspending is not penalised —
        the action is removed from the space, so it cannot be chosen."""
        return amount_cents <= self.balance

    def active_bookings(self) -> list[Booking]:
        return [
            b for b in self.bookings.values()
            if b.status in (BookingStatus.CONFIRMED, BookingStatus.HELD)
        ]

    def bookings_of_kind(self, kind: BookingKind) -> list[Booking]:
        return [b for b in self.active_bookings() if b.kind is kind]

    # ---- mutations ---------------------------------------------------------------

    def _append(
        self, *, turn: int, kind: LedgerEntryKind, booking_key: Key,
        amount: SignedCents, reason: str,
    ) -> LedgerEntry:
        self.balance += amount
        entry = LedgerEntry(
            sequence=self._sequence, turn=turn, kind=kind, booking_key=booking_key,
            amount=amount, balance_after=self.balance, reason=reason,
        )
        self._sequence += 1
        self.entries.append(entry)
        return entry

    def book(
        self, *, turn: int, key: Key, kind: BookingKind, inventory_key: Key,
        price: int, refundable: bool, travellers: int, service_date: date,
        depends_on: tuple[Key, ...] = (), refund_deadline=None,
    ) -> Booking:
        if key in self.bookings:
            raise LedgerError(f"duplicate booking key {key!r}")
        if price > self.balance:
            # Unreachable when the caller honours the mask; kept because a check a second
            # code path can bypass is not a check.
            raise LedgerError(
                f"booking {key!r} costs {price} against a balance of {self.balance}"
            )
        booking = Booking(
            key=key, kind=kind, inventory_key=inventory_key,
            status=BookingStatus.CONFIRMED, price_paid=price, refundable=refundable,
            refund_deadline=refund_deadline, travellers=travellers,
            service_date=service_date, depends_on=depends_on,
        )
        self.bookings[key] = booking
        self._append(
            turn=turn, kind=LedgerEntryKind.CHARGE, booking_key=key,
            amount=-price, reason=f"book {kind.value} {inventory_key}",
        )
        return booking

    def cancel(self, *, turn: int, key: Key, agent_initiated: bool = True) -> int:
        """Cancel a booking. Returns the refund in cents.

        Refundability is the act-I decision that pays off (or doesn't) here: a FLEX fare
        or a FREE_CANCELLATION room returns the money, a BASIC fare returns nothing.
        """
        booking = self.bookings.get(key)
        if booking is None:
            raise LedgerError(f"cancel of unknown booking {key!r}")
        if booking.status not in (BookingStatus.CONFIRMED, BookingStatus.HELD):
            raise LedgerError(f"cancel of {key!r} in status {booking.status.value}")

        refund = booking.price_paid if booking.refundable else 0
        self.bookings[key] = booking.model_copy(
            update={
                "status": BookingStatus.CANCELLED if agent_initiated
                else BookingStatus.DISRUPTED
            }
        )
        if refund:
            self._append(
                turn=turn, kind=LedgerEntryKind.REFUND, booking_key=key,
                amount=refund, reason="refundable cancellation",
            )
        return refund

    def charge_change_fee(self, *, turn: int, key: Key, fee: int, reason: str) -> None:
        if fee <= 0:
            return
        self._append(
            turn=turn, kind=LedgerEntryKind.CHANGE_FEE, booking_key=key,
            amount=-fee, reason=reason,
        )

    def mark_completed(self, key: Key) -> None:
        booking = self.bookings[key]
        self.bookings[key] = booking.model_copy(update={"status": BookingStatus.COMPLETED})

    def mark_disrupted(self, key: Key) -> None:
        booking = self.bookings[key]
        self.bookings[key] = booking.model_copy(update={"status": BookingStatus.DISRUPTED})

    # ---- the close-out check -----------------------------------------------------

    def assert_reconciled(self) -> None:
        """Re-derive the balance from the entries and compare. Raises on disagreement.

        This is cheap and it is the reason reward can be trusted: if the ledger and the
        running balance ever diverge, every downstream number is wrong, and we would rather
        crash the run than publish a quietly incorrect reward.
        """
        derived = self.initial_budget + sum(e.amount for e in self.entries)
        if derived != self.balance:
            raise LedgerError(
                f"ledger does not reconcile: entries sum to {derived}, "
                f"running balance is {self.balance} (delta {self.balance - derived})"
            )
        running = self.initial_budget
        for entry in self.entries:
            running += entry.amount
            if entry.balance_after != running:
                raise LedgerError(
                    f"entry {entry.sequence} records balance_after={entry.balance_after} "
                    f"but replay gives {running}"
                )
