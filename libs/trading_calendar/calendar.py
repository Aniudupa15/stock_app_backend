"""IST-aware NSE trading calendar and intraday market-state machine.

Answers the two questions every part of the trading engine keeps asking:
"is the market open right now?" and "is it time to square off intraday
positions?" - correctly, in Asia/Kolkata wall-clock, holiday-aware.

Holidays are **injected**, never hardcoded (Phase 1 §4): the NSE holiday
master is fetched by the data-service and passed in, matching this project's
standing rule against baked-in lists. Weekend + special (Muhurat) sessions
are handled here; the *dates* come from outside.

Session times (NSE cash equity):
  pre-open   09:00-09:15   (order collection + matching, no continuous fills)
  continuous 09:15-15:30   (the only window MARKET/LIMIT orders fill)
  post-close 15:30-16:00   (AMO / closing session)
Intraday (MIS) square-off is enforced by us starting `buffer` minutes before
15:30 (default 15 -> 15:15), never left to the broker's ~15:20 forced close.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from enum import Enum

import pytz

IST = pytz.timezone("Asia/Kolkata")

PRE_OPEN_START = time(9, 0)
CONTINUOUS_START = time(9, 15)
CONTINUOUS_END = time(15, 30)
POST_CLOSE_END = time(16, 0)

DEFAULT_SQUARE_OFF_BUFFER_MINUTES = 15


class MarketState(str, Enum):
    CLOSED = "CLOSED"
    PRE_OPEN = "PRE_OPEN"
    CONTINUOUS = "CONTINUOUS"
    POST_CLOSE = "POST_CLOSE"


class TradingCalendar:
    """Holiday- and session-aware calendar. Stateless beyond its holiday set.

    `holidays`: full-day equity holidays (weekdays the market is shut).
    `special_sessions`: date -> (continuous_start, continuous_end) for
        non-standard days such as Diwali Muhurat trading (often a weekend with
        a short custom window). A date present here is a trading day even if it
        is a weekend, and its custom window overrides the default.
    """

    def __init__(
        self,
        holidays: set[date] | None = None,
        special_sessions: dict[date, tuple[time, time]] | None = None,
    ) -> None:
        self._holidays = holidays or set()
        self._special = special_sessions or {}

    # --- day-level ---

    def is_trading_day(self, d: date) -> bool:
        if d in self._special:
            return True
        if d in self._holidays:
            return False
        return d.weekday() < 5  # Mon-Fri

    def next_trading_day(self, d: date) -> date:
        nxt = d + timedelta(days=1)
        while not self.is_trading_day(nxt):
            nxt += timedelta(days=1)
        return nxt

    def previous_trading_day(self, d: date) -> date:
        prev = d - timedelta(days=1)
        while not self.is_trading_day(prev):
            prev -= timedelta(days=1)
        return prev

    # --- session windows ---

    def continuous_window(self, d: date) -> tuple[time, time]:
        """(start, end) of the continuous session for a trading day."""
        return self._special.get(d, (CONTINUOUS_START, CONTINUOUS_END))

    def square_off_start(self, d: date, buffer_minutes: int = DEFAULT_SQUARE_OFF_BUFFER_MINUTES) -> time:
        """Wall-clock time at which we begin force-exiting MIS positions."""
        _, end = self.continuous_window(d)
        dt = datetime.combine(d, end) - timedelta(minutes=buffer_minutes)
        return dt.time()

    # --- point-in-time state ---

    def _to_ist(self, dt: datetime) -> datetime:
        """Normalise to IST. Naive datetimes are assumed to already be IST
        wall-clock (the convention across the engine); aware ones are
        converted."""
        if dt.tzinfo is None:
            return IST.localize(dt)
        return dt.astimezone(IST)

    def state_at(self, dt: datetime) -> MarketState:
        ist = self._to_ist(dt)
        d, t = ist.date(), ist.time()
        if not self.is_trading_day(d):
            return MarketState.CLOSED
        start, end = self.continuous_window(d)
        if t < PRE_OPEN_START:
            return MarketState.CLOSED
        if t < start:
            return MarketState.PRE_OPEN
        if t < end:
            return MarketState.CONTINUOUS
        if t <= POST_CLOSE_END:
            return MarketState.POST_CLOSE
        return MarketState.CLOSED

    def is_open(self, dt: datetime) -> bool:
        """True only during continuous trading - the only time orders fill."""
        return self.state_at(dt) is MarketState.CONTINUOUS

    def in_square_off_window(
        self, dt: datetime, buffer_minutes: int = DEFAULT_SQUARE_OFF_BUFFER_MINUTES
    ) -> bool:
        """True from `square_off_start` up to continuous close on a trading day.

        The safety monitor exits open MIS positions and blocks new MIS intents
        while this is true.
        """
        ist = self._to_ist(dt)
        d, t = ist.date(), ist.time()
        if not self.is_trading_day(d):
            return False
        _, end = self.continuous_window(d)
        return self.square_off_start(d, buffer_minutes) <= t < end

    def seconds_until_close(self, dt: datetime) -> int | None:
        """Seconds until continuous close, or None if not currently in the
        continuous session (nothing to count down to)."""
        ist = self._to_ist(dt)
        if not self.is_open(ist):
            return None
        _, end = self.continuous_window(ist.date())
        close_dt = IST.localize(datetime.combine(ist.date(), end))
        return int((close_dt - ist).total_seconds())
