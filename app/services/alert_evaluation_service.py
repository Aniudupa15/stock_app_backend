"""The alert evaluation engine: for every ACTIVE alert, computes the current
value the alert cares about (price, day change, RSI, volume, 52-week extreme)
straight from stored `historical_prices` - reusing `app.indicators` the same
way `IndicatorService` does - and flips matching alerts to TRIGGERED with a
templated notification. One-shot per alert (matches typical alert UX: it
fires once, not repeatedly, until the user creates a new one).

Deferred, not built here (see Phase 4 plan): SMA/EMA cross and Golden/Death
Cross alert types (need two consecutive days' indicator values to detect a
crossover, not just a snapshot threshold) and portfolio-wide alerts (need a
different evaluation shape entirely - see `Alert.stock_id`'s nullable design
note). Both are straightforward additions later given this same pattern.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np

from app.domain.entities import Alert, AlertType
from app.domain.ports import AlertRepositoryPort, HistoricalPriceRepositoryPort, NotificationRepositoryPort
from app.indicators.oscillators import rsi as compute_rsi

logger = logging.getLogger(__name__)

# Matches IndicatorService's lookback: enough calendar days for a 252-session
# (52-week) window even after weekends/holidays are accounted for.
_LOOKBACK_CALENDAR_DAYS = 450
_WINDOW_SESSIONS = 252
_VOLUME_AVG_SESSIONS = 20


@dataclass(frozen=True, slots=True)
class _SymbolContext:
    latest_close: Decimal
    change_percent: Decimal | None
    rsi_14: Decimal | None
    latest_volume: int
    avg_volume_20: Decimal | None
    is_new_52_week_high: bool
    is_new_52_week_low: bool


def _build_context(bars: list) -> "_SymbolContext | None":
    if not bars:
        return None

    latest = bars[-1]
    previous = bars[-2] if len(bars) >= 2 else None
    change_percent = (
        (latest.close - previous.close) / previous.close * 100 if previous is not None and previous.close != 0 else None
    )

    closes = np.array([float(b.close) for b in bars])
    rsi_series = compute_rsi(closes, 14)
    rsi_14 = None
    if len(rsi_series) and not np.isnan(rsi_series[-1]):
        rsi_14 = Decimal(str(round(float(rsi_series[-1]), 4)))

    window = bars[-_WINDOW_SESSIONS:]
    period_high = max(b.close for b in window)
    period_low = min(b.close for b in window)

    prior_volumes = [b.volume for b in bars[-(_VOLUME_AVG_SESSIONS + 1) : -1]]
    avg_volume_20 = Decimal(sum(prior_volumes)) / len(prior_volumes) if prior_volumes else None

    return _SymbolContext(
        latest_close=latest.close,
        change_percent=change_percent,
        rsi_14=rsi_14,
        latest_volume=latest.volume,
        avg_volume_20=avg_volume_20,
        is_new_52_week_high=latest.close >= period_high,
        is_new_52_week_low=latest.close <= period_low,
    )


def _condition_met(alert: Alert, ctx: _SymbolContext) -> bool:
    c = alert.condition
    t = alert.alert_type
    if t == AlertType.PRICE_ABOVE:
        return ctx.latest_close >= Decimal(str(c["price"]))
    if t == AlertType.PRICE_BELOW:
        return ctx.latest_close <= Decimal(str(c["price"]))
    if t == AlertType.PERCENT_CHANGE_ABOVE:
        return ctx.change_percent is not None and ctx.change_percent >= Decimal(str(c["percent"]))
    if t == AlertType.PERCENT_CHANGE_BELOW:
        return ctx.change_percent is not None and ctx.change_percent <= Decimal(str(c["percent"]))
    if t == AlertType.RSI_ABOVE:
        return ctx.rsi_14 is not None and ctx.rsi_14 >= Decimal(str(c["threshold"]))
    if t == AlertType.RSI_BELOW:
        return ctx.rsi_14 is not None and ctx.rsi_14 <= Decimal(str(c["threshold"]))
    if t == AlertType.VOLUME_SPIKE:
        return (
            ctx.avg_volume_20 is not None
            and ctx.avg_volume_20 > 0
            and Decimal(ctx.latest_volume) >= Decimal(str(c["multiplier"])) * ctx.avg_volume_20
        )
    if t == AlertType.NEW_52_WEEK_HIGH:
        return ctx.is_new_52_week_high
    if t == AlertType.NEW_52_WEEK_LOW:
        return ctx.is_new_52_week_low
    return False


def _notification_message(alert: Alert, ctx: _SymbolContext) -> tuple[str, str]:
    title = f"{alert.symbol} alert triggered"
    t = alert.alert_type
    c = alert.condition
    if t == AlertType.PRICE_ABOVE:
        return title, f"{alert.symbol} is now at {ctx.latest_close}, at or above your target of {c['price']}."
    if t == AlertType.PRICE_BELOW:
        return title, f"{alert.symbol} is now at {ctx.latest_close}, at or below your target of {c['price']}."
    if t == AlertType.PERCENT_CHANGE_ABOVE:
        return (
            title,
            f"{alert.symbol} is up {ctx.change_percent:.2f}% today, at or above your threshold of {c['percent']}%.",
        )
    if t == AlertType.PERCENT_CHANGE_BELOW:
        return title, f"{alert.symbol} is down {ctx.change_percent:.2f}% today, past your threshold of {c['percent']}%."
    if t == AlertType.RSI_ABOVE:
        return (
            title,
            f"{alert.symbol}'s RSI(14) is {ctx.rsi_14}, at or above your threshold of {c['threshold']} (overbought).",
        )
    if t == AlertType.RSI_BELOW:
        return (
            title,
            f"{alert.symbol}'s RSI(14) is {ctx.rsi_14}, at or below your threshold of {c['threshold']} (oversold).",
        )
    if t == AlertType.VOLUME_SPIKE:
        multiple = ctx.latest_volume / ctx.avg_volume_20
        return title, f"{alert.symbol}'s volume ({ctx.latest_volume:,}) is {multiple:.1f}x its 20-day average."
    if t == AlertType.NEW_52_WEEK_HIGH:
        return title, f"{alert.symbol} made a new 52-week high at {ctx.latest_close}."
    if t == AlertType.NEW_52_WEEK_LOW:
        return title, f"{alert.symbol} made a new 52-week low at {ctx.latest_close}."
    return title, f"{alert.symbol} alert condition met."


class AlertEvaluationService:
    def __init__(
        self,
        alert_repository: AlertRepositoryPort,
        notification_repository: NotificationRepositoryPort,
        price_repository: HistoricalPriceRepositoryPort,
    ):
        self._alert_repository = alert_repository
        self._notification_repository = notification_repository
        self._price_repository = price_repository

    async def evaluate_all(self) -> int:
        alerts = await self._alert_repository.list_active()
        if not alerts:
            return 0

        by_symbol: dict[str, list[Alert]] = {}
        for a in alerts:
            by_symbol.setdefault(a.symbol, []).append(a)

        to_date = date.today()
        from_date = to_date - timedelta(days=_LOOKBACK_CALENDAR_DAYS)

        triggered_count = 0
        for symbol, symbol_alerts in by_symbol.items():
            bars = await self._price_repository.get_bars(symbol, from_date, to_date)
            ctx = _build_context(bars)
            if ctx is None:
                continue

            for alert in symbol_alerts:
                try:
                    met = _condition_met(alert, ctx)
                except (KeyError, TypeError, ValueError, ArithmeticError):
                    logger.warning("skipping alert %s with an unevaluable condition: %s", alert.id, alert.condition)
                    continue
                if not met:
                    continue

                title, message = _notification_message(alert, ctx)
                await self._alert_repository.mark_triggered(alert.id, datetime.now(UTC))
                await self._notification_repository.create(alert.user_id, alert.id, title, message)
                triggered_count += 1

        return triggered_count
