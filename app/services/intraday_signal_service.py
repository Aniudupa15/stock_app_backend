import logging
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from app.analysis.candlestick import detect_latest_patterns
from app.analysis.levels import detect_breakout_or_breakdown, support_resistance_levels
from app.analysis.trend import classify_trend
from app.core.exceptions import StockNotFoundError
from app.domain.ports import HistoricalPriceRepositoryPort, StockRepositoryPort
from app.indicators.bands import bollinger_bands
from app.indicators.moving_averages import sma
from app.indicators.oscillators import rsi, stochastic_rsi
from app.indicators.trend import macd, supertrend
from app.indicators.volatility import atr
from app.schemas.intraday_signal import IntradaySignalOut

logger = logging.getLogger(__name__)

_LOOKBACK_CALENDAR_DAYS = 450
_BULLISH_PATTERNS = {"Bullish Engulfing", "Hammer", "Morning Star"}
_BEARISH_PATTERNS = {"Bearish Engulfing", "Shooting Star"}
_BUY_THRESHOLD_PCT = 30.0
_SELL_THRESHOLD_PCT = -30.0


def _score_rsi(value: float | None) -> tuple[int, str | None]:
    if value is None:
        return 0, None
    if value < 30:
        return 1, f"RSI(14) at {value:.1f} indicates oversold conditions, favoring a bounce."
    if value > 70:
        return -1, f"RSI(14) at {value:.1f} indicates overbought conditions, favoring a pullback."
    return 0, f"RSI(14) at {value:.1f} is in neutral territory."


def _score_macd(macd_val: float | None, signal_val: float | None) -> tuple[int, str | None]:
    if macd_val is None or signal_val is None:
        return 0, None
    if macd_val > signal_val:
        return 1, f"MACD ({macd_val:.2f}) is above its signal line ({signal_val:.2f}), indicating bullish momentum."
    if macd_val < signal_val:
        return -1, f"MACD ({macd_val:.2f}) is below its signal line ({signal_val:.2f}), indicating bearish momentum."
    return 0, None


def _score_moving_averages(close: float, sma20: float | None, sma50: float | None) -> tuple[int, str | None]:
    if sma20 is None or sma50 is None:
        return 0, None
    if close > sma20 > sma50:
        return 1, f"Price ({close:.2f}) is above both SMA20 ({sma20:.2f}) and SMA50 ({sma50:.2f}) - bullish alignment."
    if close < sma20 < sma50:
        return -1, f"Price ({close:.2f}) is below both SMA20 ({sma20:.2f}) and SMA50 ({sma50:.2f}) - bearish alignment."
    return 0, "Moving averages show a mixed/non-trending alignment."


def _score_bollinger(close: float, upper: float | None, lower: float | None) -> tuple[int, str | None]:
    if upper is None or lower is None:
        return 0, None
    if close <= lower:
        return 1, f"Price ({close:.2f}) is at/below the lower Bollinger Band ({lower:.2f}) - potential oversold bounce."
    if close >= upper:
        return (
            -1,
            f"Price ({close:.2f}) is at/above the upper Bollinger Band ({upper:.2f}) - potential overbought reversal.",
        )
    return 0, None


def _score_trend(trend_label: str) -> tuple[int, str | None]:
    if trend_label == "Uptrend":
        return 1, "Technical trend classification: Uptrend (ADX-confirmed)."
    if trend_label == "Downtrend":
        return -1, "Technical trend classification: Downtrend (ADX-confirmed)."
    return 0, f"Technical trend classification: {trend_label}."


def _score_supertrend(direction: int | None) -> tuple[int, str | None]:
    if direction == 1:
        return 1, "Supertrend is in an uptrend state."
    if direction == -1:
        return -1, "Supertrend is in a downtrend state."
    return 0, None


def _score_stochastic_rsi(k_value: float | None) -> tuple[int, str | None]:
    if k_value is None:
        return 0, None
    if k_value < 20:
        return 1, f"Stochastic RSI %K at {k_value:.1f} indicates oversold conditions."
    if k_value > 80:
        return -1, f"Stochastic RSI %K at {k_value:.1f} indicates overbought conditions."
    return 0, None


def _score_patterns(patterns: list[str]) -> tuple[int, str | None]:
    bullish = [p for p in patterns if p in _BULLISH_PATTERNS]
    bearish = [p for p in patterns if p in _BEARISH_PATTERNS]
    if bullish and not bearish:
        return 1, f"Bullish candlestick pattern(s) detected: {', '.join(bullish)}."
    if bearish and not bullish:
        return -1, f"Bearish candlestick pattern(s) detected: {', '.join(bearish)}."
    return 0, None


def _score_breakout(breakout: str | None) -> tuple[int, str | None]:
    if breakout == "breakout":
        return 1, "Price broke out above a recent resistance level with above-average volume."
    if breakout == "breakdown":
        return -1, "Price broke down below a recent support level with above-average volume."
    return 0, None


def _last_float(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    last = values[-1]
    return None if np.isnan(last) else float(last)


class IntradaySignalService:
    """Rule-based intraday signal engine: combines Phase 2's indicators with
    Phase 3's candlestick/level analysis into a scored Buy/Sell/Hold call.
    Deterministic, no ML/LLM - every reasoning line names the actual computed
    value behind it, never a bare signal.
    """

    def __init__(self, stock_repository: StockRepositoryPort, price_repository: HistoricalPriceRepositoryPort):
        self._stock_repository = stock_repository
        self._price_repository = price_repository

    async def get_signal(self, symbol: str) -> IntradaySignalOut:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        to_date = date.today()
        from_date = to_date - timedelta(days=_LOOKBACK_CALENDAR_DAYS)
        bars = await self._price_repository.get_bars(stock.symbol, from_date, to_date)

        if len(bars) < 20:
            return IntradaySignalOut(symbol=stock.symbol, has_data=False)

        opens = np.array([float(b.open) for b in bars])
        highs = np.array([float(b.high) for b in bars])
        lows = np.array([float(b.low) for b in bars])
        closes = np.array([float(b.close) for b in bars])
        volumes = np.array([float(b.volume) for b in bars])
        last_close = closes[-1]

        rsi_val = _last_float(rsi(closes, 14))
        macd_line, signal_line, _hist = macd(closes)
        macd_val, signal_val = _last_float(macd_line), _last_float(signal_line)
        sma20_val, sma50_val = _last_float(sma(closes, 20)), _last_float(sma(closes, 50))
        bb_upper, _bb_mid, bb_lower = bollinger_bands(closes)
        bb_upper_val, bb_lower_val = _last_float(bb_upper), _last_float(bb_lower)
        trend_label = classify_trend(highs, lows, closes)
        _st_line, st_direction = supertrend(highs, lows, closes)
        st_dir_val = int(st_direction[-1]) if len(st_direction) and st_direction[-1] != 0 else None
        stoch_k, _stoch_d = stochastic_rsi(closes)
        stoch_k_val = _last_float(stoch_k)
        patterns = detect_latest_patterns(opens, highs, lows, closes)
        levels = support_resistance_levels(highs, lows)
        breakout = detect_breakout_or_breakdown(closes, volumes, levels)
        atr_val = _last_float(atr(highs, lows, closes, 14))

        components = [
            _score_rsi(rsi_val),
            _score_macd(macd_val, signal_val),
            _score_moving_averages(last_close, sma20_val, sma50_val),
            _score_bollinger(last_close, bb_upper_val, bb_lower_val),
            _score_trend(trend_label),
            _score_supertrend(st_dir_val),
            _score_stochastic_rsi(stoch_k_val),
            _score_patterns(patterns),
            _score_breakout(breakout),
        ]
        raw_score = sum(c[0] for c in components)
        reasoning = [c[1] for c in components if c[1]]

        max_score = len(components)
        score_pct = (raw_score / max_score) * 100

        if score_pct >= _BUY_THRESHOLD_PCT:
            signal = "BUY"
        elif score_pct <= _SELL_THRESHOLD_PCT:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = round(min(abs(score_pct), 95.0), 1)
        probability = round(max(15.0, min(85.0, 50 + score_pct * 0.35)), 1)

        entry_price = target_price = stop_loss = risk_reward_ratio = None
        if signal != "HOLD" and atr_val:
            entry_price = last_close
            resistances = sorted(lvl.price for lvl in levels if lvl.kind == "resistance" and lvl.price > last_close)
            supports = sorted(
                (lvl.price for lvl in levels if lvl.kind == "support" and lvl.price < last_close), reverse=True
            )

            # ATR-based fallback targets a 2:1 reward:risk by design. A support/
            # resistance level is only preferred over that fallback if it falls
            # within a sane distance band - a level barely off entry would
            # otherwise produce a tiny, meaningless reward (or risk).
            atr_target_buy = last_close + 2 * atr_val
            atr_stop_buy = last_close - atr_val
            atr_target_sell = last_close - 2 * atr_val
            atr_stop_sell = last_close + atr_val

            if signal == "BUY":
                candidate_resistance = next(
                    (r for r in resistances if last_close + 0.5 * atr_val <= r <= atr_target_buy), None
                )
                target_price = candidate_resistance if candidate_resistance else atr_target_buy
                candidate_support = next((s for s in supports if atr_stop_buy <= s <= last_close - 0.3 * atr_val), None)
                stop_loss = candidate_support if candidate_support else atr_stop_buy
            else:  # SELL
                candidate_support = next(
                    (s for s in supports if atr_target_sell <= s <= last_close - 0.5 * atr_val), None
                )
                target_price = candidate_support if candidate_support else atr_target_sell
                candidate_resistance = next(
                    (r for r in resistances if last_close + 0.3 * atr_val <= r <= atr_stop_sell), None
                )
                stop_loss = candidate_resistance if candidate_resistance else atr_stop_sell

            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = round(reward / risk, 2) if risk > 0 else None

        return IntradaySignalOut(
            symbol=stock.symbol,
            as_of=bars[-1].trade_date,
            has_data=True,
            signal=signal,
            confidence=Decimal(str(confidence)),
            entry_price=Decimal(str(round(entry_price, 2))) if entry_price is not None else None,
            target_price=Decimal(str(round(target_price, 2))) if target_price is not None else None,
            stop_loss=Decimal(str(round(stop_loss, 2))) if stop_loss is not None else None,
            risk_reward_ratio=Decimal(str(risk_reward_ratio)) if risk_reward_ratio is not None else None,
            probability=Decimal(str(probability)),
            reasoning=reasoning,
        )
