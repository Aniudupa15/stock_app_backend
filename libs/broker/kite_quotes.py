"""Live market data from Zerodha Kite for PAPER trading.

`KiteQuoteSource` implements `QuoteSourcePort` with `kite.quote()`, so the
paper venue fills F&O (and equity) orders against real Zerodha bid/ask, volume
and circuit limits - while never calling a single order endpoint. Read-only
by construction: this class only uses `quote`, `ltp` and `instruments`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal

from libs.broker.base import BrokerError
from libs.derivatives.contracts import ContractRegistry, contracts_from_kite_instruments
from libs.trading_domain.entities import MarketQuote
from libs.trading_domain.ports import QuoteSourcePort

# F&O underlying -> Kite instrument key of its spot (used for ATM selection
# and expiry settlement).
INDEX_SPOT_KEYS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
    "SENSEX": "BSE:SENSEX",
    "BANKEX": "BSE:BANKEX",
}


def spot_key(underlying: str) -> str:
    return INDEX_SPOT_KEYS.get(underlying, f"NSE:{underlying}")


def _dec(value) -> Decimal | None:
    if value is None:
        return None
    d = Decimal(str(value))
    return d if d > 0 else None


def parse_kite_quote(symbol: str, raw: dict) -> MarketQuote:
    """Map one `kite.quote()` entry to a MarketQuote. Pure / testable."""
    depth = raw.get("depth") or {}
    buy = depth.get("buy") or []
    sell = depth.get("sell") or []
    ts = raw.get("last_trade_time") or raw.get("timestamp")
    return MarketQuote(
        symbol=symbol,
        ltp=Decimal(str(raw["last_price"])),
        bid=_dec(buy[0].get("price")) if buy else None,
        ask=_dec(sell[0].get("price")) if sell else None,
        day_volume=int(raw["volume"]) if raw.get("volume") else None,
        upper_band=_dec(raw.get("upper_circuit_limit")),
        lower_band=_dec(raw.get("lower_circuit_limit")),
        ts=ts if isinstance(ts, datetime) else None,
    )


class KiteQuoteSource(QuoteSourcePort):
    def __init__(self, kite, contracts: ContractRegistry | None = None) -> None:
        self._kite = kite
        self._contracts = contracts or ContractRegistry()

    def _key(self, symbol: str) -> str:
        contract = self._contracts.get(symbol)
        return f"{contract.exchange}:{symbol}" if contract else f"NSE:{symbol}"

    async def get_quote(self, symbol: str) -> MarketQuote:
        return (await self.get_quotes([symbol]))[symbol]

    async def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        keys = {self._key(s): s for s in symbols}
        try:
            raw = await asyncio.to_thread(self._kite.quote, list(keys))
        except Exception as exc:  # external boundary
            raise BrokerError(f"Kite quote failed: {exc}") from exc
        missing = [s for k, s in keys.items() if k not in raw]
        if missing:
            raise BrokerError(f"no Kite quote for {', '.join(missing)}")
        return {s: parse_kite_quote(s, raw[k]) for k, s in keys.items()}

    async def spot_prices(self, underlyings: list[str]) -> dict[str, Decimal]:
        keys = {spot_key(u): u for u in underlyings}
        try:
            raw = await asyncio.to_thread(self._kite.ltp, list(keys))
        except Exception as exc:
            raise BrokerError(f"Kite ltp failed: {exc}") from exc
        return {u: Decimal(str(raw[k]["last_price"])) for k, u in keys.items() if k in raw}


async def load_fno_contracts(kite, exchange: str = "NFO", underlyings: set[str] | None = None) -> ContractRegistry:
    """Download the day's F&O instrument master (lot sizes, strikes, expiries)."""
    try:
        rows = await asyncio.to_thread(kite.instruments, exchange)
    except Exception as exc:
        raise BrokerError(f"Kite instruments failed: {exc}") from exc
    contracts = contracts_from_kite_instruments(rows)
    if underlyings:
        contracts = [c for c in contracts if c.underlying in underlyings]
    return ContractRegistry(contracts)
