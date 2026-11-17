"""F&O contract metadata: what a futures/options tradingsymbol actually is.

The paper venue, risk gate and charges all need to know, for a symbol like
`NIFTY26NOV24000CE`, its kind (FUT/CE/PE), lot size, strike, expiry and
underlying. That metadata comes from Zerodha's instrument dump
(`kite.instruments("NFO")`) - it is never parsed out of the symbol string,
because weekly-expiry symbols are not reliably decodable and lot sizes are
revised by the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

# Index derivatives (cash-settled, lower margin) - everything else on NFO is
# a single-stock derivative.
INDEX_UNDERLYINGS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"})

FNO_EXCHANGES = frozenset({"NFO", "BFO"})


class InstrumentKind(str, Enum):
    FUT = "FUT"
    CE = "CE"  # call option
    PE = "PE"  # put option

    @property
    def is_option(self) -> bool:
        return self is not InstrumentKind.FUT


@dataclass(frozen=True, slots=True)
class FnoContract:
    tradingsymbol: str
    underlying: str  # e.g. "NIFTY", "RELIANCE"
    kind: InstrumentKind
    expiry: date
    lot_size: int
    strike: Decimal | None = None  # None for futures
    tick_size: Decimal = Decimal("0.05")
    exchange: str = "NFO"
    instrument_token: int | None = None

    @property
    def is_index(self) -> bool:
        return self.underlying in INDEX_UNDERLYINGS

    def is_expired(self, today: date) -> bool:
        return today > self.expiry

    def settlement_price(self, underlying_price: Decimal) -> Decimal:
        """Per-unit value at expiry given the underlying's final settlement
        price: futures settle at the underlying, options at intrinsic value."""
        if self.kind is InstrumentKind.FUT:
            return underlying_price
        assert self.strike is not None
        if self.kind is InstrumentKind.CE:
            return max(Decimal("0"), underlying_price - self.strike)
        return max(Decimal("0"), self.strike - underlying_price)


class ContractRegistry:
    """Symbol -> FnoContract lookup. Anything not registered is treated as
    cash equity by the venue/risk/charges layers."""

    def __init__(self, contracts: list[FnoContract] | None = None) -> None:
        self._by_symbol: dict[str, FnoContract] = {}
        for c in contracts or []:
            self.add(c)

    def add(self, contract: FnoContract) -> None:
        self._by_symbol[contract.tradingsymbol] = contract

    def get(self, symbol: str) -> FnoContract | None:
        return self._by_symbol.get(symbol)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._by_symbol

    def __len__(self) -> int:
        return len(self._by_symbol)

    def all(self) -> list[FnoContract]:
        return list(self._by_symbol.values())

    # --- selection helpers (used to pick what to paper-trade) ---

    def expiries(self, underlying: str, kind: InstrumentKind | None = None, *, on_or_after: date) -> list[date]:
        return sorted(
            {
                c.expiry
                for c in self._by_symbol.values()
                if c.underlying == underlying and (kind is None or c.kind is kind) and c.expiry >= on_or_after
            }
        )

    def nearest_future(self, underlying: str, *, today: date) -> FnoContract | None:
        futs = [
            c
            for c in self._by_symbol.values()
            if c.underlying == underlying and c.kind is InstrumentKind.FUT and c.expiry >= today
        ]
        return min(futs, key=lambda c: c.expiry, default=None)

    def atm_option(
        self, underlying: str, kind: InstrumentKind, spot: Decimal, *, today: date, expiry: date | None = None
    ) -> FnoContract | None:
        """Strike closest to `spot` in the given (default: nearest) expiry."""
        if not kind.is_option:
            raise ValueError("atm_option needs CE or PE")
        if expiry is None:
            expiries = self.expiries(underlying, kind, on_or_after=today)
            if not expiries:
                return None
            expiry = expiries[0]
        candidates = [
            c
            for c in self._by_symbol.values()
            if c.underlying == underlying and c.kind is kind and c.expiry == expiry and c.strike is not None
        ]
        return min(candidates, key=lambda c: (abs(c.strike - spot), c.strike), default=None)


def _to_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def contracts_from_kite_instruments(rows: list[dict]) -> list[FnoContract]:
    """Build contracts from `kite.instruments("NFO")` rows. Rows that are not
    FUT/CE/PE (or lack an expiry/lot size) are skipped."""
    out: list[FnoContract] = []
    for r in rows:
        kind_raw = str(r.get("instrument_type", "")).upper()
        if kind_raw not in InstrumentKind.__members__:
            continue
        expiry = _to_date(r.get("expiry"))
        lot = int(r.get("lot_size") or 0)
        if expiry is None or lot <= 0:
            continue
        kind = InstrumentKind(kind_raw)
        strike = Decimal(str(r["strike"])) if kind.is_option and r.get("strike") is not None else None
        out.append(
            FnoContract(
                tradingsymbol=r["tradingsymbol"],
                underlying=str(r.get("name") or "").upper(),
                kind=kind,
                expiry=expiry,
                lot_size=lot,
                strike=strike,
                tick_size=Decimal(str(r.get("tick_size") or "0.05")),
                exchange=str(r.get("exchange") or "NFO"),
                instrument_token=r.get("instrument_token"),
            )
        )
    return out


def kind_from_fno_symbol(tradingsymbol: str) -> InstrumentKind:
    """Instrument kind from an NFO/BFO tradingsymbol's suffix (FUT/CE/PE).
    Only valid for symbols already known to be on a derivatives exchange -
    some cash-equity symbols also end in "CE"/"PE"."""
    for kind in (InstrumentKind.FUT, InstrumentKind.CE, InstrumentKind.PE):
        if tradingsymbol.endswith(kind.value):
            return kind
    raise ValueError(f"not an F&O tradingsymbol: {tradingsymbol}")
