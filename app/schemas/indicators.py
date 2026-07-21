from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class MacdOut(BaseModel):
    macd: Decimal | None
    signal: Decimal | None
    histogram: Decimal | None


class BollingerOut(BaseModel):
    upper: Decimal | None
    middle: Decimal | None
    lower: Decimal | None


class SupertrendOut(BaseModel):
    value: Decimal | None
    direction: int | None  # 1 = uptrend, -1 = downtrend, None = not yet computed


class StochRsiOut(BaseModel):
    k: Decimal | None
    d: Decimal | None


class PivotPointsOut(BaseModel):
    pivot: Decimal
    r1: Decimal
    r2: Decimal
    r3: Decimal
    s1: Decimal
    s2: Decimal
    s3: Decimal


class VolumeProfileBinOut(BaseModel):
    price_low: Decimal
    price_high: Decimal
    volume: int


class IndicatorsOut(BaseModel):
    symbol: str
    as_of: date | None = None
    has_data: bool = False

    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    sma_200: Decimal | None = None
    ema_20: Decimal | None = None
    ema_50: Decimal | None = None
    rsi_14: Decimal | None = None
    macd: MacdOut | None = None
    bollinger: BollingerOut | None = None
    vwap_20: Decimal | None = None
    adx_14: Decimal | None = None
    atr_14: Decimal | None = None
    supertrend: SupertrendOut | None = None
    stochastic_rsi: StochRsiOut | None = None
    pivot_points: PivotPointsOut | None = None
    volume_profile: list[VolumeProfileBinOut] = []
    point_of_control: VolumeProfileBinOut | None = None
