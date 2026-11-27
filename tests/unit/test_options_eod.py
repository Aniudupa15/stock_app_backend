"""EOD option backtester: bhavcopy parsing (both NSE formats), Black-Scholes
IV/delta, chain lookups, engine cash/cost/settlement maths, and all four
strategies on a SYNTHETIC option market (prices from Black-Scholes) -
mechanics only, says nothing about real profitability."""

import math
from datetime import date, timedelta
from decimal import Decimal

import pytest

from libs.charges import nse_fno
from libs.charges.models import Side
from libs.derivatives import bs
from libs.derivatives.contracts import InstrumentKind
from libs.options_eod.bhav import OptRow, parse_fo_bhavcopy, read_day, trim, url_for, write_day
from libs.options_eod.chain import DayChain
from libs.options_eod.data import iter_days
from libs.options_eod.engine import Book, Ctx, Leg, run_eod
from libs.options_eod.report import passes, split_date, stats, summary_lines
from libs.options_eod.strategies import MonthlyIronCondor, TrendCreditSpread, TrendDebitSpread, VixSpikeCondor

OLD = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,
FUTIDX,NIFTY,28-Mar-2019,0,XX,11500,11600,11450,11550.5,11550.5,1000,1,1,0,14-MAR-2019,
FUTIDX,NIFTY,25-Apr-2019,0,XX,11550,11650,11500,11600,11600,100,1,1,0,14-MAR-2019,
OPTIDX,NIFTY,28-Mar-2019,11500,CE,100,120,90,110.25,110.25,5000,1,2000,0,14-MAR-2019,
OPTIDX,NIFTY,28-Mar-2019,11500,PE,60,70,50,55,55,4000,1,1500,0,14-MAR-2019,
OPTIDX,NIFTY,28-Mar-2019,15000,CE,0,0,0,0.05,0.1,0,0,0,0,14-MAR-2019,
OPTIDX,BANKNIFTY,28-Mar-2019,29000,CE,1,1,1,1,1,1,1,1,0,14-MAR-2019,
"""

UDIFF = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2026-11-18,2026-11-18,FO,NSE,IDF,1,,NIFTY,,2026-11-24,2026-11-24,,,NIFTY24NOVFUT,25300,25400,25200,25350,25350,25300,25315.65,25350,100,1,500,1,1,F1,65,,,,,
2026-11-18,2026-11-18,FO,NSE,IDO,2,,NIFTY,,2026-11-24,2026-11-24,25300.00,PE,NIFTY24NOV25300PE,190,200,150,191.8,191.8,180,25315.65,191.8,1000,10,25000,1,1,F1,65,,,,,
2026-11-18,2026-11-18,FO,NSE,IDO,3,,NIFTY,,2026-12-01,2026-12-01,25300.00,CE,NIFTY01DEC25300CE,210,220,200,215,215,205,25315.65,215,900,10,9000,1,1,F1,65,,,,,
2026-11-18,2026-11-18,FO,NSE,STO,4,,RELIANCE,,2026-11-24,2026-11-24,1500.00,CE,X,1,1,1,1,1,1,1,1,1,1,1,1,1,F1,500,,,,,
"""


def test_parse_old_and_udiff_formats():
    rows, fut = parse_fo_bhavcopy(OLD)
    assert fut == 11550.5  # near-month future
    assert {(r.strike, r.kind) for r in rows} == {(11500.0, "CE"), (11500.0, "PE"), (15000.0, "CE")}
    ce = next(r for r in rows if r.kind == "CE" and r.strike == 11500)
    assert (ce.expiry, ce.close, ce.volume, ce.oi) == (date(2019, 3, 28), 110.25, 5000, 2000)

    rows, fut = parse_fo_bhavcopy(UDIFF)
    assert fut == 25350
    pe = next(r for r in rows if r.kind == "PE")
    assert (pe.expiry, pe.strike, pe.close, pe.volume) == (date(2026, 11, 24), 25300.0, 191.8, 25000)
    assert len(rows) == 2  # RELIANCE stock option ignored


def test_urls_switch_format_on_udiff_date():
    assert url_for(date(2019, 3, 14)).endswith("/DERIVATIVES/2019/MAR/fo14MAR2019bhav.csv.zip")
    assert url_for(date(2026, 11, 18)).endswith("/fo/BhavCopy_NSE_FO_0_0_0_20261118_F_0000.csv.zip")


def test_store_roundtrip_and_trim(tmp_path):
    rows, fut = parse_fo_bhavcopy(OLD)
    kept = trim(rows, date(2019, 3, 14), fut)
    assert {r.strike for r in kept} == {11500.0}  # 15000 is >25% above the 11550 future: dropped
    write_day(tmp_path / "2019-03-14.csv.gz", kept)
    assert read_day(tmp_path / "2019-03-14.csv.gz") == kept


def test_black_scholes_iv_roundtrip_and_delta():
    p = bs.price(False, 24000, 23500, 30 / 365, 0.14)
    assert bs.implied_vol(False, p, 24000, 23500, 30 / 365) == pytest.approx(0.14, abs=1e-6)
    assert 0 < bs.delta(True, 24000, 24500, 30 / 365, 0.14) < 0.5
    assert -0.5 < bs.delta(False, 24000, 23500, 30 / 365, 0.14) < 0
    assert bs.implied_vol(True, 0.0, 24000, 24500, 0.1) is None


# ---------------------------------------------------------------- synthetic market


def _expiries(start: date, end: date) -> list[date]:
    """Weekly Tuesdays; the last Tuesday of each month is the monthly."""
    out, d = [], start
    while d <= end + timedelta(days=100):
        if d.weekday() == 1:
            out.append(d)
        d += timedelta(days=1)
    return out


def market(days: int = 320, spot0: float = 24000.0, vix_fn=None, spot_fn=None, band: float = 0.12) -> list[DayChain]:
    start = date(2024, 1, 1)
    trading, d = [], start
    while len(trading) < days:
        if d.weekday() < 5:
            trading.append(d)
        d += timedelta(days=1)
    exps = _expiries(start, trading[-1])
    chains = []
    for i, td in enumerate(trading):
        spot = spot_fn(i) if spot_fn else spot0 * (1 + 0.06 * math.sin(i / 25))
        vix = vix_fn(i) if vix_fn else 15.0
        rows = []
        for e in exps:
            dte = (e - td).days
            if not 0 <= dte <= 70:
                continue
            for k in range(int(spot * (1 - band)) // 100 * 100, int(spot * (1 + band)), 100):
                for kind in ("CE", "PE"):
                    p = bs.price(kind == "CE", spot, k, dte / 365, vix / 100)
                    rows.append(OptRow(e, float(k), kind, round(max(p, 0.05), 2), round(max(p, 0.05), 2), 100, 100))
        chains.append(DayChain(td, spot, vix, rows))
    return chains


def test_chain_strike_by_delta_and_monthlies():
    day = market(1)[0]
    monthly = day.monthly_expiries()
    assert all(e.weekday() == 1 for e in monthly) and len(monthly) >= 3
    e = monthly[1]
    k = day.strike_by_delta(e, "PE", 0.15)
    assert k < day.spot and abs(abs(day.delta(e, k, "PE")) - 0.15) < 0.03
    assert day.strike_near(e, "CE", day.spot + 120, "above") >= day.spot + 120


def test_engine_cash_costs_and_expiry_settlement():
    days = market(40, spot_fn=lambda i: 24000.0)  # flat market
    day0 = days[0]
    e = next(x for x in day0.expiries() if day0.dte(x) >= 7)
    book = Book("t", {})
    ctx = Ctx(day0, book, lot=65, slippage=1.0)
    legs = [Leg(e, 24500.0, "CE", -1), Leg(e, 24600.0, "CE", +1)]  # bear call spread, OTM
    assert ctx.open(legs)
    short_px, long_px = day0.price(e, 24500.0, "CE"), day0.price(e, 24600.0, "CE")
    assert book.position.entry_cash == pytest.approx((short_px - 1) - (long_px + 1))
    expected_costs = float(
        nse_fno.compute(Side.SELL, InstrumentKind.CE, 65, Decimal(str(round(short_px - 1, 2)))).total
        + nse_fno.compute(Side.BUY, InstrumentKind.CE, 65, Decimal(str(round(long_px + 1, 2)))).total
    )
    assert book.position.entry_costs == pytest.approx(expected_costs)
    expiry_day = next(d for d in days if d.date >= e)
    trade = Ctx(expiry_day, book, 65, 1.0).settle()
    assert trade.reason == "EXPIRY" and trade.exit_cash == 0  # both legs expire worthless (spot 24000)
    assert trade.rupees_net == pytest.approx(trade.entry_cash * 65 - expected_costs, abs=0.01)


def test_open_refuses_when_a_leg_has_no_quote():
    day = market(1)[0]
    e = day.expiries()[2]
    assert not Ctx(day, Book("t", {}), 65, 1).open([Leg(e, 99999.0, "CE", -1)])


@pytest.mark.parametrize(
    "strategy",
    [
        MonthlyIronCondor(),
        TrendCreditSpread(fast=5, slow=15),
        TrendCreditSpread(fast=5, slow=15, cycle="monthly"),
        TrendDebitSpread(fast=5, slow=15),
    ],
    ids=["condor", "credit-weekly", "credit-monthly", "debit"],
)
def test_strategies_trade_and_books_reconcile(strategy):
    (book,) = run_eod(market(), [strategy])
    assert book.trades, f"{strategy.label} never traded"
    assert book.position is None or book.equity  # marked daily
    if book.position is None:
        assert book.equity[-1][1] == pytest.approx(sum(t.rupees_net for t in book.trades), abs=0.5)
    for t in book.trades:
        assert t.exit_date >= t.entry_date
        assert t.rupees_net == pytest.approx(t.points * 65 - t.costs, abs=0.05)


def test_condor_respects_vix_band_and_entry_window():
    (low,) = run_eod(market(vix_fn=lambda i: 10.0), [MonthlyIronCondor()])
    assert low.trades == []  # VIX below 12: never enters
    days = market()
    (book,) = run_eod(days, [MonthlyIronCondor()])
    by_date = {d.date: d for d in days}
    assert book.trades
    for t in book.trades:
        chain = by_date[t.entry_date]
        assert any(30 <= chain.dte(e) <= 35 for e in chain.monthly_expiries())
    assert len({(t.entry_date.year, t.entry_date.month) for t in book.trades}) == len(book.trades)  # one per cycle


def test_vix_spike_condor_enters_after_the_turn():
    def vix(i):  # calm, a spike at day 280-284 that peaks at 282, then falls
        if 280 <= i <= 282:
            return 15 + 8 * (i - 279)
        if 283 <= i <= 290:
            return 39 - 3 * (i - 282)
        return 14.0

    days = market(vix_fn=vix, band=0.3)  # high VIX -> 0.10-delta strikes sit far out
    (wait,) = run_eod(days, [VixSpikeCondor(pct=0.8, wait_turn=True)])
    assert wait.trades or wait.position is not None
    entry = wait.trades[0].entry_date if wait.trades else wait.position.entry_date
    idx = next(i for i, d in enumerate(days) if d.date == entry)
    assert idx == 284  # turn seen at 283 (first lower close) -> enter the next day
    assert len(wait.trades) + (wait.position is not None) == 1  # one trade per spike


def test_walk_forward_split_and_pass_criteria():
    first, last = date(2019, 1, 1), date(2025, 12, 31)
    cut = split_date(first, last, 0.7)
    assert date(2023, 11, 1) < cut < date(2024, 1, 31)
    (book,) = run_eod(market(), [MonthlyIronCondor()])
    s = stats(book.trades, book.equity)
    assert "only" in " ".join(passes(s, 200000))  # far fewer than 100 trades on synthetic year
    assert any("Per year" in line for line in summary_lines(book, cut, 200000))


def test_iter_days_from_store_with_warmup(tmp_path):
    store = tmp_path / "NIFTY"
    for day in market(3):
        write_day(store / f"{day.date.isoformat()}.csv.gz", [r for r in day._rows.values()][:50])
    idx = tmp_path / "index_daily.csv"
    idx.write_text(
        "date,nifty,vix\n2023-12-28,23900,14\n2024-01-01,24000,15\n2024-01-02,24010,15\n2024-01-03,24020,15\n"
    )
    days = list(iter_days(store, idx, date(2024, 1, 1), date(2024, 1, 3)))
    assert [d.date for d in days] == [date(2023, 12, 28), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    assert days[0].expiries() == [] and days[1].expiries()  # warm-up day has no options


def test_profit_factor_infinite_without_losses():
    from libs.options_eod.engine import EodTrade

    t = EodTrade(date(2024, 1, 1), date(2024, 1, 5), "x", 10, 0, 10, 600, 50, "TARGET", "", 0)
    s = stats([t] * 120)
    assert s.profit_factor == float("inf")
    assert not any("PF" in f for f in passes(s, 200000))
