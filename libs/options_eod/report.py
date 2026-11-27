"""Metrics, walk-forward split and pass criteria for EOD option backtests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from libs.options_eod.engine import Book, EodTrade


@dataclass(frozen=True, slots=True)
class Stats:
    trades: int
    win_rate: float
    profit_factor: float | None
    expectancy: float  # rupees per trade, after costs
    net: float
    avg_points: float
    max_dd: float  # rupees, on the daily marked equity (full run only)
    years_positive: int
    years_traded: int


def stats(trades: list[EodTrade], equity: list[tuple[date, float]] | None = None) -> Stats:
    n = len(trades)
    wins = [t.rupees_net for t in trades if t.rupees_net > 0]
    losses = [-t.rupees_net for t in trades if t.rupees_net < 0]
    gw, gl = sum(wins), sum(losses)
    by_year: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        by_year[t.exit_date.year].append(t.rupees_net)
    max_dd = 0.0
    if equity:
        peak = equity[0][1]
        for _, v in equity:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)
    return Stats(
        trades=n,
        win_rate=len(wins) / n if n else 0.0,
        profit_factor=(gw / gl) if gl > 0 else (float("inf") if gw > 0 else None),
        expectancy=(gw - gl) / n if n else 0.0,
        net=gw - gl,
        avg_points=sum(t.points for t in trades) / n if n else 0.0,
        max_dd=max_dd,
        years_positive=sum(1 for v in by_year.values() if sum(v) > 0),
        years_traded=len(by_year),
    )


def split_date(first: date, last: date, train_frac: float = 0.7) -> date:
    return date.fromordinal(first.toordinal() + int((last.toordinal() - first.toordinal()) * train_frac))


def split(book: Book, cut: date) -> tuple[list[EodTrade], list[EodTrade]]:
    return [t for t in book.trades if t.entry_date < cut], [t for t in book.trades if t.entry_date >= cut]


def passes(s: Stats, capital: float) -> list[str]:
    """Failed criteria (empty = passes): >=100 trades, PF >= 1.3 after costs,
    positive expectancy in most years, max drawdown < 20% of capital."""
    fails = []
    if s.trades < 100:
        fails.append(f"only {s.trades} trades (<100)")
    if s.profit_factor is None or s.profit_factor < 1.3:
        fails.append(f"PF {s.profit_factor:.2f} < 1.3" if s.profit_factor is not None else "PF n/a")
    if s.years_traded and s.years_positive * 2 <= s.years_traded:
        fails.append(f"positive in only {s.years_positive}/{s.years_traded} years")
    if s.max_dd >= 0.2 * capital:
        fails.append(f"max DD Rs{s.max_dd:,.0f} >= 20% of capital")
    return fails


def _settings(book: Book) -> str:
    return " ".join(f"{k}={v}" for k, v in book.params.items())


def _pf(s: Stats) -> str:
    return f"{s.profit_factor:.2f}" if s.profit_factor is not None else "n/a"


def per_year_lines(book: Book) -> list[str]:
    by_year: dict[int, list[float]] = defaultdict(list)
    for t in book.trades:
        by_year[t.exit_date.year].append(t.rupees_net)
    return [
        f"  {y}: {len(v):3d} trades  net Rs{sum(v):+10,.0f}  per trade Rs{sum(v) / len(v):+8,.0f}"
        for y, v in sorted(by_year.items())
    ]


def grid_table(books: list[Book], cut: date, capital: float, min_train: int = 15) -> list[str]:
    """Rows sorted by TRAIN profit factor; test columns are the out-of-sample check."""
    rows = []
    for b in books:
        tr, te = split(b, cut)
        s_tr, s_te, s_all = stats(tr), stats(te), stats(b.trades, b.equity)
        rows.append((b, s_tr, s_te, s_all))
    rows.sort(key=lambda r: (r[1].trades >= min_train, r[1].profit_factor or 0), reverse=True)
    width = max([len(_settings(b)) for b in books] + [8])
    head = (
        f"{'Settings':{width}} | {'train n':>7} {'PF':>5} {'Rs/trade':>9} | {'test n':>6} {'PF':>5} "
        f"{'Rs/trade':>9} | {'all PF':>6} {'maxDD':>8}  pass"
    )
    out = [head, "-" * len(head)]
    for b, s_tr, s_te, s_all in rows:
        ok = "YES" if not passes(s_all, capital) else "no"
        out.append(
            f"{_settings(b):{width}} | {s_tr.trades:>7} {_pf(s_tr):>5} {s_tr.expectancy:>+9,.0f} | {s_te.trades:>6} "
            f"{_pf(s_te):>5} {s_te.expectancy:>+9,.0f} | {_pf(s_all):>6} {s_all.max_dd:>8,.0f}  {ok}"
        )
    return out


def summary_lines(book: Book, cut: date, capital: float) -> list[str]:
    tr, te = split(book, cut)
    s_all, s_tr, s_te = stats(book.trades, book.equity), stats(tr), stats(te)
    reasons: dict[str, list[float]] = defaultdict(list)
    for t in book.trades:
        reasons[t.reason].append(t.rupees_net)
    modelled = book.modelled_legs / book.priced_legs if book.priced_legs else 0.0
    fails = passes(s_all, capital)
    lines = [
        f"{book.label}: " + ", ".join(f"{k}={v}" for k, v in book.params.items()),
        f"All:   {s_all.trades} trades, win {s_all.win_rate:.0%}, PF {_pf(s_all)}, Rs{s_all.expectancy:+,.0f}/trade, "
        f"net Rs{s_all.net:+,.0f}, max DD Rs{s_all.max_dd:,.0f} ({s_all.max_dd / capital:.1%} of Rs{capital:,.0f})",
        f"Train (before {cut}): {s_tr.trades} trades, PF {_pf(s_tr)}, Rs{s_tr.expectancy:+,.0f}/trade",
        f"Test  (from {cut}):   {s_te.trades} trades, PF {_pf(s_te)}, Rs{s_te.expectancy:+,.0f}/trade",
        "Exits: " + ", ".join(f"{k} {len(v)} (Rs{sum(v):+,.0f})" for k, v in sorted(reasons.items())),
        "Per year:",
        *per_year_lines(book),
        f"Legs priced by model (no NSE quote): {modelled:.1%}",
        "PASS" if not fails else "FAIL: " + "; ".join(fails),
    ]
    return lines
