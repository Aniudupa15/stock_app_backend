"""Intraday (1-day lookback) momentum picks report -> email.

Same format as weekly_momentum_email.py/momentum_email_report.py, but ranks
on the single latest day's return - the shortest cut of the signal
available from daily EOD Bhavcopy data (there's no intraday/tick feed in
this system, so "intraday" here means day-over-day, not minute-by-minute).
NOTE: only the 30-day/monthly config is backtested (see
services/trading_service/momentum/ranking.py). A walk-forward ML experiment
(8 features, HistGradientBoostingClassifier, 383 folds) confirmed this
1-day horizon has essentially NO predictive signal: Information Coefficient
+0.0006 (~zero), only 51% of folds even directionally right (a coin flip).
It looked good on basket-level CAGR/Sharpe alone, but that was a compounding
artifact of a near-zero mean edge, not real skill - and ignores the
transaction costs a daily-rebalance strategy would actually incur. This
report is purely informational.

Unlike the weekly/monthly reports, this one is sent to aniudupa15@gmail.com
only, and ONLY IF the price data is fresh - if it's stale, it skips sending
entirely (no warning email, just prints why and exits) rather than sending
a possibly-misleading 1-day return computed from old data.

Config: same .env keys as momentum_email_report.py (MOMENTUM_DB_URL,
MAIL_SMTP_*, MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM), plus:
    INTRADAY_MAIL_TO = aniudupa15@gmail.com   (defaults to this)

If MAIL_USERNAME/MAIL_PASSWORD are absent it prints the report instead of
sending (dry run).
"""

import asyncio
import os
import smtplib
import sys
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scripts.whatsapp_notify import notify as notify_whatsapp  # noqa: E402

from services.trading_service.momentum.ranking import compute_ranking, confidence_for_rank  # noqa: E402

LOOKBACK_DAYS = 1


def _last_expected_trading_day(today: date) -> date:
    """The most recent weekday before today - what we expect historical_prices to
    at least cover. Weekday-aware so a Monday check expects Friday's data, not a
    false 'stale' flag over the weekend - but a genuine 1-trading-day gap on any
    other day of the week still gets caught."""
    from datetime import timedelta

    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


async def build_report() -> tuple[bool, str]:
    """Returns (data_is_valid, body). body is '' when data_is_valid is False."""
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    lines = [f"Intraday momentum picks report - {date.today().isoformat()}", "=" * 44, ""]
    try:
        async with sf() as s:
            latest_trade_date = (await s.execute(text("select max(trade_date) from historical_prices"))).scalar()
            if latest_trade_date is None or latest_trade_date < _last_expected_trading_day(date.today()):
                return False, ""

            recent_stoplosses = (
                await s.execute(
                    text(
                        "select symbol, entry_price, exit_price, created_at from trading.trades "
                        "where exit_reason = 'STOP_LOSS' and created_at >= now() - interval '3 days' "
                        "order by created_at desc"
                    )
                )
            ).all()
            if recent_stoplosses:
                lines.append("STOPPED OUT (last 3 days):")
                for symbol, entry, exit_p, ts in recent_stoplosses:
                    loss_pct = (float(exit_p) - float(entry)) / float(entry) * 100
                    lines.append(
                        f"  {symbol:14} Rs{float(entry):,.2f} -> Rs{float(exit_p):,.2f}  "
                        f"({loss_pct:+.1f}%)  {ts.date().isoformat()}"
                    )
                lines.append("")

            held_symbols = {
                r[0]
                for r in (
                    await s.execute(text("select distinct symbol from trading.positions where net_qty > 0"))
                ).all()
            }

            picks = await compute_ranking(s, top=10, lookback=LOOKBACK_DAYS)
            top_symbols = {pk.symbol for pk in picks}
            lines.append(f"TOP {LOOKBACK_DAYS}-DAY (INTRADAY) MOMENTUM PICKS")
            lines.append("")
            cols = ("Rank", "Symbol", "Buy Price", "Illustrative Target", "Confidence", "Held")
            widths = (4, 14, 12, 24, 10, 4)
            header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
            lines.append(header)
            lines.append("  ".join("-" * w for w in widths))
            for i, pk in enumerate(picks, 1):
                buy_price = pk.last_close
                pct = pk.trailing_return_pct
                target_price = buy_price * (1 + pct / 100)
                row = (
                    f"{i:>4}",
                    pk.symbol.ljust(14),
                    f"Rs{buy_price:>9,.2f}".ljust(12),
                    f"Rs{target_price:>9,.2f} ({'+' if pct >= 0 else ''}{pct:.1f}%)".ljust(24),
                    f"{confidence_for_rank(i)}%".ljust(10),
                    ("Yes" if pk.symbol in held_symbols else "-").ljust(4),
                )
                lines.append("  ".join(row))
            lines.append("")
            lines.append(
                "Illustrative Target = Buy Price projected forward by the SAME % move that already happened "
                f"over the last {LOOKBACK_DAYS} day. It is NOT a prediction, forecast, or backtested exit level - "
                "just a simple what-if for reference. Confidence is a rank label (1st=highest), not a probability."
            )
            lines.append("")

            if held_symbols:
                sell = sorted(held_symbols - top_symbols)
                buy_new = sorted(top_symbols - held_symbols)
                lines.append("Rebalance actions vs current holdings:")
                lines.append(f"  SELL (held, dropped out of top 10): {', '.join(sell) if sell else 'none'}")
                lines.append(f"  BUY (new entries, not yet held):    {', '.join(buy_new) if buy_new else 'none'}")
    finally:
        await engine.dispose()
    lines += [
        "",
        f"Paper trading. This {LOOKBACK_DAYS}-day cut is NOT the backtested strategy "
        "(that's the 30-day/monthly picks report) - informational only.",
    ]
    return True, "\n".join(lines)


def send(subject: str, body: str) -> None:
    notify_whatsapp(subject, body)  # independent of email: still sent if SMTP is unset or fails
    user = os.environ.get("MAIL_USERNAME")
    pw = (os.environ.get("MAIL_PASSWORD") or "").replace(" ", "")
    to_raw = os.environ.get("INTRADAY_MAIL_TO", "aniudupa15@gmail.com")
    to = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
    if not user or not pw:
        print("[dry run: MAIL_USERNAME/MAIL_PASSWORD not set - printing report]\n")
        print(body)
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("MAIL_FROM", user)
    msg["To"] = ", ".join(to)
    msg.set_content(body)
    host = os.environ.get("MAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("MAIL_SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, pw)
        smtp.send_message(msg)
    print(f"Emailed report to {', '.join(to)}")


async def main() -> None:
    is_valid, body = await build_report()
    if not is_valid:
        print(f"{datetime.now().isoformat()} | SKIPPED: price data is stale or missing - not sending intraday report")
        return
    send(f"Intraday momentum picks report - {date.today().isoformat()}", body)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
