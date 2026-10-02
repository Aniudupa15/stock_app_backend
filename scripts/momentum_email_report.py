"""Local momentum picks report -> email.

Queries the live database, builds a report - the 30-day momentum picks (the
backtested/validated config, plus BUY/SELL vs current holdings) and a 7-day
short-term cut of the same signal (informational only, not separately
validated) - and emails it. Run manually or daily on a schedule.

Config (put in the repo-root .env, which is gitignored):
    MOMENTUM_DB_URL   = postgresql+asyncpg://...   (your Neon URL; falls back to DATABASE_URL)
    MAIL_SMTP_HOST    = smtp.gmail.com             (default)
    MAIL_SMTP_PORT    = 587                        (default)
    MAIL_USERNAME     = your-gmail@gmail.com       (the SENDING account)
    MAIL_PASSWORD     = <gmail app password>       (NOT your normal password - see README note)
    MAIL_FROM         = your-gmail@gmail.com        (defaults to MAIL_USERNAME)
    MAIL_TO           = aniudupa15@gmail.com        (comma-separated for multiple recipients)

If MAIL_USERNAME/MAIL_PASSWORD are absent it prints the report instead of
sending (dry run) - handy for testing before you add the app password.
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


def _last_expected_trading_day(today: date) -> date:
    """The most recent weekday before today - what we expect historical_prices to
    at least cover. Weekday-aware so a Monday check expects Friday's data, not a
    false 'stale' flag over the weekend - but a genuine 1-trading-day gap on any
    other day of the week still gets caught (unlike a blunt >N-calendar-days rule)."""
    from datetime import timedelta

    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


async def build_report() -> tuple[bool, str]:
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    lines = [f"Momentum picks report - {date.today().isoformat()}", "=" * 44, ""]
    is_stale = False
    try:
        async with sf() as s:
            latest_trade_date = (await s.execute(text("select max(trade_date) from historical_prices"))).scalar()
            expected = _last_expected_trading_day(date.today())
            if latest_trade_date is None or latest_trade_date < expected:
                is_stale = True
                last_str = latest_trade_date.isoformat() if latest_trade_date else "never"
                lines.append(
                    f"WARNING: price data is stale - last update {last_str}, expected at least "
                    f"{expected.isoformat()}. Rankings below may be outdated."
                )
                lines.append("")

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

            weekly_picks = await compute_ranking(s, top=10, lookback=7)
            lines.append("Top 7-day momentum picks (short-term cut, NOT separately validated - informational only):")
            for i, pk in enumerate(weekly_picks, 1):
                held_tag = " (held)" if pk.symbol in held_symbols else ""
                lines.append(
                    f"  {i:>2}. {pk.symbol:14} BUY  +{pk.trailing_return_pct:>5.1f}% (7d)  "
                    f"Rs{pk.last_close:,.0f}  conf {confidence_for_rank(i)}%{held_tag}"
                )
            lines.append("")

            picks = await compute_ranking(s, top=10)
            lines.append("This month's top momentum picks (hold ~1 month, rebalance monthly):")
            for i, pk in enumerate(picks, 1):
                held_tag = " (held)" if pk.symbol in held_symbols else ""
                lines.append(
                    f"  {i:>2}. {pk.symbol:14} BUY  +{pk.trailing_return_pct:>5.1f}% (30d)  "
                    f"Rs{pk.last_close:,.0f}  conf {confidence_for_rank(i)}%{held_tag}"
                )
    finally:
        await engine.dispose()
    lines += ["", "Paper trading. Validated momentum factor, but bumpy month-to-month. Not investment advice."]
    return is_stale, "\n".join(lines)


def send(subject: str, body: str) -> None:
    notify_whatsapp(subject, body)  # independent of email: still sent if SMTP is unset or fails
    user = os.environ.get("MAIL_USERNAME")
    pw = (os.environ.get("MAIL_PASSWORD") or "").replace(" ", "")
    to_raw = os.environ.get("MAIL_TO", "aniudupa15@gmail.com")
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
    is_stale, body = await build_report()
    tag = " - OUTDATED" if is_stale else ""
    send(f"Momentum picks report - {date.today().isoformat()}{tag}", body)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
