"""Daily pre-market movers report -> email.

Runs at 9:00 AM IST, before the market opens (9:15), so it's built from the
most recent completed session's data: top 5 gainers, top 5 losers, and top 5
by volume - a watchlist of names that were active and could keep moving today.

Config (repo-root .env, gitignored):
    MOVERS_MAIL_TO = comma-separated recipient list
    (reuses MOMENTUM_DB_URL/DATABASE_URL and the MAIL_SMTP_*/MAIL_USERNAME/
    MAIL_PASSWORD/MAIL_FROM settings already used by momentum_email_report.py)

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

# Below this session volume, a big % move is noise (a handful of shares
# trading), not something worth watching - so gainers/losers apply this
# floor. The volume list has no floor - it ranks on volume directly.
MIN_VOLUME_FOR_MOVERS = 50_000


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
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    lines = [f"Daily movers report - {date.today().isoformat()}", "=" * 44, ""]
    try:
        async with sf() as s:
            dates = (
                await s.execute(text("select distinct trade_date from historical_prices order by trade_date desc limit 2"))
            ).scalars().all()
            if len(dates) < 2:
                lines.append("Not enough price history yet to compute movers.")
                return False, "\n".join(lines)
            latest_date, prev_date = dates[0], dates[1]
            is_stale = latest_date < _last_expected_trading_day(date.today())
            if is_stale:
                lines.append(
                    f"WARNING: price data is stale - last session {latest_date.isoformat()}, expected at least "
                    f"{_last_expected_trading_day(date.today()).isoformat()}."
                )
                lines.append("")

            rows = (
                await s.execute(
                    text(
                        "select s.symbol, s.name, hp1.close, hp1.volume, hp0.close "
                        "from historical_prices hp1 "
                        "join stocks s on s.id = hp1.stock_id "
                        "join historical_prices hp0 on hp0.stock_id = hp1.stock_id and hp0.trade_date = :prev "
                        "where hp1.trade_date = :latest and s.is_active = true and hp0.close > 0"
                    ),
                    {"latest": latest_date, "prev": prev_date},
                )
            ).all()

            movers = []
            for symbol, name, close, volume, prev_close in rows:
                close, prev_close, volume = float(close), float(prev_close), int(volume)
                pct = (close / prev_close - 1) * 100
                movers.append((symbol, name, close, volume, pct))

            lines.append(f"Session {latest_date.isoformat()} vs previous close {prev_date.isoformat()}")
            lines.append("")

            liquid = [m for m in movers if m[3] >= MIN_VOLUME_FOR_MOVERS]

            gainers = sorted(liquid, key=lambda m: m[4], reverse=True)[:5]
            lines.append("TOP 5 GAINERS")
            for symbol, name, close, volume, pct in gainers:
                lines.append(f"  {symbol:14} +{pct:>5.1f}%  Rs{close:>9,.2f}   vol {volume:>10,}")
            lines.append("")

            losers = sorted(liquid, key=lambda m: m[4])[:5]
            lines.append("TOP 5 LOSERS")
            for symbol, name, close, volume, pct in losers:
                lines.append(f"  {symbol:14} {pct:>6.1f}%  Rs{close:>9,.2f}   vol {volume:>10,}")
            lines.append("")

            by_volume = sorted(movers, key=lambda m: m[3], reverse=True)[:5]
            lines.append("TOP 5 BY VOLUME")
            for symbol, name, close, volume, pct in by_volume:
                lines.append(f"  {symbol:14} vol {volume:>10,}   Rs{close:>9,.2f}   ({'+' if pct >= 0 else ''}{pct:.1f}%)")
    finally:
        await engine.dispose()
    lines += [
        "",
        f"Gainers/losers exclude names under {MIN_VOLUME_FOR_MOVERS:,} shares volume (too thin to be meaningful).",
        "Pre-market watchlist from the previous session. Not investment advice.",
    ]
    return is_stale, "\n".join(lines)


def send(subject: str, body: str) -> None:
    notify_whatsapp(subject, body)  # independent of email: still sent if SMTP is unset or fails
    user = os.environ.get("MAIL_USERNAME")
    pw = (os.environ.get("MAIL_PASSWORD") or "").replace(" ", "")
    to_raw = os.environ.get("MOVERS_MAIL_TO", "aniudupa15@gmail.com")
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
    send(f"Daily movers report - {date.today().isoformat()}{tag}", body)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
