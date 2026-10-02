"""Weekly (7-day lookback) momentum picks report -> email.

Same format and recipients as momentum_email_report.py, but ranks on a
7-day trailing return instead of 30-day, and runs daily instead of
weekly/monthly. NOTE: the 30-day/monthly config is the one with backtest
backing (see services/trading_service/momentum/ranking.py) - this 7-day
variant is an unvalidated short-term cut of the same signal, not a
separately-tested strategy. Treat it as informational, more so than the
monthly picks.

Config: same .env keys as momentum_email_report.py (MOMENTUM_DB_URL,
MAIL_SMTP_*, MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, MAIL_TO).

If MAIL_USERNAME/MAIL_PASSWORD are absent it prints the report instead of
sending (dry run).
"""

import asyncio
import os
import smtplib
import sys
from datetime import date
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scripts.whatsapp_notify import notify as notify_whatsapp  # noqa: E402
from services.trading_service.momentum.ranking import compute_ranking, confidence_for_rank  # noqa: E402

LOOKBACK_DAYS = 7


async def build_report() -> str:
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    lines = [f"Weekly momentum picks report - {date.today().isoformat()}", "=" * 44, ""]
    try:
        async with sf() as s:
            latest_trade_date = (await s.execute(text("select max(trade_date) from historical_prices"))).scalar()
            if latest_trade_date is not None:
                stale_days = (date.today() - latest_trade_date).days
                if stale_days > 4:  # >4 covers ordinary weekends without false-alarming
                    lines.append(
                        f"WARNING: price data is stale - last update {latest_trade_date.isoformat()} "
                        f"({stale_days} days ago). Rankings below may be outdated."
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
            lines.append(f"Top {LOOKBACK_DAYS}-day momentum picks (short-term, unvalidated - see note below):")
            for i, pk in enumerate(picks, 1):
                held_tag = " (held)" if pk.symbol in held_symbols else ""
                lines.append(
                    f"  {i:>2}. {pk.symbol:14} BUY  +{pk.trailing_return_pct:>5.1f}% ({LOOKBACK_DAYS}d)  "
                    f"Rs{pk.last_close:,.0f}  conf {confidence_for_rank(i)}%{held_tag}"
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
        f"Paper trading. Confidence % is a rank label, not a probability. This {LOOKBACK_DAYS}-day cut is NOT "
        "the backtested strategy (that's the 30-day/monthly picks report) - informational only.",
    ]
    return "\n".join(lines)


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
    send(f"Weekly momentum picks report - {date.today().isoformat()}", await build_report())


if __name__ == "__main__":
    asyncio.run(main())
