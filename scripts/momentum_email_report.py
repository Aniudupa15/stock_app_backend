"""Local momentum-portfolio report -> email.

Queries the live database, builds a report (portfolio value + holdings + this
month's top momentum picks), and emails it. Run manually or on a schedule.

Config (put in the repo-root .env, which is gitignored):
    MOMENTUM_DB_URL   = postgresql+asyncpg://...   (your Neon URL; falls back to DATABASE_URL)
    MAIL_SMTP_HOST    = smtp.gmail.com             (default)
    MAIL_SMTP_PORT    = 587                        (default)
    MAIL_USERNAME     = your-gmail@gmail.com       (the SENDING account)
    MAIL_PASSWORD     = <gmail app password>       (NOT your normal password - see README note)
    MAIL_FROM         = your-gmail@gmail.com        (defaults to MAIL_USERNAME)
    MAIL_TO           = aniudupa15@gmail.com        (default recipient)

If MAIL_USERNAME/MAIL_PASSWORD are absent it prints the report instead of
sending (dry run) - handy for testing before you add the app password.
"""

import asyncio
import os
import smtplib
import sys
from datetime import date
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from services.trading_service.momentum.ranking import compute_ranking, confidence_for_rank  # noqa: E402
from services.trading_service.momentum.rebalance import _latest_closes  # noqa: E402
from services.trading_service.persistence.repositories import (  # noqa: E402
    PositionRepository,
    TradingAccountRepository,
)


async def build_report() -> str:
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    lines = [f"Momentum portfolio report - {date.today().isoformat()}", "=" * 44, ""]
    try:
        async with sf() as s:
            closes = await _latest_closes(s)
            account_ids = [
                r[0]
                for r in (
                    await s.execute(text("select distinct account_id from trading.positions where net_qty > 0"))
                ).all()
            ]
            if not account_ids:
                lines.append("No momentum portfolio yet - open the app and tap Rebalance to build one.")
            for aid in account_ids:
                acct = await TradingAccountRepository(s).get(aid)
                if acct is None:
                    continue
                positions = [p for p in await PositionRepository(s).list_for_account(aid) if p.net_qty > 0]
                hv = sum((Decimal(p.net_qty) * closes.get(p.symbol, p.avg_price) for p in positions), Decimal("0"))
                cash = Decimal(acct.virtual_balance) if acct.virtual_balance is not None else Decimal("0")
                total = cash + hv
                start = Decimal(acct.starting_balance) if acct.starting_balance is not None else total
                sign = "+" if total >= start else "-"
                lines.append(f"Portfolio value: Rs{total:,.0f}   ({sign}Rs{abs(total - start):,.0f} since start)")
                lines.append(f"Cash Rs{cash:,.0f} | Invested Rs{hv:,.0f} | {len(positions)} holdings")
                lines.append("")
                for p in sorted(
                    positions, key=lambda x: Decimal(x.net_qty) * closes.get(x.symbol, x.avg_price), reverse=True
                ):
                    px = closes.get(p.symbol, p.avg_price)
                    pnl = (px - p.avg_price) * p.net_qty
                    lines.append(
                        f"  {p.symbol:14} {p.net_qty:>5} @ {p.avg_price:>8.1f} -> {px:>8.1f}   ({'+' if pnl >= 0 else ''}{pnl:,.0f})"
                    )
                lines.append("")

            picks = await compute_ranking(s, top=10)
            lines.append("This month's top momentum picks (BUY, hold ~1 month, rebalance monthly):")
            for i, pk in enumerate(picks, 1):
                lines.append(
                    f"  {i:>2}. {pk.symbol:14} BUY  +{pk.trailing_return_pct:>5.1f}% (30d)  "
                    f"Rs{pk.last_close:,.0f}  conf {confidence_for_rank(i)}%"
                )
    finally:
        await engine.dispose()
    lines += ["", "Paper trading. Validated momentum factor, but bumpy month-to-month. Not investment advice."]
    return "\n".join(lines)


def send(subject: str, body: str) -> None:
    user = os.environ.get("MAIL_USERNAME")
    pw = (os.environ.get("MAIL_PASSWORD") or "").replace(" ", "")
    to = os.environ.get("MAIL_TO", "aniudupa15@gmail.com")
    if not user or not pw:
        print("[dry run: MAIL_USERNAME/MAIL_PASSWORD not set - printing report]\n")
        print(body)
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("MAIL_FROM", user)
    msg["To"] = to
    msg.set_content(body)
    host = os.environ.get("MAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("MAIL_SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, pw)
        smtp.send_message(msg)
    print(f"Emailed report to {to}")


async def main() -> None:
    send(f"Momentum portfolio report - {date.today().isoformat()}", await build_report())


if __name__ == "__main__":
    asyncio.run(main())
