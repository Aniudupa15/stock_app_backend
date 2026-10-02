"""Daily status of the LIVE Zerodha momentum bot -> WhatsApp + email.

Between monthly rebalances the bot is silent and GTT stop-losses fire on
Zerodha's side, so this marks the bot's ledger (scripts/state/) to the latest
NSE close from the DB: value, P&L per stock, distance to each stop, a warning
when a close is at/below the stop trigger (GTT very likely fired - check Kite),
and when the next rebalance is due. Needs no Kite login.

    python scripts/zerodha_bot_status.py      # scheduled weekdays 18:45, after the price sync
"""

import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
if os.environ.get("BOT_MAIL_TO"):
    os.environ["MAIL_TO"] = os.environ["BOT_MAIL_TO"]

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from scripts.momentum_email_report import send  # noqa: E402

STATE_FILE = ROOT / "scripts" / "state" / "zerodha_bot_state.json"


async def latest_closes(symbols: list[str]) -> tuple[dict[str, Decimal], date | None]:
    engine = create_async_engine(os.environ["MOMENTUM_DB_URL"])
    try:
        async with engine.connect() as c:
            latest = (await c.execute(text("select max(trade_date) from historical_prices"))).scalar()
            rows = (
                await c.execute(
                    text(
                        "select s.symbol, hp.close from historical_prices hp join stocks s on s.id = hp.stock_id "
                        "where hp.trade_date = :d and s.symbol = any(:syms)"
                    ),
                    {"d": latest, "syms": symbols},
                )
            ).all()
    finally:
        await engine.dispose()
    return {sym: Decimal(str(close)) for sym, close in rows}, latest


def next_weekday_on_or_after(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def build(state: dict, closes: dict[str, Decimal], as_of: date | None) -> tuple[str, str]:
    stop_pct = Decimal(os.environ.get("BOT_STOP_LOSS_PCT", "15"))
    ledger = state.get("ledger", {})
    lines = [f"Prices as of {as_of} close", ""]
    cost_total = value_total = Decimal(0)
    alerts = []
    rows = []
    for sym, pos in ledger.items():
        qty, avg = int(pos["qty"]), Decimal(str(pos["avg_price"]))
        close = closes.get(sym)
        cost = qty * avg
        cost_total += cost
        if close is None:
            value_total += cost
            rows.append((Decimal(0), f"{sym:11} x{qty:<3} no price today"))
            continue
        value = qty * close
        value_total += value
        pnl_pct = (close / avg - 1) * 100
        stop = avg * (1 - stop_pct / 100)
        to_stop = (close / stop - 1) * 100
        if close <= stop:
            alerts.append(f"{sym}: close Rs{close:,.2f} <= stop Rs{stop:,.2f} - GTT stop very likely FIRED, check Kite")
        elif to_stop < 5:
            alerts.append(f"{sym}: only {to_stop:.1f}% above its stop (Rs{stop:,.2f})")
        rows.append(
            (
                pnl_pct,
                f"{sym:11} x{qty:<3} Rs{close:>9,.2f}  {pnl_pct:>+6.1f}%  Rs{value - cost:>+8,.0f}  stop {to_stop:>4.0f}% away",
            )
        )
    rows.sort(key=lambda r: r[0], reverse=True)
    pnl = value_total - cost_total
    pnl_pct = (pnl / cost_total * 100) if cost_total else Decimal(0)
    lines.append(f"Invested Rs{cost_total:,.0f}  ->  now Rs{value_total:,.0f}  ({pnl:+,.0f} / {pnl_pct:+.2f}%)")
    lines.append("")
    lines += [r[1] for r in rows] or ["No bot holdings."]
    if alerts:
        lines += ["", "ALERTS:", *[f"  {a}" for a in alerts]]
    last = state.get("last_live_date")
    if last:
        gap = int(os.environ.get("BOT_MIN_DAYS_BETWEEN", "20"))
        due = next_weekday_on_or_after(date.fromisoformat(last) + timedelta(days=gap))
        lines += ["", f"Last rebalance {last}. Next one due from {due} - you'll get a login-to-approve message."]
    lines += ["", "Live Zerodha bot. Not investment advice."]
    tag = " - ALERT" if any("FIRED" in a for a in alerts) else ""
    subject = f"Zerodha bot: Rs{value_total:,.0f} ({pnl_pct:+.2f}%){tag}"
    return subject, "\n".join(lines)


def main() -> None:
    if not STATE_FILE.exists():
        print("No bot state yet - nothing to report")
        return
    state = json.loads(STATE_FILE.read_text())
    if not state.get("ledger"):
        print("Bot holds nothing - nothing to report")
        return
    closes, as_of = asyncio.run(latest_closes(list(state["ledger"])))
    subject, body = build(state, closes, as_of)
    print(subject + "\n" + body)
    send(subject, body)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
