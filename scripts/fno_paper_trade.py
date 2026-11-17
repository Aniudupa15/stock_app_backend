"""PAPER-trade NSE futures & options intraday on live Zerodha prices.

No real orders are ever placed: Kite is used for market data only (quotes,
instrument master) through a read-only wrapper, and every order is filled by
the paper venue with Zerodha's F&O charge schedule and a margin check.

Usage (on the PC with your Kite Connect app, during market hours):

    python scripts/fno_paper_trade.py --underlying NIFTY --instrument FUT --lots 1
    python scripts/fno_paper_trade.py --underlying BANKNIFTY --instrument CE --lots 1
    python scripts/fno_paper_trade.py --underlying NIFTY --instrument PE --capital 200000

Login: if KITE_ACCESS_TOKEN is set (or --access-token is given) it is used
directly; otherwise this prints the Kite login link and waits for the
redirect on KITE_CALLBACK_PORT, exactly like zerodha_rebalance.py.

Config (repo-root .env): KITE_API_KEY, KITE_API_SECRET, KITE_CALLBACK_PORT,
KITE_CALLBACK_BIND, NSE_HOLIDAYS (comma-separated YYYY-MM-DD).

Output: a journal JSON in scripts/state/fno_paper/ and a printed summary
(also emailed with --email).
"""

from __future__ import annotations

import argparse
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

from libs.trading_calendar.calendar import IST  # noqa: E402
from services.trading_service.fno_paper import ReadOnlyKite, SessionConfig, SessionReport, run_session  # noqa: E402

JOURNAL_DIR = ROOT / "scripts" / "state" / "fno_paper"


def log(msg: str) -> None:
    print(f"{datetime.now(IST).isoformat(timespec='seconds')} | {msg}", flush=True)


def login(api_key: str, api_secret: str, access_token: str | None):
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    if access_token:
        kite.set_access_token(access_token)
        return kite

    from scripts.zerodha_rebalance import wait_for_login

    port = int(os.environ.get("KITE_CALLBACK_PORT", "5010"))
    bind = os.environ.get("KITE_CALLBACK_BIND", "127.0.0.1")
    deadline = datetime.now(IST) + timedelta(minutes=30)
    log(f"Open this link to log in to Kite (waiting until {deadline:%H:%M} IST):\n  {kite.login_url()}")

    def accept(token: str) -> bool:
        try:
            kite.generate_session(token, api_secret=api_secret)
            return True
        except Exception as exc:
            log(f"Rejected a login callback ({type(exc).__name__}: {exc}) - still waiting")
            return False

    if not wait_for_login(bind, port, deadline, accept):
        raise SystemExit("No Kite login received - exiting")
    log("Kite session established (market data only)")
    return kite


def journal(report: SessionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "contract": report.contract.tradingsymbol,
        "kind": report.contract.kind.value,
        "lot_size": report.contract.lot_size,
        "expiry": report.contract.expiry.isoformat(),
        "starting_cash": str(report.starting_cash),
        "final_cash": str(report.final_cash),
        "net_pnl": str(report.net_pnl),
        "fills": [
            {
                "order_id": f.order_id,
                "side": f.side.value,
                "qty": f.qty,
                "price": str(f.price),
                "charges": f.charges.as_dict(),
                "ts": f.ts.isoformat() if f.ts else None,
            }
            for f in report.fills
        ],
        "trades": [
            {
                "qty": t.qty,
                "entry": str(t.entry_price),
                "exit": str(t.exit_price),
                "gross": str(t.pnl_gross),
                "charges": str(t.charges_total),
                "net": str(t.pnl_net),
                "reason": t.exit_reason.value if t.exit_reason else None,
                "entry_ts": t.entry_ts.isoformat() if t.entry_ts else None,
                "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
            }
            for t in report.trades
        ],
        "notes": report.notes,
    }
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Intraday F&O paper trading on live Zerodha prices (no real orders)")
    p.add_argument("--underlying", default="NIFTY", help="NIFTY, BANKNIFTY, FINNIFTY, RELIANCE, ...")
    p.add_argument("--instrument", default="FUT", choices=["FUT", "CE", "PE"], help="nearest future or ATM option")
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--capital", type=Decimal, default=Decimal("500000"), help="paper capital in rupees")
    p.add_argument("--bar-seconds", type=int, default=60)
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--max-trades", type=int, default=5)
    p.add_argument("--access-token", default=os.environ.get("KITE_ACCESS_TOKEN"))
    p.add_argument("--email", action="store_true", help="email the summary (uses the existing mail settings)")
    args = p.parse_args()

    holidays = frozenset(
        date.fromisoformat(d.strip()) for d in os.environ.get("NSE_HOLIDAYS", "").split(",") if d.strip()
    )
    config = SessionConfig(
        underlying=args.underlying.upper(),
        instrument=args.instrument,
        lots=args.lots,
        capital=args.capital,
        bar_seconds=args.bar_seconds,
        poll_seconds=args.poll_seconds,
        max_trades=args.max_trades,
        holidays=holidays,
    )
    api_key = os.environ.get("KITE_API_KEY") or sys.exit("Missing KITE_API_KEY in .env")
    api_secret = os.environ.get("KITE_API_SECRET") or sys.exit("Missing KITE_API_SECRET in .env")
    kite = ReadOnlyKite(login(api_key, api_secret, args.access_token))

    report = asyncio.run(run_session(kite, config, log=log))
    path = JOURNAL_DIR / f"{datetime.now(IST):%Y-%m-%d}_{report.contract.tradingsymbol}.json"
    journal(report, path)
    summary = "\n".join(report.lines())
    log("Session finished\n" + summary + f"\nJournal: {path}")
    if args.email:
        from scripts.momentum_email_report import send

        send(f"[PAPER] F&O session {report.contract.tradingsymbol}: Rs{report.net_pnl:+,.0f}", summary)


if __name__ == "__main__":
    main()
