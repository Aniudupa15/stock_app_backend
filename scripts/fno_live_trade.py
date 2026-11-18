"""LIVE intraday NIFTY put trading (backtested `pe-trend` rules) on Zerodha.

    python scripts/fno_live_trade.py                 # DRY RUN (default): live prices, NO orders
    python scripts/fno_live_trade.py --live --email  # REAL ORDERS - needs FNO_LIVE=true too

Real orders are sent only when ALL of these hold:
  1. --live on the command line, AND FNO_LIVE=true in .env (separate from the
     rebalance bot's ZERODHA_LIVE on purpose);
  2. the logged-in Kite account is KITE_USER_ID;
  3. this PC's public IP is in KITE_ALLOWED_IPS (the static IP whitelisted on
     your Kite app - Zerodha rejects API orders from any other IP);
  4. there is no open F&O position and no working bot order.

Risk rails: 1 lot max (--lots/--max-lots), daily loss limit (--daily-loss-limit,
default 3000) - a signal whose stop-loss risk exceeds the remaining limit is
SKIPPED - and a stop-loss order that rests at the exchange from the moment
the entry fills. MIS product, so Zerodha auto-squares off ~15:20 as a backstop.

.env: KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_ALLOWED_IPS, FNO_LIVE,
NGROK_DOMAIN (phone login), NSE_HOLIDAYS. Journal: scripts/state/fno_live/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
# Trading alerts go only to BOT_MAIL_TO (like the rebalance bot), not the shared report list.
if os.environ.get("BOT_MAIL_TO"):
    os.environ["MAIL_TO"] = os.environ["BOT_MAIL_TO"]

from libs.trading_calendar.calendar import IST  # noqa: E402
from scripts.fno_paper_trade import log, login  # noqa: E402
from services.trading_service.fno_live import LiveConfig, LiveReport, preflight, run_live_session  # noqa: E402

JOURNAL_DIR = ROOT / "scripts" / "state" / "fno_live"


def public_ip() -> str | None:
    try:
        return urlopen("https://api.ipify.org", timeout=10).read().decode().strip()
    except Exception:
        return None


def background(send):
    """Alerts (WhatsApp + email via send()) can take minutes when WhatsApp
    retries; never let them pause order management. Non-daemon threads, so
    the final summary still goes out before the process exits."""
    import threading

    def notify(subject: str, body: str) -> None:
        def run() -> None:
            try:
                send(subject, body)
            except Exception as exc:  # alerting must never crash trading
                log(f"alert failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=run, name="alert").start()

    return notify


def journal(report: LiveReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mode": "live" if report.live else "dry-run",
        "contract": report.contract.tradingsymbol,
        "net_pnl": str(report.net_pnl),
        "trades": [
            {
                "qty": t.qty,
                "entry": str(t.entry),
                "exit": str(t.exit) if t.exit is not None else None,
                "charges": str(t.charges),
                "net": str(t.net),
                "reason": t.reason,
                "entry_ts": t.entry_ts.isoformat() if t.entry_ts else None,
                "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
            }
            for t in report.trades
        ],
        "skipped": report.skipped,
        "orders": report.orders,
        "notes": report.notes,
    }
    path.write_text(json.dumps(data, indent=2, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description="Live NIFTY put trading with the backtested pe-trend rules")
    p.add_argument("--check", action="store_true", help="only run the live pre-flight checks, then exit (no trading)")
    p.add_argument("--live", action="store_true", help="send REAL orders (also needs FNO_LIVE=true in .env)")
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--max-lots", type=int, default=1)
    p.add_argument("--daily-loss-limit", type=Decimal, default=Decimal("3000"))
    p.add_argument(
        "--retry-after-skip",
        action="store_true",
        help="after a too-expensive signal keep watching for a cheaper one (default: done for the day)",
    )
    p.add_argument(
        "--breakeven-pct",
        type=Decimal,
        default=Decimal("20"),
        help="raise the stop to the entry price once up this %% (0 = off)",
    )
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument(
        "--email", action="store_true", help="alerts by WhatsApp + email: login link, entries, exits, errors, summary"
    )
    p.add_argument("--login-wait-minutes", type=int, default=120)
    p.add_argument("--access-token", default=os.environ.get("KITE_ACCESS_TOKEN"))
    args = p.parse_args()

    if args.lots > args.max_lots:
        sys.exit(f"--lots {args.lots} exceeds --max-lots {args.max_lots}")
    live = args.live and os.environ.get("FNO_LIVE", "false").strip().lower() == "true"
    if args.live and not live:
        sys.exit("--live given but FNO_LIVE is not 'true' in .env - refusing to send orders")

    notify = None
    if args.email:
        notify = background(__import__("scripts.momentum_email_report", fromlist=["send"]).send)

    api_key = os.environ.get("KITE_API_KEY") or sys.exit("Missing KITE_API_KEY in .env")
    api_secret = os.environ.get("KITE_API_SECRET") or sys.exit("Missing KITE_API_SECRET in .env")
    kite = login(api_key, api_secret, args.access_token, wait_minutes=args.login_wait_minutes, notify=notify)

    if live or args.check:
        allowed = {ip.strip() for ip in os.environ.get("KITE_ALLOWED_IPS", "").split(",") if ip.strip()}
        pf = preflight(kite, expected_user=os.environ.get("KITE_USER_ID"), allowed_ips=allowed, public_ip=public_ip())
        log("Pre-flight:\n  " + "\n  ".join(pf.lines))
        if args.check:
            fno = os.environ.get("FNO_LIVE", "false").strip().lower() == "true"
            log(
                f"FNO_LIVE={'true' if fno else 'false'} -> "
                + ("live orders possible with --live" if fno else "dry run only")
            )
            sys.exit(0 if pf.ok else 1)
        if not pf.ok:
            if notify:
                notify("[LIVE] F&O bot BLOCKED by pre-flight checks", "\n".join(pf.lines))
            sys.exit("Pre-flight failed - no orders sent")

    holidays = frozenset(
        date.fromisoformat(d.strip()) for d in os.environ.get("NSE_HOLIDAYS", "").split(",") if d.strip()
    )
    config = LiveConfig(
        underlying=args.underlying.upper(),
        lots=args.lots,
        max_lots=args.max_lots,
        daily_loss_limit=args.daily_loss_limit,
        breakeven_pct=args.breakeven_pct or None,
        retry_after_skip=args.retry_after_skip,
        live=live,
        holidays=holidays,
    )
    report = asyncio.run(run_live_session(kite, config, log=log, notify=notify))
    path = (
        JOURNAL_DIR / f"{datetime.now(IST):%Y-%m-%d}_{'live' if live else 'dry'}_{report.contract.tradingsymbol}.json"
    )
    journal(report, path)
    log("Session finished\n" + "\n".join(report.lines()) + f"\nJournal: {path}")


if __name__ == "__main__":
    main()
