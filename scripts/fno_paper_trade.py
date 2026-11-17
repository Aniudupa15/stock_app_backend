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


SESSION_FILE = ROOT / "scripts" / "state" / "kite_session.json"  # gitignored; token dies 06:00 IST


def _cached_token() -> str | None:
    try:
        data = json.loads(SESSION_FILE.read_text())
    except (OSError, ValueError):
        return None
    if datetime.fromisoformat(data["expires_at"]) > datetime.now(IST):
        return data["access_token"]
    return None


def _save_token(token: str) -> None:
    now = datetime.now(IST)
    six = now.replace(hour=6, minute=0, second=0, microsecond=0)
    expires = six if now < six else six + timedelta(days=1)
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps({"access_token": token, "expires_at": expires.isoformat()}))


def _wait_for_redirect(host: str, port: int, deadline: datetime, accept, pasted) -> bool:
    """Catch Kite's redirect (/kite/callback?request_token=...) on host:port.
    Returns early if the user pasted a token instead. If another program
    (e.g. the rebalance bot) already holds the port, rely on pasting."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = parse_qs(urlparse(self.path).query)
            tok = q.get("request_token", [None])[0]
            ok = bool(tok) and not done.is_set() and accept(tok)
            if ok:
                done.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            msg = b"Login received - you can close this tab." if ok or done.is_set() else b"Login failed - try again."
            self.wfile.write(b"<h2>" + msg + b"</h2>")

        def log_message(self, *args):
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        log(f"Port {port} is busy (is the rebalance bot running?) - paste the redirected URL instead")
        server = None
    if server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while not (done.is_set() or pasted.is_set()) and datetime.now(IST) < deadline:
            done.wait(timeout=1)
    finally:
        if server:
            server.shutdown()
    return done.is_set() or pasted.is_set()


def login(api_key: str, api_secret: str, access_token: str | None, request_token: str | None = None):
    """Kite session for MARKET DATA. Order of preference: explicit access
    token, today's cached token, a pasted request_token, then the browser
    login (catching the redirect locally, via ngrok when NGROK_DOMAIN is set,
    or by pasting the redirected URL / request_token into this window)."""
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    token = access_token or _cached_token()
    if token:
        kite.set_access_token(token)
        try:
            kite.profile()
            log("Using existing Kite session")
            return kite
        except Exception:
            log("Saved Kite session is no longer valid - logging in again")

    def accept(tok: str) -> bool:
        try:
            data = kite.generate_session(tok, api_secret=api_secret)
            _save_token(data["access_token"])
            return True
        except Exception as exc:
            log(f"Rejected a login token ({type(exc).__name__}: {exc})")
            return False

    if request_token:
        if not accept(request_token):
            raise SystemExit("That request_token was rejected (they are single-use and expire in minutes)")
        return kite

    import threading

    from scripts.zerodha_rebalance import start_ngrok

    port = int(os.environ.get("KITE_CALLBACK_PORT", "5010"))
    bind = os.environ.get("KITE_CALLBACK_BIND", "127.0.0.1")
    ngrok_domain = os.environ.get("NGROK_DOMAIN", "").strip().removeprefix("https://").strip("/")
    tunnel = None
    if ngrok_domain:
        try:
            tunnel = start_ngrok(ngrok_domain, port)
        except Exception as exc:
            log(f"ngrok not started ({exc}) - paste the redirected URL instead")
    deadline = datetime.now(IST) + timedelta(minutes=30)
    log(
        f"Open this link to log in to Kite (waiting until {deadline:%H:%M} IST):\n  {kite.login_url()}\n"
        "If the page after login doesn't load, copy the FULL address from the browser bar "
        "(it contains request_token=...) and paste it here, then press Enter."
    )

    pasted = threading.Event()

    def read_paste() -> None:
        from urllib.parse import parse_qs, urlparse

        for line in sys.stdin:
            line = line.strip()
            tok = parse_qs(urlparse(line).query).get("request_token", [line])[0]
            if tok and accept(tok):
                pasted.set()
                return

    threading.Thread(target=read_paste, daemon=True).start()
    try:
        ok = _wait_for_redirect(bind, port, deadline, accept, pasted)
    finally:
        if tunnel:
            tunnel.terminate()
    if not (ok or pasted.is_set()):
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
    p.add_argument("--request-token", help="request_token from the Kite login redirect URL")
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
    kite = ReadOnlyKite(login(api_key, api_secret, args.access_token, args.request_token))

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
