"""LIVE monthly momentum rebalance on Zerodha (Kite Connect).

Flow (run daily ~08:45 by Windows Task Scheduler; does nothing once this
month's rebalance is done, so a skipped login / market holiday just retries
the next weekday):

  1. Rank this month's 30-day momentum top-N from the DB (same signal as the
     email report) and run the data safety gates (fresh, complete, sane
     prices - see live_safety.py). Any failure -> "BLOCKED" email, no login asked.
  2. Email you the planned SELL/BUY symbols + a Kite login link.
  3. Wait (until BOT_LOGIN_DEADLINE_IST) for you to log in. Logging in IS the
     approval - Zerodha tokens expire 06:00 daily, there is no way around a
     daily human login, so we use it as the confirm step.
  4. After 09:20 IST: sell the bot's holdings that dropped out of the top-N,
     then buy the new entries (equal-weight, capped at BOT_CAPITAL_CAP), and
     place a GTT stop-loss for each buy. GTTs live on Zerodha's servers, so the
     stop fires even while this PC is off and without a daily login.
  5. Email the result.

Only stocks the bot bought itself (its ledger in scripts/state/) are ever
sold - your own long-term holdings are untouched.

Config (repo-root .env):
    KITE_API_KEY / KITE_API_SECRET   your Kite Connect app
    KITE_CALLBACK_PORT = 5010        app's redirect URL must be http://127.0.0.1:5010/kite/callback
    NGROK_DOMAIN       = xyz.ngrok-free.app   approve from any device: bot runs an ngrok
                                     tunnel while waiting; Kite redirect URL must be
                                     https://<NGROK_DOMAIN>/kite/callback  (NGROK_PATH optional)
    KITE_CALLBACK_BIND = 127.0.0.1   0.0.0.0 to approve from any device (router forwards
                                     the port here; redirect URL http://<static-ip>:5010/kite/callback)
    BOT_CAPITAL_CAP    = 10000       max rupees the bot ever has deployed
    BOT_TOP_N          = 10
    BOT_STOP_LOSS_PCT  = 15
    BOT_LOGIN_DEADLINE_IST = 15:00
    BOT_MIN_DAYS_BETWEEN = 20        min days between live rebalances
    KITE_USER_ID       = AB1234      optional: refuse to trade if a different account logs in
    NSE_HOLIDAYS       = 2026-10-02,2026-10-21   weekday holidays (else freshness check fails safe)
    BOT_MAIL_TO        = you@x.com   trading emails recipient (defaults to MAIL_TO)
    ZERODHA_LIVE       = false       KILL SWITCH: anything but "true" -> dry run,
                                     computes exact orders after login, places NOTHING
    (MOMENTUM_DB_URL + MAIL_* as for momentum_email_report.py)

Usage:
    python scripts/zerodha_rebalance.py            # scheduled daily
    python scripts/zerodha_rebalance.py --force    # run even if done this month / weekend
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
# Trading emails (holdings, fills) go only to BOT_MAIL_TO, not the shared picks-report list.
if os.environ.get("BOT_MAIL_TO"):
    os.environ["MAIL_TO"] = os.environ["BOT_MAIL_TO"]

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from scripts.momentum_email_report import send  # noqa: E402
from services.trading_service.momentum.live_plan import (  # noqa: E402
    plan_buys,
    plan_sells,
    reconcile_ledger,
    stop_loss_prices,
)
from services.trading_service.momentum.live_safety import (  # noqa: E402
    Check,
    check_account,
    check_buy_orders,
    check_coverage,
    check_freshness,
    filter_picks,
    price_deviations,
)
from services.trading_service.momentum.ranking import compute_ranking  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = ROOT / "scripts" / "state" / "zerodha_bot_state.json"
ORDER_TAG = "momobot"


def env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None or val == "":
        raise SystemExit(f"Missing {name} in .env")
    return val


def log(msg: str) -> None:
    print(f"{datetime.now(IST).isoformat(timespec='seconds')} | {msg}", flush=True)


# ---------------------------------------------------------------- state


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"ledger": {}, "done_month": None, "done_mode": None, "history": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)


# ---------------------------------------------------------------- data


async def load_market_data(top: int, extra_symbols: list[str]) -> dict:
    """Everything the data safety gates + planner need, in one DB round."""
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    try:
        async with async_sessionmaker(bind=engine, expire_on_commit=False)() as s:
            latest = (await s.execute(text("select max(trade_date) from historical_prices"))).scalar()
            counts = (
                await s.execute(
                    text(
                        "select trade_date, count(*) from historical_prices "
                        "where trade_date in (select distinct trade_date from historical_prices "
                        "order by trade_date desc limit 2) group by trade_date order by trade_date desc"
                    )
                )
            ).all()
            # Extra candidates so unsafe picks can be replaced by the next-ranked clean ones.
            candidates = await compute_ranking(s, top=top + 10, lookback=30)
            symbols = [p.symbol for p in candidates] + list(extra_symbols)
            rows = (
                await s.execute(
                    text(
                        "select s.symbol, hp.close from historical_prices hp join stocks s on s.id = hp.stock_id "
                        "where hp.trade_date >= :since and s.symbol = any(:syms) order by s.symbol, hp.trade_date"
                    ),
                    {"since": latest - timedelta(days=70), "syms": symbols},
                )
            ).all()
    finally:
        await engine.dispose()
    series: dict[str, list[float]] = {}
    for sym, close in rows:
        series.setdefault(sym, []).append(float(close))
    return {
        "latest": latest,
        "latest_count": counts[0][1] if counts else 0,
        "prev_count": counts[1][1] if len(counts) > 1 else 0,
        "candidates": [p.symbol for p in candidates],
        "series": series,
        "closes": {sym: Decimal(str(v[-1])) for sym, v in series.items()},
    }


def run_data_checks(data: dict, top: int, today: date) -> tuple[list[str], list[Check], list[str]]:
    holidays = {date.fromisoformat(d.strip()) for d in os.environ.get("NSE_HOLIDAYS", "").split(",") if d.strip()}
    picks, excluded, pick_check = filter_picks(data["candidates"], data["series"], top)
    checks = [
        check_freshness(data["latest"], today, holidays),
        check_coverage(data["latest_count"], data["prev_count"]),
        pick_check,
    ]
    return picks, checks, excluded


# ---------------------------------------------------------------- login


def wait_for_login(host: str, port: int, deadline: datetime, accept) -> bool:
    """Tiny HTTP server catching Kite's redirect with the request_token.

    `accept(token) -> bool` exchanges the token with Zerodha (needs our API
    secret). When the port is reachable from the internet (KITE_CALLBACK_BIND
    = 0.0.0.0 + router port-forward), anyone can hit this URL - so a token is
    only trusted once Zerodha accepts it; junk is answered and ignored.
    """
    done = threading.Event()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            token = q.get("request_token", [None])[0]
            body = b"<h2>Login not successful - try the link again.</h2>"
            if url.path == "/kite/callback" and q.get("status") == ["success"] and token and not done.is_set():
                with lock:
                    if not done.is_set() and accept(token):
                        done.set()
                        body = b"<h2>Login received - rebalance is running. You can close this tab.</h2>"
            elif done.is_set():
                body = b"<h2>Already approved - rebalance is running.</h2>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while not done.is_set() and datetime.now(IST) < deadline:
            done.wait(timeout=5)
    finally:
        server.shutdown()
    return done.is_set()


def start_ngrok(domain: str, port: int) -> subprocess.Popen:
    """Start an ngrok tunnel https://<domain> -> 127.0.0.1:<port> so the Kite
    redirect (which must be HTTPS) reaches this PC from any device. Free ngrok
    gives one static domain; the authtoken lives in ngrok's own config."""
    exe = os.environ.get("NGROK_PATH") or shutil.which("ngrok")
    if not exe:
        raise RuntimeError("ngrok not found - install it or set NGROK_PATH in .env")
    proc = subprocess.Popen(
        [exe, "http", str(port), f"--domain={domain}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if proc.poll() is not None:
            raise RuntimeError(f"ngrok exited with code {proc.returncode} - check authtoken/domain (run it by hand)")
        try:
            tunnels = json.load(urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2))["tunnels"]
            if any(domain in t.get("public_url", "") for t in tunnels):
                log(f"ngrok tunnel up: https://{domain} -> 127.0.0.1:{port}")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("ngrok tunnel did not come up within 30s")


# ---------------------------------------------------------------- broker helpers


# NSE moves surveillance stocks to other series; Kite then lists them as e.g.
# "STLTECH-BE" while Bhavcopy/our DB still says "STLTECH".
_SERIES_SUFFIXES = ("-BE", "-BZ", "-SM", "-ST")


def base_symbol(tradingsymbol: str) -> str:
    for suffix in _SERIES_SUFFIXES:
        if tradingsymbol.endswith(suffix):
            return tradingsymbol[: -len(suffix)]
    return tradingsymbol


def broker_quantities(kite) -> dict[str, int]:
    """Settled + T1 holdings, plus today's CNC position change (buys/sells done earlier today)."""
    qty: dict[str, int] = {}
    for h in kite.holdings():
        sym = base_symbol(h["tradingsymbol"])
        qty[sym] = qty.get(sym, 0) + int(h["quantity"]) + int(h.get("t1_quantity", 0))
    for p in kite.positions().get("day", []):
        if p["exchange"] == "NSE" and p["product"] == "CNC":
            sym = base_symbol(p["tradingsymbol"])
            qty[sym] = qty.get(sym, 0) + int(p["quantity"])
    return qty


def live_prices(kite, symbols: list[str], fallback: dict[str, Decimal]) -> tuple[dict[str, Decimal], str]:
    try:
        data = kite.ltp(*[f"NSE:{s}" for s in symbols])
        prices = {k.split(":", 1)[1]: Decimal(str(v["last_price"])) for k, v in data.items()}
        missing = [s for s in symbols if s not in prices]
        prices.update({s: fallback[s] for s in missing if s in fallback})
        return prices, "live LTP"
    except Exception as exc:  # personal (free) Kite plan has no market-data APIs
        log(f"LTP unavailable ({type(exc).__name__}: {exc}) - sizing from last close")
        return dict(fallback), "last close (no Kite market-data plan)"


def resolve_instruments(kite, symbols: set[str]) -> tuple[dict[str, str], dict[str, Decimal]]:
    """Map our symbols to the tradingsymbol Kite actually lists (plain EQ first,
    then a series-suffixed one like -BE) plus tick size. Symbols missing from the
    map are not tradable on Kite right now."""
    listed = {i["tradingsymbol"]: Decimal(str(i["tick_size"])) for i in kite.instruments("NSE")}
    tsym, ticks = {}, {}
    for s in symbols:
        for candidate in (s, *(s + suffix for suffix in _SERIES_SUFFIXES)):
            if candidate in listed:
                tsym[s], ticks[s] = candidate, listed[candidate]
                break
    return tsym, ticks


def is_permanent_error(status: str) -> bool:
    """Rejections that retrying tomorrow will not fix."""
    return "does not exist" in status or "expired" in status


def place_and_wait(kite, symbol: str, side: str, qty: int, timeout_s: int = 180) -> tuple[int, Decimal, str]:
    """MARKET CNC order with automatic market protection. Returns (filled_qty, avg_price, status)."""
    order_id = kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=symbol,
        transaction_type=side,
        quantity=qty,
        product=kite.PRODUCT_CNC,
        order_type=kite.ORDER_TYPE_MARKET,
        market_protection=-1,
        tag=ORDER_TAG,
    )
    log(f"{side} {symbol} x{qty} -> order {order_id}")
    end = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < end:
        last = kite.order_history(order_id)[-1]
        if last["status"] in ("COMPLETE", "REJECTED", "CANCELLED"):
            break
        time.sleep(2)
    filled = int(last.get("filled_quantity") or 0)
    avg = Decimal(str(last.get("average_price") or 0))
    status = last.get("status", "UNKNOWN")
    if status != "COMPLETE":
        status = f"{status}: {last.get('status_message') or 'not filled in time'}"
    return filled, avg, status


def place_stop_gtt(kite, symbol: str, qty: int, avg: Decimal, ltp: Decimal, stop_pct: Decimal, tick: Decimal) -> int:
    trigger, limit = stop_loss_prices(avg, stop_pct, tick)
    resp = kite.place_gtt(
        trigger_type=kite.GTT_TYPE_SINGLE,
        tradingsymbol=symbol,
        exchange=kite.EXCHANGE_NSE,
        trigger_values=[float(trigger)],
        last_price=float(ltp),
        orders=[
            {
                "transaction_type": kite.TRANSACTION_TYPE_SELL,
                "quantity": qty,
                "order_type": kite.ORDER_TYPE_LIMIT,
                "product": kite.PRODUCT_CNC,
                "price": float(limit),
            }
        ],
    )
    return int(resp["trigger_id"])


# ---------------------------------------------------------------- main


_INSTANCE_LOCK = None


def single_instance() -> bool:
    """Hold a localhost port for the whole run: a second copy (scheduled task
    firing while a manual run waits for login) exits instead of colliding."""
    import socket

    global _INSTANCE_LOCK
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 5099))
    except OSError:
        sock.close()
        return False
    _INSTANCE_LOCK = sock
    return True


def run(force: bool) -> None:
    if not single_instance():
        log("Another rebalance run is already in progress - exiting")
        return
    now = datetime.now(IST)
    month = now.strftime("%Y-%m")
    live = os.environ.get("ZERODHA_LIVE", "false").strip().lower() == "true"
    mode = "live" if live else "dry"
    state = load_state()

    min_gap = int(env("BOT_MIN_DAYS_BETWEEN", "20"))
    if not force:
        if now.weekday() >= 5:
            log("Weekend - nothing to do")
            return
        holidays = {d.strip() for d in os.environ.get("NSE_HOLIDAYS", "").split(",") if d.strip()}
        if now.date().isoformat() in holidays:
            log("NSE holiday - nothing to do")
            return
        if state.get("done_month") == month and (state.get("done_mode") == "live" or not live):
            log(f"Rebalance for {month} already done ({state.get('done_mode')})")
            return
        # A late-month first run (e.g. the 30th) must not trigger another one on the 1st.
        last_live = state.get("last_live_date")
        if live and last_live and (now.date() - date.fromisoformat(last_live)).days < min_gap:
            log(f"Last live rebalance {last_live} is < {min_gap} days ago - waiting")
            return

    from kiteconnect import KiteConnect

    api_key, api_secret = env("KITE_API_KEY"), env("KITE_API_SECRET")
    cap = Decimal(env("BOT_CAPITAL_CAP", "10000"))
    top = int(env("BOT_TOP_N", "10"))
    stop_pct = Decimal(env("BOT_STOP_LOSS_PCT", "15"))
    port = int(env("KITE_CALLBACK_PORT", "5010"))
    bind = env("KITE_CALLBACK_BIND", "127.0.0.1")
    hh, mm = env("BOT_LOGIN_DEADLINE_IST", "15:00").split(":")
    deadline = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)

    ledger_full: dict[str, dict] = state["ledger"]
    # Zerodha rejects orders from any IP not whitelisted on the Kite app. Check
    # before asking for approval, so a changed IP doesn't waste a login.
    allowed_ips = {ip.strip() for ip in os.environ.get("KITE_ALLOWED_IPS", "").split(",") if ip.strip()}
    if allowed_ips:
        try:
            my_ip = urlopen("https://api.ipify.org", timeout=10).read().decode().strip()
        except Exception as exc:
            my_ip = f"unknown ({exc})"
        if my_ip not in allowed_ips:
            send(
                f"[{mode.upper()}] Momentum rebalance {month} - BLOCKED: public IP changed",
                f"This PC's public IP is {my_ip}, but Kite only allows {', '.join(sorted(allowed_ips))}.\n"
                "Zerodha would reject every order. Add the new IP in the Kite developer console "
                "(your app -> IP whitelist) AND to KITE_ALLOWED_IPS in .env. Will re-check next weekday.",
            )
            log(f"BLOCKED: public IP {my_ip} not in KITE_ALLOWED_IPS {sorted(allowed_ips)}")
            return

    data = asyncio.run(load_market_data(top, list(ledger_full)))
    closes, data_date = data["closes"], data["latest"]
    picks, checks, excluded = run_data_checks(data, top, now.date())
    check_lines = ["Safety checks:", *[c.line() for c in checks]]
    check_lines += [f"  excluded {e}" for e in excluded]
    if not all(c.ok for c in checks):
        send(
            f"[{mode.upper()}] Momentum rebalance {month} - BLOCKED by safety checks",
            "\n".join([*check_lines, "", "Nothing was traded and no login is requested. Will re-check next weekday."]),
        )
        log("BLOCKED by data safety checks: " + "; ".join(c.detail for c in checks if not c.ok))
        return

    prelim_sells = [s for s in ledger_full if s not in picks]
    prelim_buys = [s for s in picks if s not in ledger_full]
    kite = KiteConnect(api_key=api_key)
    ngrok_domain = os.environ.get("NGROK_DOMAIN", "").strip().removeprefix("https://").strip("/")
    tunnel = start_ngrok(ngrok_domain, port) if ngrok_domain else None
    remote = bool(tunnel) or bind != "127.0.0.1"
    send(
        f"[{mode.upper()}] Momentum rebalance {month} - log in to approve",
        "\n".join(
            [
                f"Mode: {mode.upper()}" + ("" if live else "  (ZERODHA_LIVE is not true - NO orders will be placed)"),
                f"Signal: 30-day momentum top-{top}, price data as of {data_date}",
                f"Capital cap: Rs{cap:,.0f}   Stop-loss: {stop_pct}% (GTT)",
                "",
                "SELL (dropped out): " + (", ".join(prelim_sells) or "none"),
                "BUY  (new entries): " + (", ".join(prelim_buys) or "none"),
                "KEEP: " + (", ".join(s for s in picks if s in ledger_full) or "none"),
                "",
                *check_lines,
                "",
                "Quantities are finalised at live prices after you log in, after a second",
                "round of checks (right account, live price vs close, order size vs cap).",
                f"To APPROVE, open this link before {deadline:%H:%M} IST"
                + (" (any device)" if remote else " ON THE PC running the bot")
                + ":",
                kite.login_url(),
                "",
                "Ignore this email to skip - it will ask again next weekday.",
                "Not investment advice. Past momentum performance does not guarantee future returns.",
            ]
        ),
    )
    log(f"Waiting for Kite login until {deadline:%H:%M} IST on {bind}:{port}")

    def accept(token: str) -> bool:
        try:
            kite.generate_session(token, api_secret=api_secret)  # sets access token on the client
            return True
        except Exception as exc:
            log(f"Rejected a login callback ({type(exc).__name__}: {exc}) - still waiting")
            return False

    try:
        approved = wait_for_login(bind, port, deadline, accept)
    finally:
        if tunnel:
            tunnel.terminate()
    if not approved:
        log("No login before deadline - skipped, will retry next weekday")
        return

    log("Kite session established")
    profile = kite.profile()
    account = check_account(os.environ.get("KITE_USER_ID"), profile["user_id"])
    # meta.demat_consent: "consent" = DDPI/POA active. Without it every delivery
    # SELL needs a CDSL TPIN, so the bot's sells and GTT stop-losses get rejected.
    consent = (profile.get("meta") or {}).get("demat_consent", "unknown")
    ddpi = Check(
        "DDPI",
        consent == "consent",
        f"demat_consent={consent}"
        + ("" if consent == "consent" else " - auto SELLS and GTT stop-losses will be REJECTED until DDPI is active"),
    )
    log(ddpi.line())
    if not account.ok:
        send(f"[{mode.upper()}] Momentum rebalance {month} - BLOCKED: wrong account", account.line())
        log(account.line())
        return

    now = datetime.now(IST)
    open_at = now.replace(hour=9, minute=20, second=0, microsecond=0)
    if now < open_at:
        log("Waiting for 09:20 IST (after the opening auction)")
        time.sleep((open_at - now).total_seconds())
    elif now > now.replace(hour=15, minute=15):
        log("Too late in the session - will retry next weekday")
        return

    # Reconcile the ledger against what Zerodha really holds (GTT stop may have fired).
    broker_qty = broker_quantities(kite)
    owned = reconcile_ledger({s: int(v["qty"]) for s, v in ledger_full.items()}, broker_qty)
    stopped_out = [s for s in ledger_full if s not in owned]
    for s in stopped_out:
        ledger_full.pop(s)
    for s, q in owned.items():
        ledger_full[s]["qty"] = q
    save_state(state)

    symbols = sorted(set(picks) | set(owned))
    prices, price_src = live_prices(kite, symbols, closes)
    tsym, ticks = resolve_instruments(kite, set(symbols))

    report = [f"Mode: {mode.upper()}   Prices: {price_src}", "", *check_lines, account.line(), ddpi.line()]
    if stopped_out:
        report.append("Stopped out since last run (GTT fired): " + ", ".join(stopped_out))
    failures: list[str] = []

    # Live price far from the ranked close = split/bad data slipped through: leave that stock alone this run.
    deviant = price_deviations(prices, closes) if price_src == "live LTP" else {}
    for sym, why in deviant.items():
        report.append(f"  [SKIP] {sym}: {why}")
        failures.append(f"{sym}: price deviation, skipped")
    report.append("")
    picks = [p for p in picks if p not in deviant]
    trade_owned = {s: q for s, q in owned.items() if s not in deviant}
    untradable = [p for p in picks if p not in tsym and p not in owned]
    for sym in untradable:
        report.append(f"  [SKIP] {sym}: not listed on Kite (NSE delisted/suspended it) - slot stays cash")
    buy_picks = [p for p in picks if p not in untradable]

    # ---- sells first, to free cash
    dry_proceeds = Decimal(0)
    for o in plan_sells(picks, trade_owned, prices):
        gtt_id = ledger_full.get(o.symbol, {}).get("gtt_id")
        if not live:
            report.append(f"WOULD SELL {o.symbol} x{o.qty} (~Rs{o.ref_price:,.2f})")
            dry_proceeds += o.qty * o.ref_price
            owned.pop(o.symbol, None)
            continue
        if gtt_id:
            try:
                kite.delete_gtt(gtt_id)
            except Exception as exc:
                log(f"delete GTT {gtt_id} for {o.symbol} failed: {exc}")
        try:
            filled, avg, status = place_and_wait(kite, tsym.get(o.symbol, o.symbol), "SELL", o.qty)
        except Exception as exc:
            filled, avg, status = 0, Decimal(0), f"ERROR: {exc}"
        left = o.qty - filled
        if left > 0:
            failures.append(f"SELL {o.symbol}: {status}")
            ledger_full[o.symbol]["qty"] = left
            ledger_full[o.symbol]["gtt_id"] = None
            owned[o.symbol] = left
        else:
            ledger_full.pop(o.symbol, None)
            owned.pop(o.symbol, None)
        save_state(state)
        report.append(f"SOLD {o.symbol} x{filled} @ Rs{avg:,.2f}  [{status}]")

    # ---- buys
    free_cash = Decimal(str(kite.margins("equity")["net"])) + dry_proceeds
    buy_plan = plan_buys(buy_picks, owned, prices, free_cash=free_cash, capital_cap=cap, top=top)
    report.append(f"Free cash Rs{free_cash:,.2f}  budget Rs{buy_plan.budget:,.2f}  slot Rs{buy_plan.per_stock:,.2f}")
    for s in buy_plan.skipped:
        report.append(f"SKIP {s}: one share costs more than the per-stock slot")
    kept_value = sum((Decimal(q) * prices.get(s, Decimal(0)) for s, q in owned.items()), Decimal(0))
    order_check = check_buy_orders(
        buy_plan.orders, kept_value=kept_value, per_stock=buy_plan.per_stock, capital_cap=cap, top=top
    )
    report.append(order_check.line())
    if not order_check.ok:
        failures.append(f"buys blocked: {order_check.detail}")
        buy_plan.orders = []
    for o in buy_plan.orders:
        if not live:
            report.append(f"WOULD BUY {o.symbol} x{o.qty} (~Rs{o.ref_price:,.2f} = Rs{o.qty * o.ref_price:,.0f})")
            continue
        try:
            filled, avg, status = place_and_wait(kite, tsym.get(o.symbol, o.symbol), "BUY", o.qty)
        except Exception as exc:
            filled, avg, status = 0, Decimal(0), f"ERROR: {exc}"
        if filled < o.qty and not is_permanent_error(status):
            failures.append(f"BUY {o.symbol}: {status}")
        if filled > 0:
            gtt_id = None
            try:
                gtt_id = place_stop_gtt(
                    kite,
                    tsym.get(o.symbol, o.symbol),
                    filled,
                    avg,
                    prices[o.symbol],
                    stop_pct,
                    ticks.get(o.symbol, Decimal("0.05")),
                )
            except Exception as exc:
                failures.append(f"GTT stop {o.symbol}: {exc}  <- SET THIS STOP MANUALLY IN KITE")
            ledger_full[o.symbol] = {
                "qty": filled,
                "avg_price": str(avg),
                "gtt_id": gtt_id,
                "bought": now.date().isoformat(),
            }
            save_state(state)
        report.append(
            f"BOUGHT {o.symbol} x{filled} @ Rs{avg:,.2f}  [{status}]  GTT stop {'set' if ledger_full.get(o.symbol, {}).get('gtt_id') else 'MISSING'}"
        )

    if not failures:
        state["done_month"], state["done_mode"] = month, mode
        if live:
            state["last_live_date"] = now.date().isoformat()
    state["history"].append({"at": now.isoformat(), "mode": mode, "picks": picks, "failures": failures})
    save_state(state)

    report += ["", "Bot holdings now: " + (", ".join(f"{s} x{v['qty']}" for s, v in ledger_full.items()) or "none")]
    if failures:
        report += ["", "PROBLEMS (will retry next weekday):", *[f"  {f}" for f in failures]]
    subject = f"[{mode.upper()}] Momentum rebalance {month} - {'DONE' if not failures else 'NEEDS ATTENTION'}"
    send(subject, "\n".join(report))
    log(subject + "\n" + "\n".join(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="run even on weekend / if already done this month")
    args = p.parse_args()
    try:
        run(args.force)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        try:
            send("Momentum rebalance FAILED", f"{type(exc).__name__}: {exc}\nSee scripts/zerodha_rebalance.log")
        finally:
            raise
