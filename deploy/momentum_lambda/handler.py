"""Zero-cost cloud version of the Zerodha momentum bot (AWS Lambda, no static IP).

Scheduled (EventBridge Scheduler, Asia/Kolkata):
  {"job": "morning"}  09:20 Mon-Fri  price catch-up; on rebalance day -> "tap to rebalance" message
  {"job": "evening"}  18:45 Mon-Fri  price sync, daily status report, stop-loss alerts

HTTP (Lambda Function URL = the Kite app's redirect URL):
  /kite/callback  Kite login redirect  -> plan page with a one-tap Kite *basket* (orders are placed
                  from the user's own phone, so no static IP is needed); basket redirect -> sync fills.

The planning/safety logic is the same tested code the PC bot used (live_plan.py, live_safety.py).
"""

import html
import json
import os
import smtplib
import urllib.request
from datetime import date, datetime, timedelta
from decimal import Decimal
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import boto3
import db
from kite import BASKET, Kite, KiteError, instruments_nse, login_url

from services.trading_service.momentum.live_plan import plan_buys, plan_sells, reconcile_ledger, stop_loss_prices
from services.trading_service.momentum.live_safety import (
    Check,
    check_account,
    check_buy_orders,
    check_coverage,
    check_freshness,
    filter_picks,
)

IST = ZoneInfo("Asia/Kolkata")
TAG = "momobot"
SUFFIXES = ("-BE", "-BZ", "-SM", "-ST")
_s3 = boto3.client("s3")


def cfg(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def now() -> datetime:
    return datetime.now(IST)


def holidays() -> set[date]:
    return {date.fromisoformat(d.strip()) for d in cfg("NSE_HOLIDAYS").split(",") if d.strip()}


# ------------------------------------------------------------------ state (S3)


def load(key: str, default):
    try:
        return json.loads(_s3.get_object(Bucket=cfg("STATE_BUCKET"), Key=key)["Body"].read())
    except _s3.exceptions.NoSuchKey:
        return default


def save(key: str, obj) -> None:
    _s3.put_object(
        Bucket=cfg("STATE_BUCKET"),
        Key=key,
        Body=json.dumps(obj, indent=2, default=str).encode(),
        ContentType="application/json",
    )


def load_state() -> dict:
    return load("state.json", {"ledger": {}, "done_month": None, "done_mode": None, "history": []})


# ------------------------------------------------------------------ notify


def notify(subject: str, body: str) -> None:
    url, secret = cfg("WHATSAPP_NOTIFY_URL"), cfg("WHATSAPP_NOTIFY_SECRET")
    text = f"*{subject}*\n{body}"
    if url and secret:
        chunks = [text[i : i + 3500] for i in range(0, len(text), 3500)]
        for to in [t.strip() for t in cfg("WHATSAPP_TO", "me").split(",") if t.strip()]:
            for part in chunks:
                for _attempt in range(3):
                    try:
                        req = urllib.request.Request(
                            url,
                            data=json.dumps({"to": to, "message": part}).encode(),
                            headers={"content-type": "application/json", "x-api-key": secret},
                        )
                        urllib.request.urlopen(req, timeout=60).read()
                        break
                    except Exception as exc:  # 429 = another send in progress
                        print(f"whatsapp send failed: {exc}")
    user, pw = cfg("MAIL_USERNAME"), cfg("MAIL_PASSWORD").replace(" ", "")
    if user and pw:
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"] = subject, cfg("MAIL_FROM", user)
            msg["To"] = cfg("BOT_MAIL_TO") or cfg("MAIL_TO") or user
            msg.set_content(body)
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(user, pw)
                smtp.send_message(msg)
        except Exception as exc:
            print(f"email failed: {exc}")
    print(subject + "\n" + body)


# ------------------------------------------------------------------ helpers


def settings() -> dict:
    return {
        "cap": Decimal(cfg("BOT_CAPITAL_CAP", "10000")),
        "top": int(cfg("BOT_TOP_N", "10")),
        "stop": Decimal(cfg("BOT_STOP_LOSS_PCT", "15")),
        "gap": int(cfg("BOT_MIN_DAYS_BETWEEN", "20")),
        "live": cfg("ZERODHA_LIVE", "false").lower() == "true",
    }


def base(tsym: str) -> str:
    for s in SUFFIXES:
        if tsym.endswith(s):
            return tsym[: -len(s)]
    return tsym


def tradable(symbols, listed: dict[str, float]) -> dict[str, tuple[str, Decimal]]:
    out = {}
    for s in symbols:
        for cand in (s, *(s + x for x in SUFFIXES)):
            if cand in listed:
                out[s] = (cand, Decimal(str(listed[cand])))
                break
    return out


def market_open(t: datetime) -> bool:
    if t.weekday() >= 5 or t.date() in holidays():
        return False
    return t.replace(hour=9, minute=15, second=0) <= t <= t.replace(hour=15, minute=25, second=0)


def rebalance_due(state: dict, t: datetime, gap: int) -> tuple[bool, str]:
    if t.weekday() >= 5 or t.date() in holidays():
        return False, "market closed today"
    if state.get("done_month") == t.strftime("%Y-%m") and state.get("done_mode") == "live":
        return False, f"{t:%B} rebalance already done"
    last = state.get("last_live_date")
    if last and (t.date() - date.fromisoformat(last)).days < gap:
        return False, f"last rebalance {last}, next on {next_rebalance(state, gap)}"
    return True, "due"


def next_rebalance(state: dict, gap: int) -> date | None:
    """First weekday that is both in a month not yet rebalanced and `gap` days after the last one."""
    last = state.get("last_live_date")
    if not last:
        return None
    d = date.fromisoformat(last) + timedelta(days=gap)
    done = state.get("done_month")
    if done and state.get("done_mode") == "live":
        y, m = map(int, done.split("-"))
        first_next = date(y + (m == 12), m % 12 + 1, 1)
        d = max(d, first_next)
    while d.weekday() >= 5 or d in holidays():
        d += timedelta(days=1)
    return d


def closes_on_latest(con, symbols: list[str]) -> dict[str, Decimal]:
    series = db.close_series(con, symbols, days=10)
    return {s: Decimal(str(v[-1])) for s, v in series.items() if v}


# ------------------------------------------------------------------ scheduled jobs


def job_morning() -> dict:
    t = now()
    st, s = load_state(), settings()
    con = db.connect()
    try:
        sync_log = db.catch_up(con, t.date() - timedelta(days=1))
        due, why = rebalance_due(st, t, s["gap"])
        if not due:
            return {"sync": sync_log, "rebalance": why}
        candidates = db.ranking_candidates(con, take=s["top"] + 10)
        series = db.close_series(con, candidates + list(st["ledger"]))
        picks, excluded, pick_check = filter_picks(candidates, series, s["top"])
        checks = [
            check_freshness(db.latest_trade_date(con), t.date(), holidays()),
            check_coverage(*db.day_counts(con)),
            pick_check,
        ]
        data_date = db.latest_trade_date(con)
    finally:
        con.close()
    lines = ["Safety checks:", *[c.line() for c in checks], *[f"  excluded {e}" for e in excluded]]
    month = t.strftime("%Y-%m")
    if not all(c.ok for c in checks):
        notify(
            f"Momentum rebalance {month} - BLOCKED",
            "\n".join([*lines, "", "Nothing to approve. Re-checks next weekday."]),
        )
        return {"blocked": [c.detail for c in checks if not c.ok]}
    st["pending"] = {"picks": picks, "data_date": str(data_date), "created": t.isoformat()}
    save("state.json", st)
    held = st["ledger"]
    notify(
        f"Momentum rebalance {month} - tap to review",
        "\n".join(
            [
                f"Signal: 30-day momentum top-{s['top']}, prices as of {data_date}. Cap Rs{s['cap']:,.0f}.",
                "SELL: " + (", ".join(x for x in held if x not in picks) or "none"),
                "BUY:  " + (", ".join(x for x in picks if x not in held) or "none"),
                "KEEP: " + (", ".join(x for x in picks if x in held) or "none"),
                "",
                *lines,
                "",
                "1) Tap to log in to Kite (between 09:15 and 15:25):",
                login_url(cfg("KITE_API_KEY"), intent="rebalance"),
                "2) Check the exact orders, tap 'Place orders in Kite', then 'Place all' in Kite.",
                "Ignore to skip - it asks again next weekday.",
                "Not investment advice.",
            ]
        ),
    )
    return {"pending": picks}


def job_evening() -> dict:
    t = now()
    st, s = load_state(), settings()
    con = db.connect()
    try:
        sync_log = db.catch_up(con, t.date())
        ledger = st["ledger"]
        closes = closes_on_latest(con, list(ledger))
        as_of = db.latest_trade_date(con)
    finally:
        con.close()
    if not ledger:
        return {"sync": sync_log, "status": "no holdings"}
    cost = value = Decimal(0)
    rows, hits, near = [], [], []
    for sym, pos in ledger.items():
        qty, avg = int(pos["qty"]), Decimal(str(pos["avg_price"]))
        close = closes.get(sym, avg)
        cost += qty * avg
        value += qty * close
        stop = avg * (1 - s["stop"] / 100)
        away = (close / stop - 1) * 100
        if close <= stop:
            hits.append(sym)
        elif away < 5:
            near.append(f"{sym} only {away:.1f}% above its stop")
        rows.append(
            (
                (close / avg - 1) * 100,
                f"{sym:11} x{qty:<3} Rs{close:>9,.2f} {(close / avg - 1) * 100:+6.1f}%  stop {away:3.0f}% away",
            )
        )
    rows.sort(reverse=True)
    pnl_pct = (value / cost - 1) * 100 if cost else Decimal(0)
    body = [
        f"Prices as of {as_of} close",
        f"Invested Rs{cost:,.0f} -> Rs{value:,.0f} ({value - cost:+,.0f} / {pnl_pct:+.2f}%)",
        "",
    ]
    body += [r[1] for r in rows]
    if near:
        body += ["", "Close to stop: " + "; ".join(near)]
    if hits:
        gtt = [x for x in hits if ledger[x].get("gtt_id")]
        body += [
            "",
            "STOP HIT at close: " + ", ".join(hits),
            ("Kite GTT should already have sold: " + ", ".join(gtt)) if gtt else "",
            "To sell the rest, tap to log in (market hours):",
            login_url(cfg("KITE_API_KEY"), intent="stop"),
        ]
    nxt = next_rebalance(st, s["gap"])
    if nxt:
        body += ["", f"Last rebalance {st.get('last_live_date')}; next one {nxt:%a %d %b}."]
    body += ["", "Not investment advice."]
    notify(
        f"Zerodha bot: Rs{value:,.0f} ({pnl_pct:+.2f}%)" + (" - STOP ALERT" if hits else ""),
        "\n".join(x for x in body if x is not None),
    )
    return {"sync": sync_log, "value": str(value), "stops": hits}


# ------------------------------------------------------------------ HTTP: login -> plan page


def page(title: str, body_html: str, status: int = 200) -> dict:
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#f7f7f8;color:#111}}
h1{{font-size:20px}} table{{width:100%;border-collapse:collapse;background:#fff}} td,th{{padding:6px;border-bottom:1px solid #eee;text-align:left;font-size:14px}}
.sell{{color:#b00020}} .buy{{color:#05753a}} .note{{background:#fff;padding:10px;border-radius:8px;font-size:14px}}
button{{width:100%;padding:16px;font-size:18px;border:0;border-radius:10px;background:#387ed1;color:#fff;margin-top:16px}}
.warn{{background:#fff4e5;padding:10px;border-radius:8px}}</style></head><body><h1>{html.escape(title)}</h1>{body_html}</body></html>"""
    return {"statusCode": status, "headers": {"content-type": "text/html; charset=utf-8"}, "body": doc}


def orders_table(orders: list[dict]) -> str:
    if not orders:
        return "<p>No orders needed.</p>"
    rows = "".join(
        f"<tr class='{o['transaction_type'].lower()}'><td>{o['transaction_type']}</td><td>{html.escape(o['tradingsymbol'])}</td>"
        f"<td>{o['quantity']}</td><td>~Rs{o['_value']:,.0f}</td></tr>"
        for o in orders
    )
    return f"<table><tr><th>Side</th><th>Stock</th><th>Qty</th><th>Value</th></tr>{rows}</table>"


def basket_form(orders: list[dict], enabled: bool, why: str) -> str:
    if not orders:
        return ""
    if not enabled:
        return f"<p class='warn'>{html.escape(why)}</p>"
    data = [{k: v for k, v in o.items() if not k.startswith("_")} for o in orders]
    return (
        f"<form method='post' action='{BASKET}'><input type='hidden' name='api_key' value='{html.escape(cfg('BASKET_API_KEY') or cfg('KITE_API_KEY'))}'>"
        f"<input type='hidden' name='data' value='{html.escape(json.dumps(data))}'>"
        "<button type='submit'>Place orders in Kite</button></form>"
        "<p class='note'>Kite opens with these orders filled in. Tap <b>Place all</b>. You come back here automatically and the bot records the fills.</p>"
    )


def basket_order(tsym: str, side: str, qty: int, value: Decimal) -> dict:
    return {
        "variety": "regular",
        "exchange": "NSE",
        "tradingsymbol": tsym,
        "transaction_type": side,
        "order_type": "MARKET",
        "product": "CNC",
        "quantity": qty,
        "readonly": True,
        "tag": TAG,
        "_value": value,
    }


def reconcile(kite: Kite, st: dict) -> tuple[dict[str, int], list[str]]:
    qty: dict[str, int] = {}
    for h in kite.holdings():
        b = base(h["tradingsymbol"])
        qty[b] = qty.get(b, 0) + int(h["quantity"]) + int(h.get("t1_quantity") or 0)
    for p in kite.positions().get("day", []):
        if p["exchange"] == "NSE" and p["product"] == "CNC":
            b = base(p["tradingsymbol"])
            qty[b] = qty.get(b, 0) + int(p["quantity"])
    ledger = st["ledger"]
    owned = reconcile_ledger({s: int(v["qty"]) for s, v in ledger.items()}, qty)
    gone = [s for s in ledger if s not in owned]
    for s in gone:
        ledger.pop(s)
    for s, q in owned.items():
        ledger[s]["qty"] = q
    return owned, gone


def plan_page(kite: Kite, st: dict, intent: str, account: Check, ddpi: Check) -> dict:
    s, t = settings(), now()
    picks: list[str] = []
    owned, gone = reconcile(kite, st)
    con = db.connect()
    try:
        if intent == "rebalance":
            picks = (st.get("pending") or {}).get("picks")
            if not picks:
                due, why = rebalance_due(st, t, s["gap"])
                return page("No rebalance pending", f"<p>{html.escape(why)}.</p>")
        closes = closes_on_latest(con, list(set(owned) | set(picks if intent == "rebalance" else [])))
    finally:
        con.close()
    listed = instruments_nse()
    orders, notes = [], [account.line(), ddpi.line()]
    if gone:
        notes.append("No longer held (stop-loss or manual sell): " + ", ".join(gone))
    if intent == "stop":
        for sym, q in owned.items():
            avg = Decimal(str(st["ledger"][sym]["avg_price"]))
            close = closes.get(sym)
            if close is not None and close <= avg * (1 - s["stop"] / 100):
                tsym = tradable([sym], listed).get(sym, (sym, None))[0]
                orders.append(basket_order(tsym, "SELL", q, q * close))
        title = "Stop-loss sells"
    else:
        tmap = tradable(set(picks) | set(owned), listed)
        untradable = [p for p in picks if p not in tmap and p not in owned]
        buy_picks = [p for p in picks if p not in untradable]
        sells = plan_sells(picks, owned, closes)
        proceeds = sum((o.qty * o.ref_price for o in sells), Decimal(0))
        after = {k: v for k, v in owned.items() if k not in {o.symbol for o in sells}}
        # Sale proceeds are credited for same-day buys; 0.9 keeps headroom for charges/price moves.
        free_cash = Decimal(str(kite.equity_margin())) + proceeds * Decimal("0.9")
        bp = plan_buys(buy_picks, after, closes, free_cash=free_cash, capital_cap=s["cap"], top=s["top"])
        kept_value = sum((Decimal(q) * closes.get(k, Decimal(0)) for k, q in after.items()), Decimal(0))
        oc = check_buy_orders(
            bp.orders, kept_value=kept_value, per_stock=bp.per_stock, capital_cap=s["cap"], top=s["top"]
        )
        notes.append(oc.line())
        notes += [f"SKIP {x}: one share costs more than the per-stock slot" for x in bp.skipped]
        notes += [f"SKIP {x}: not listed on Kite" for x in untradable]
        for o in sells:
            orders.append(basket_order(tmap.get(o.symbol, (o.symbol,))[0], "SELL", o.qty, o.qty * o.ref_price))
        if oc.ok:
            for o in bp.orders:
                orders.append(basket_order(tmap[o.symbol][0], "BUY", o.qty, o.qty * o.ref_price))
        notes.append(
            f"Free cash Rs{free_cash:,.0f} (incl. ~90% of sale proceeds), budget Rs{bp.budget:,.0f}, slot Rs{bp.per_stock:,.0f}"
        )
        title = f"Rebalance {t:%B %Y}"
    save("state.json", st)
    save(
        "basket.json",
        {
            "intent": intent,
            "created": t.isoformat(),
            "orders": [{k: v for k, v in o.items() if k != "_value"} for o in orders],
        },
    )
    ok = account.ok and s["live"] and market_open(t)
    why = (
        "Wrong Zerodha account - orders disabled."
        if not account.ok
        else "DRY RUN (ZERODHA_LIVE is not true) - orders disabled."
        if not s["live"]
        else "Market is closed - open this link again between 09:15 and 15:25."
    )
    body = (
        orders_table(orders)
        + basket_form(orders, ok, why)
        + "<h3>Checks</h3><pre class='note'>"
        + html.escape("\n".join(notes))
        + "</pre><p style='font-size:12px'>Sizing uses the last NSE close; Kite fills at market. Not investment advice.</p>"
    )
    return page(title, body)


def sync_page(st: dict) -> dict:
    sess = load("session.json", {})
    if sess.get("date") != now().date().isoformat():
        return page("Session expired", "<p>Log in again from the WhatsApp link to record today's orders.</p>")
    kite = Kite(cfg("KITE_API_KEY"), sess["access_token"])
    s, t = settings(), now()
    done_ids = set(st.setdefault("processed_orders", []))
    listed = instruments_nse()
    ledger, lines, new_buys = st["ledger"], [], []
    for o in kite.orders():
        if o.get("tag") != TAG or o["order_id"] in done_ids:
            continue
        sym, filled = base(o["tradingsymbol"]), int(o.get("filled_quantity") or 0)
        if o["status"] != "COMPLETE":
            if o["status"] in ("REJECTED", "CANCELLED"):
                lines.append(f"{o['transaction_type']} {sym}: {o['status']} {o.get('status_message') or ''}")
                st["processed_orders"].append(o["order_id"])
            continue
        avg = Decimal(str(o["average_price"]))
        st["processed_orders"].append(o["order_id"])
        if o["transaction_type"] == "BUY":
            cur = ledger.get(sym)
            if cur:
                q0, a0 = int(cur["qty"]), Decimal(str(cur["avg_price"]))
                cur["avg_price"] = str((q0 * a0 + filled * avg) / (q0 + filled))
                cur["qty"] = q0 + filled
            else:
                ledger[sym] = {"qty": filled, "avg_price": str(avg), "gtt_id": None, "bought": t.date().isoformat()}
            new_buys.append((sym, o["tradingsymbol"]))
            lines.append(f"BOUGHT {sym} x{filled} @ Rs{avg:,.2f}")
        else:
            cur = ledger.get(sym)
            if cur:
                if cur.get("gtt_id"):
                    try:
                        kite.delete_gtt(cur["gtt_id"])
                    except KiteError as exc:
                        lines.append(f"  (couldn't delete old GTT for {sym}: {exc} - delete it in Kite > GTT)")
                left = int(cur["qty"]) - filled
                if left > 0:
                    cur["qty"] = left
                else:
                    ledger.pop(sym)
            lines.append(f"SOLD {sym} x{filled} @ Rs{avg:,.2f}")
    for sym, tsym in new_buys:
        pos = ledger.get(sym)
        if not pos or pos.get("gtt_id"):
            continue
        trigger, limit = stop_loss_prices(
            Decimal(str(pos["avg_price"])), s["stop"], Decimal(str(listed.get(tsym, 0.05)))
        )
        try:
            pos["gtt_id"] = kite.place_gtt_stop(
                tsym, int(pos["qty"]), float(trigger), float(limit), float(pos["avg_price"])
            )
            lines.append(f"  GTT stop set for {sym} at Rs{trigger}")
        except KiteError as exc:
            lines.append(f"  No GTT for {sym} ({exc}) - the bot checks its stop at every close instead")
    owned, gone = reconcile(kite, st)
    basket = load("basket.json", {})
    if basket.get("intent") == "rebalance":
        picks = (st.get("pending") or {}).get("picks", [])
        planned_buys = {base(o["tradingsymbol"]) for o in basket.get("orders", []) if o["transaction_type"] == "BUY"}
        remaining = [x for x in owned if x not in picks] + [x for x in planned_buys if x not in owned]
        if not remaining:
            st.update(done_month=t.strftime("%Y-%m"), done_mode="live", last_live_date=t.date().isoformat())
            st.pop("pending", None)
            lines.append("Rebalance complete.")
        else:
            lines.append("Still open (asks again next weekday): " + ", ".join(remaining))
    st["processed_orders"] = st["processed_orders"][-300:]
    st.setdefault("history", []).append({"at": t.isoformat(), "mode": "basket", "lines": lines})
    save("state.json", st)
    summary = (
        "\n".join(lines) or "No new bot orders found yet - if you just placed them, wait a minute and reopen this page."
    )
    holdings = ", ".join(f"{k} x{v['qty']}" for k, v in ledger.items()) or "none"
    if lines:
        notify(f"Momentum bot: orders recorded ({t:%d %b})", summary + f"\n\nBot holdings: {holdings}")
    return page(
        "Orders recorded", f"<pre class='note'>{html.escape(summary)}</pre><p>Bot holdings: {html.escape(holdings)}</p>"
    )


def handle_http(event: dict) -> dict:
    path = event.get("rawPath", "/")
    q = event.get("queryStringParameters") or {}
    if path != "/kite/callback":
        return {"statusCode": 404, "body": "not found"}
    st = load_state()
    if q.get("action") == "basket" or (q.get("type") == "basket"):
        return sync_page(st)
    if q.get("status") != "success" or not q.get("request_token"):
        if load("session.json", {}).get("date") == now().date().isoformat():
            return sync_page(st)
        return page("Login not completed", "<p>Open the link from WhatsApp again.</p>")
    kite = Kite(cfg("KITE_API_KEY"))
    try:
        kite.create_session(q["request_token"], cfg("KITE_API_SECRET"))
    except KiteError as exc:
        return page("Login failed", f"<p>{html.escape(str(exc))}</p>", 400)
    save("session.json", {"access_token": kite.access_token, "date": now().date().isoformat()})
    # Kite's basket redirects back "exactly like login" (fresh request_token), so the URL alone
    # can't tell a basket return from a login: unrecorded bot orders today => this is a basket return.
    seen = set(st.get("processed_orders", []))
    if any(o.get("tag") == TAG and o["order_id"] not in seen and o["status"] != "OPEN" for o in kite.orders()):
        return sync_page(st)
    prof = kite.profile()
    account = check_account(cfg("KITE_USER_ID") or None, prof["user_id"])
    consent = (prof.get("meta") or {}).get("demat_consent", "unknown")
    ddpi = Check("DDPI", consent == "consent", f"demat_consent={consent}")
    intent = q.get("intent", "rebalance")
    return plan_page(kite, st, intent if intent in ("rebalance", "stop") else "rebalance", account, ddpi)


def lambda_handler(event, context):
    try:
        if event.get("requestContext", {}).get("http"):
            return handle_http(event)
        job = event.get("job")
        if job == "morning":
            return job_morning()
        if job == "evening":
            return job_evening()
        if job == "selftest":  # manual check: can this Lambda download NSE files + reach the DB?
            con = db.connect()
            try:
                d = db.latest_trade_date(con)
            finally:
                con.close()
            recs = db.fetch_bhavcopy(d)
            return {"latest_db_date": str(d), "bhavcopy_rows": None if recs is None else len(recs)}
        return {"error": f"unknown job {job!r}"}
    except Exception as exc:
        notify("Momentum bot ERROR", f"{type(exc).__name__}: {exc}")
        raise
