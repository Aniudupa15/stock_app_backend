"""ML momentum picks report -> email.

Scores today's liquid universe with the trained model (see
services/trading_service/momentum/ml_ranking.py and scripts/train_ml_model.py)
and emails the top picks. Separate from the rule-based reports so its
performance can be tracked independently, starting from a real live date
rather than only backtested history.

Walk-forward backtest before deployment: Information Coefficient +0.049
(vs ~0 for the raw 30-day rule), positive median pick return (vs negative
for the rule), Sharpe 0.94 vs 0.68, max drawdown -21% vs -36%. Still
experimental - only ~17 out-of-sample periods went into that validation.
Treat this as informational until it has a real live track record.

Sent to ML_PICKS_MAIL_TO (defaults to aniudupa15@gmail.com only - unproven,
kept separate from the family-wide reports until it earns more confidence).
Skips sending if the model hasn't been trained yet, or if price data is stale.

If MAIL_USERNAME/MAIL_PASSWORD are absent it prints the report instead of
sending (dry run).
"""

import asyncio
import os
import smtplib
import sys
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scripts.whatsapp_notify import notify as notify_whatsapp  # noqa: E402

from services.trading_service.momentum.ml_ranking import (  # noqa: E402
    ModelNotTrainedError,
    compute_ml_ranking,
    load_model_metadata,
)


def _last_expected_trading_day(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


async def build_report() -> tuple[bool, str]:
    """Returns (should_send, body)."""
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    lines = [f"ML momentum picks report - {date.today().isoformat()}", "=" * 44, ""]
    try:
        async with sf() as s:
            latest_trade_date = (await s.execute(text("select max(trade_date) from historical_prices"))).scalar()
            if latest_trade_date is None or latest_trade_date < _last_expected_trading_day(date.today()):
                return False, ""

            meta = load_model_metadata()
            if meta:
                lines.append(
                    f"Model trained {meta['trained_at'][:10]} on {meta['n_rows']:,} rows "
                    f"({meta['date_range'][0]} to {meta['date_range'][1]})"
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

            try:
                picks = await compute_ml_ranking(s, top=10)
            except ModelNotTrainedError:
                return False, ""

            lines.append("TOP ML MOMENTUM PICKS (model confidence = predicted top-quartile probability)")
            lines.append("")
            cols = ("Rank", "Symbol", "Last Close", "20d Return", "Model Conf.", "Held")
            widths = (4, 14, 12, 12, 12, 4)
            header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
            lines.append(header)
            lines.append("  ".join("-" * w for w in widths))
            for i, pk in enumerate(picks, 1):
                held_tag = "Yes" if pk.symbol in held_symbols else "-"
                row = (
                    f"{i:>4}",
                    pk.symbol.ljust(14),
                    f"Rs{pk.last_close:>9,.2f}".ljust(12),
                    f"{'+' if pk.ret_20d >= 0 else ''}{pk.ret_20d:.1f}%".ljust(12),
                    f"{pk.pred_proba * 100:.1f}%".ljust(12),
                    held_tag.ljust(4),
                )
                lines.append("  ".join(row))
    finally:
        await engine.dispose()
    lines += [
        "",
        "Experimental. Walk-forward backtested (IC +0.049, Sharpe 0.94) before deployment, but only ~17",
        "out-of-sample periods went into that validation - not proof of a permanent edge. Model confidence",
        "is the predicted probability of a top-quartile 21-day forward return, not a guarantee. Paper trading only.",
    ]
    return True, "\n".join(lines)


def send(subject: str, body: str) -> None:
    notify_whatsapp(subject, body)  # independent of email: still sent if SMTP is unset or fails
    user = os.environ.get("MAIL_USERNAME")
    pw = (os.environ.get("MAIL_PASSWORD") or "").replace(" ", "")
    to_raw = os.environ.get("ML_PICKS_MAIL_TO", "aniudupa15@gmail.com")
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
    should_send, body = await build_report()
    if not should_send:
        print(f"{datetime.now().isoformat()} | SKIPPED: no trained model or stale price data")
        return
    send(f"ML momentum picks report - {date.today().isoformat()}", body)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
