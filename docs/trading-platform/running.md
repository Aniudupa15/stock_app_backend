# Running the platform (dev)

## 1. Infrastructure (Postgres + Redis)

```bash
docker compose -f docker/docker-compose.yml up -d db redis
```

## 2. Migrations (creates public + `trading` schema)

```bash
python -m alembic upgrade head
```

## 3. Data-service (auth + NSE data) — port 8000

```bash
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

## 4. Trading-service (accounts / strategies / backtest / paper / broker) — port 8001

```bash
uvicorn services.trading_service.api.app:create_trading_app --factory --host 0.0.0.0 --port 8001
```

Interactive API docs: `http://localhost:8001/docs`.

## 5. Flutter client

See [apps/flutter/README.md](../../apps/flutter/README.md).

## Trading API surface (all under `/trading`, JWT required)

| Method | Path | Purpose |
|---|---|---|
| POST | `/accounts` | Create a paper (or live) account |
| GET | `/accounts` · `/accounts/{id}` | List / get accounts |
| PUT/GET | `/accounts/{id}/risk` | Configure / read risk limits |
| POST | `/accounts/{id}/kill-switch` | Engage / release the kill switch |
| GET | `/accounts/{id}/positions` · `/orders` · `/trades` | Portfolio reads |
| POST | `/accounts/{id}/paper-run` | Run a saved strategy over history as a persisted paper session |
| POST/GET/PATCH/DELETE | `/strategies` | White-box strategy CRUD |
| POST | `/backtest` | Backtest a strategy (stored or inline) |
| POST | `/broker/connect` · `/broker/complete-login` · GET `/broker/status` | Zerodha connectivity |

## Live trading (not yet enabled)

Live execution requires: `pip install kiteconnect`, a real Kite `api_key`/`api_secret`
connected via `/broker/connect`, a **static egress IP** registered with Zerodha,
and the multi-confirmation live gate. The order-stream (`KiteTicker`) wiring and
daily 06:00-IST-token re-login reminder are the remaining live-path pieces.

## F&O paper trading on live Zerodha prices

`scripts/fno_paper_trade.py` paper-trades NSE futures and options intraday.
Zerodha is used **for market data only** (quotes and the NFO instrument master).
`ReadOnlyKite` wraps the client and blocks every order method, and fills come
from `PaperExecutionVenue`.

```bash
# .env: KITE_API_KEY, KITE_API_SECRET (+ KITE_CALLBACK_PORT / NSE_HOLIDAYS as for the rebalance bot)
python scripts/fno_paper_trade.py --underlying NIFTY --instrument FUT --lots 1            # nearest future
python scripts/fno_paper_trade.py --underlying BANKNIFTY --instrument CE --lots 1         # ATM call, nearest expiry
python scripts/fno_paper_trade.py --underlying NIFTY --instrument PE --capital 200000 --email
```

Run it before or during market hours. It prints the Kite login link (or uses
`KITE_ACCESS_TOKEN`), waits for 09:15, and then:

1. Polls the contract's quote every 5 s.
2. Builds 1-minute bars.
3. Enters long when EMA 9 crosses above EMA 21 with RSI 14 above 50.
4. Manages a stop-loss, target and trailing stop through the OMS.
5. Squares everything off at 15:15.

The journal is written to `scripts/state/fno_paper/`.

How F&O is modelled (`libs/derivatives`, `libs/charges/nse_fno.py`):

- **Lots.** Orders must be whole lots. A strategy's `quantity` means lots for
  F&O, and the risk gate rounds down to whole lots.
- **Futures.** Futures are margined. Only realised P&L and charges move cash,
  and unrealised MTM counts toward free funds.
- **Options.** Options are premium-settled. A buy pays the premium. A sell
  receives the premium and blocks margin.
- **Margin.** Margin is a flat-percentage proxy for SPAN: 12% for index
  contracts and 20% for stock contracts (on the strike for short options).
  This is an approximation, not Zerodha's exact figure.
- **Charges.** Charges follow Zerodha's F&O schedule with the Budget 2026 STT
  rates. These rates are config; re-verify them before relying on P&L to the
  rupee.
- **Expiry.** `settle_expired()` closes positions at their settlement value
  (futures at the underlying, options at intrinsic value) and journals
  `EXPIRY` trades.

### Backtesting the F&O strategy

```bash
python scripts/fno_backtest.py --underlying NIFTY --instrument FUT --from 2025-11-01
python scripts/fno_backtest.py --underlying NIFTY --instrument CE --from 2026-05-01 --trades-csv trades.csv
```

The backtest replays Kite 1-minute history through the same strategy, OMS,
risk gate, charges and paper venue as the live paper script. It needs
historical-data access on your Kite Connect app. Downloads are cached in
`scripts/state/fno_backtest/`, and `--csv` / `--vix-csv` run it without Kite.

- **Futures** are proxied by the index; the futures basis is ignored.
- **Options** have no expired-contract history on Kite, so premiums are
  Black-Scholes modelled from the index and India VIX. Treat option results as
  indicative.
- **Lot size** uses today's lot size for the whole period.

## LIVE NIFTY put trading (`pe-trend`)

`scripts/fno_live_trade.py` runs the backtested `pe-trend` rules. It starts as
a **dry run**: live prices, simulated fills, no orders sent.

Real orders need `--live`, `FNO_LIVE=true` in `.env`, and the pre-flight
checks to pass:
- `KITE_USER_ID` matches the logged-in account
- this PC's IP is in `KITE_ALLOWED_IPS` (the static IP whitelisted on the Kite app)
- no open F&O position or working bot order

Risk rails:
- 1 lot maximum.
- `--daily-loss-limit` (default ₹3,000). A signal whose stop-loss risk exceeds
  the remaining limit is skipped.
- An SL order rests at the exchange from the moment the entry fills. It trails
  30% below the premium's high in ₹2 steps, and is re-placed when Zerodha's
  per-order modification cap is reached.
- Everything is sold at 15:15 (MIS, with Zerodha's ~15:20 auto square-off as a
  backstop).

Schedule `scripts/run_fno_live.bat` for 09:00 Mon-Fri. Add `--live` to it only
after dry-run days look right. The journal is written to `scripts/state/fno_live/`.

## End-of-day option strategies on real NSE prices

```bash
python scripts/fetch_fo_bhavcopy.py                       # once, resumable: NSE F&O bhavcopies 2019->today + Kite NIFTY/VIX dailies
python scripts/eod_options_backtest.py --strategy all --grid
python scripts/eod_options_backtest.py --strategy condor  # one strategy, default settings, full report
```

Four strategies (`libs/options_eod/strategies.py`): a monthly iron condor, a
trend credit spread, a VIX-spike condor and a trend debit spread. They're
backtested on **real** daily closing option prices from NSE's F&O bhavcopy;
strikes are chosen by delta from implied volatility.

How the backtest models a trade:
- **Fills:** at the day's close, ±1 point slippage per leg per side.
- **Charges:** Zerodha F&O charges are applied to every leg.
- **Expiry:** positions still open settle at intrinsic value.
- **Stops and targets** are checked at the close only.

Grid output is sorted by the first 70% of the period (train). The test columns
are the untouched last 30%. The pass criteria are: at least 100 trades, PF of
1.3 or more, positive in most years, and a maximum drawdown under 20% of
capital. Strategy 5 (expiry-day theta) isn't included, because it needs
minute-level option data.
