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
