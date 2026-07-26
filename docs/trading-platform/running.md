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
