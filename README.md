# Indian Stock Market Backend

A production-grade, NSE-only backend for an Indian stock market app: real-time-ish quotes, historical prices, a 12-indicator technical engine, rule-based AI trading/investment assistants, watchlists, portfolios (with XIRR), price/technical alerts, news aggregation, and a composed market dashboard - all backed by real free NSE data sources, no paid APIs, no mocked data.

> This backend does **not** use yfinance, scikit-learn, or any ML price-prediction model. An earlier prototype did; it was fully replaced by a clean-architecture rewrite. If you're looking for that old API shape, it no longer exists.

---

## Table of Contents
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Data Sources](#data-sources)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Background Jobs](#background-jobs)
- [Testing](#testing)
- [Known Gaps](#known-gaps--limitations)
- [Deployment](#deployment)

---

## Architecture

Clean/hexagonal architecture - the business logic never depends on a concrete database or external API:

```
API layer (FastAPI routers, app/api/v1/)
    -> Services (app/services/) - business logic, orchestration
        -> Ports (app/domain/ports.py) - abstract interfaces
            -> Repositories (app/repositories/) - SQLAlchemy implementations
            -> Providers (app/providers/) - NSE HTTP client, RSS news client
    -> Domain entities (app/domain/entities.py) - framework-agnostic dataclasses
```

All concrete bindings (which repository implements which port, which provider is injected where) happen in exactly one place: `app/api/deps.py`. Nothing in `services/` or `api/v1/` imports a concrete SQLAlchemy/NSE/Redis class directly - only the port interface. This is what makes the NSE data source, cache backend, and persistence layer swappable without touching business logic.

```
app/
  main.py                    # FastAPI app factory, middleware, exception handlers, lifespan
  core/                      # config, logging, exceptions, JWT/password security
  domain/                    # entities.py (dataclasses), ports.py (abstract interfaces)
  models/                    # SQLAlchemy ORM models
  repositories/               # SQLAlchemy implementations of the ports
  providers/
    nse/                     # resilient NSE HTTP client (retry, circuit breaker, rate limit)
    news/                    # RSS feed client + provider
  services/                  # business logic - the only layer with real "rules"
  indicators/                # pure numpy technical indicator functions (SMA, RSI, MACD, ...)
  analysis/                  # candlestick patterns, support/resistance, trend/gap detection
  finance/                   # XIRR (Newton's method), weighted-average-cost holdings
  cache/                     # InMemoryCache and RedisCache, both implementing CachePort
  api/v1/                    # FastAPI routers
  infrastructure/
    db/                      # async engine/session
    scheduler/                # APScheduler jobs (universe sync, price sync, alerts, news, ...)
alembic/                     # migrations, one file per schema change, applied in order
tests/
  unit/                      # fake-port based, no DB/HTTP - fastest tier
  integration/                # real Postgres via testcontainers - repository-level
  api/                        # full FastAPI app, real DB, faked NSE provider at the DI boundary
```

## Tech Stack

- **Framework**: FastAPI + async SQLAlchemy 2.0 + asyncpg, Postgres
- **Auth**: JWT access tokens (stateless, `PyJWT`) + rotating, revocable refresh tokens (opaque, server-side hashed) - `bcrypt` for password hashing
- **Caching**: pluggable `CachePort` - in-memory by default, Redis-backed (`redis.asyncio`) when `CACHE_BACKEND=redis`, fails open on Redis errors
- **Scheduling**: APScheduler (in-process, no external queue/broker)
- **Indicators/analysis**: pure `numpy`, no `pandas`/`ta-lib`
- **AI assistants**: rule-based, deterministic, templated reasoning citing real computed values - **no LLM**
- **Observability**: `prometheus-fastapi-instrumentator` (`GET /metrics`)
- **Testing**: `pytest` + `pytest-asyncio`, `respx` (HTTP mocking), `testcontainers` (real ephemeral Postgres)
- **Load testing**: Locust (`loadtests/locustfile.py`)
- **Lint**: `ruff`
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) - lint + full test suite on every push/PR

## Data Sources

Everything is free, public NSE data - no paid market-data API, ever:

| Data | Source | Reliability |
|---|---|---|
| Equity universe (symbol/name/ISIN/series) | `nsearchives.nseindia.com` static CSV | Reliable (no session/cookie needed) |
| Daily OHLCV (Bhavcopy) | `nsearchives.nseindia.com` static ZIP archive | Reliable |
| Live quote, market status, indices | `www.nseindia.com/api/*` (cookie-gated) | Best-effort - Akamai bot-protection blocks this intermittently; every caller degrades gracefully (returns partial/empty data with a reason, never a 500) |
| Corporate actions, financial results (XBRL) | `www.nseindia.com/api/*` index + `nsearchives.nseindia.com` XBRL documents | Index is best-effort (cookie-gated); the XBRL documents themselves are on the reliable static archive |
| News | Economic Times + Moneycontrol RSS feeds | Best-effort per-feed; a dead feed is skipped, not fatal |

Fundamentals available from XBRL: Revenue, Profit, Basic/Diluted EPS (and derived PE/growth/TTM-EPS). **Not available from any free source found so far**: Book Value, ROE, ROCE, Debt-to-Equity, sector/industry classification, market capitalization - these fields are always explicitly `null`/omitted rather than faked.

## Getting Started

### Prerequisites
- Python 3.12
- Docker (for Postgres, and optionally Redis)

### Local setup

```bash
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env            # then edit values as needed

docker compose -f docker/docker-compose.yml up -d db
alembic upgrade head

uvicorn app.main:app --reload
```

Server runs at **http://localhost:8000**. Interactive API docs: **http://localhost:8000/docs**.

### Full stack via Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

Brings up Postgres, Redis, and the app together (`CACHE_BACKEND=redis` is set automatically in that compose file).

### Seed real data

```bash
python scripts/run_universe_sync.py                              # NSE equity universe
python scripts/run_price_backfill.py --from 2026-01-01 --to 2026-07-01   # historical OHLCV
python scripts/run_alert_evaluation.py                            # manually trigger alert evaluation
```

## Configuration

All settings are environment variables (see `.env.example` for the full list with defaults), grouped by concern: App, Database, NSE provider (timeouts/retries/rate-limit/circuit-breaker), Cache (backend + TTLs), Auth (JWT secret/algorithm/expiry), Scheduler (per-job times/intervals).

**Must be overridden in any real deployment**: `JWT_SECRET_KEY` (ships with an obviously-fake dev default), `DATABASE_URL`, `CORS_ALLOWED_ORIGINS` (defaults to `*`, fine for local dev, not for production alongside credentialed requests).

## API Reference

Base path for everything below: `/api/v1`. Full interactive schema always available at `/docs` (Swagger) and `/redoc`.

### Health
| Method | Path | Notes |
|---|---|---|
| GET | `/healthz` | Liveness - static OK |
| GET | `/readyz` | Readiness - real `SELECT 1` against Postgres |

### Auth (`/auth`)
| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/auth/register` | - | Email + password (min 8 chars) + display name. Returns an access + refresh token pair. |
| POST | `/auth/login` | - | Returns a new token pair. |
| POST | `/auth/refresh` | - | Rotates the refresh token (old one is revoked on use). |
| POST | `/auth/logout` | - | Revokes a specific refresh token. |
| GET | `/auth/me` | Yes | Current user's profile. |
| PATCH | `/auth/me` | Yes | Update `display_name` and/or `email` (re-checks uniqueness). |

Protected endpoints expect `Authorization: Bearer <access_token>`. Missing header → `403`; present-but-invalid/expired token → `401`.

### Stocks (`/stocks`)
| Method | Path | Notes |
|---|---|---|
| GET | `/stocks/search?q=` | Prefix + fuzzy (`pg_trgm`) search by symbol or name. Logs to search history when called with a valid (optional) `Authorization` header. |
| GET | `/stocks/compare?symbols=A,B,C` | 2-5 symbols; composes detail + indicators + fundamentals per symbol into one response |
| GET | `/stocks/{symbol}` | Static info + live quote (quote is `null` if NSE is unreachable) |
| GET | `/stocks/{symbol}/history?range=` | OHLCV bars; `1D`/`5D`/`1M`/`3M`/`6M`/`1Y`/`3Y`/`5Y`/`MAX` |
| GET | `/stocks/{symbol}/indicators` | SMA/EMA/RSI/MACD/Bollinger/VWAP/ADX/ATR/Supertrend/StochRSI/Pivot Points/Volume Profile |
| GET | `/stocks/{symbol}/corporate-actions` | Dividends/splits/bonuses, DB-backed |
| GET | `/stocks/{symbol}/intraday-signal` | Rule-based BUY/SELL/HOLD, confidence, entry/target/stop, templated reasoning |
| GET | `/stocks/{symbol}/fundamentals` | Revenue/profit/EPS growth, PE, dividend yield (XBRL-derived); Book Value/ROE/ROCE/Debt-Equity always `null` |
| GET | `/stocks/{symbol}/long-term-signal` | Rule-based BUY/HOLD/AVOID with strengths/weaknesses/risks |
| GET | `/stocks/{symbol}/news` | Articles whose `related_symbols` include this stock |

### Search History (`/search-history`, all protected)
| Method | Path | Notes |
|---|---|---|
| GET | `/search-history?limit=&offset=` | Most recent searches first |
| DELETE | `/search-history` | Clear all of the caller's history |

### Market (`/market`)
| Method | Path | Notes |
|---|---|---|
| GET | `/market/gainers?period=&limit=` | `period`: `1D`/`1W`/`1M`/`3M`/`1Y` |
| GET | `/market/losers?period=&limit=` | |
| GET | `/market/most-active?limit=` | Ranked by latest volume |
| GET | `/market/52-week-high?limit=` | |
| GET | `/market/52-week-low?limit=` | |
| GET | `/market/heatmap` | Active stocks bucketed by day-change%, sized by volume (no market-cap data available - see Known Gaps) |

### Screener (`/screener`)
| Method | Path | Notes |
|---|---|---|
| POST | `/screener` | Filter by `rsi_below`/`rsi_above`/`price_min`/`price_max`/`above_sma_50`/`min_volume`, any combination. Reads a daily-refreshed indicator snapshot table, not live indicators, so ~2,400 stocks filter in one cheap query. |

### IPO (`/ipo`)
| Method | Path | Notes |
|---|---|---|
| GET | `/ipo?status=&limit=&offset=` | Upcoming/active/listed IPO filings from NSE. `status` is a free-form filter (`ACTIVE`, `LISTED`, or another NSE-reported value); omit for all. |

### AI Chat (`/chat`, protected)
| Method | Path | Notes |
|---|---|---|
| POST | `/chat` | Rule-based Q&A (no LLM) - keyword-matches `message` against a fixed set of intents (portfolio, stock quote, indicators, watchlist, alerts) and dispatches to the real service that computes the answer. The response's `intent` field says which intent matched, so a client can show "I understood this as: ..." instead of pretending it's a real assistant. Unmatched messages get a templated help response, not an error. |

### Watchlists (`/watchlists`, all protected)
| Method | Path | Notes |
|---|---|---|
| POST | `/watchlists` | Create |
| GET | `/watchlists` | List own watchlists |
| GET | `/watchlists/{id}` | Detail, with live price/change per holding |
| DELETE | `/watchlists/{id}` | |
| POST | `/watchlists/{id}/items` | Add a symbol - `404` if the symbol doesn't exist |
| DELETE | `/watchlists/{id}/items/{symbol}` | |

### Portfolios (`/portfolios`, all protected)
| Method | Path | Notes |
|---|---|---|
| POST | `/portfolios` | Create |
| GET | `/portfolios` | List own portfolios |
| GET | `/portfolios/{id}` | Holdings, weighted-average cost, current value, P&L |
| POST | `/portfolios/{id}/transactions` | Record a BUY/SELL |
| GET | `/portfolios/{id}/performance` | Total invested, current value, P&L, XIRR |

### News (`/news`)
| Method | Path | Notes |
|---|---|---|
| GET | `/news?category=&symbol=&limit=&offset=` | `category`: `MARKET`/`COMPANY`/`ECONOMY`/`REGULATION`/`SECTOR` |

### Alerts (`/alerts`, all protected)
| Method | Path | Notes |
|---|---|---|
| POST | `/alerts` | Create - see alert types below |
| GET | `/alerts?status=` | `status`: `ACTIVE`/`TRIGGERED`/`CANCELLED` |
| DELETE | `/alerts/{id}` | |

**Alert types** (`alert_type` + required `condition` keys): `PRICE_ABOVE`/`PRICE_BELOW` (`price`), `PERCENT_CHANGE_ABOVE`/`PERCENT_CHANGE_BELOW` (`percent`), `RSI_ABOVE`/`RSI_BELOW` (`threshold`), `VOLUME_SPIKE` (`multiplier`, vs 20-day average volume), `NEW_52_WEEK_HIGH`/`NEW_52_WEEK_LOW` (no condition needed). Evaluated on a schedule (see below); triggering an alert creates a notification and flips its status to `TRIGGERED` (one-shot, not repeating).

### Notifications (`/notifications`, all protected)
| Method | Path | Notes |
|---|---|---|
| GET | `/notifications?unread_only=&limit=&offset=` | |
| POST | `/notifications/{id}/read` | |

### Dashboard
| Method | Path | Notes |
|---|---|---|
| GET | `/dashboard` | Composed: market status/indices (best-effort), gainers/losers/most-active/52w extremes, latest news, and explicit `notes` about known limitations (sector data, trending-stocks definition, NSE reachability). Cached (~30s) via `CachePort`. |

### Metrics
Mounted at the app root, **not** under `/api/v1` (Prometheus scrapers expect a fixed, version-independent path):

| Method | Path | Notes |
|---|---|---|
| GET | `/metrics` | Prometheus exposition format - request counts/latencies/in-flight gauge |

## Background Jobs

All run in-process via APScheduler (`app/infrastructure/scheduler/`), IST-scheduled unless noted:

| Job | Schedule | Purpose |
|---|---|---|
| `run_universe_sync` | Daily 08:00 | Refresh the NSE equity list, soft-delist symbols no longer listed |
| `run_daily_price_sync` | Daily 18:00 | Backfill the day's Bhavcopy into `historical_prices` |
| `run_corporate_actions_sync` | Daily 07:30 | Rolling -30/+90 day window of corporate actions |
| `run_financial_results_sync` | Daily 09:00 | Refresh stale/missing quarterly financials, capped per run |
| `run_news_sync` | Every 30 min | Pull all configured RSS feeds |
| `run_alert_evaluation` | Every 15 min | Evaluate every ACTIVE alert, create notifications for matches |
| `run_indicator_snapshot_sync` | Daily 18:30 | Refresh `stock_indicator_snapshots` (the table the Screener reads) after that day's price sync |
| `run_ipo_sync` | Daily 07:45 | Refresh upcoming/active/listed IPO filings from NSE |

Each has a manual-trigger script under `scripts/` for local development/demo.

## Testing

Three tiers, same pattern for every feature:
- **Unit** (`tests/unit/`): fake in-memory implementations of every port, no DB/HTTP - fastest, run on every change.
- **Integration** (`tests/integration/`): real Postgres via `testcontainers`, one ephemeral container per test session - repository-level correctness (upserts, constraints, queries).
- **API** (`tests/api/`): the full FastAPI app, real DB, NSE provider faked at the DI boundary - exercises the real router → service → repository chain end-to-end, including real JWT auth.

```bash
pytest                 # full suite (skips Docker-gated tests if Docker isn't available)
pytest tests/unit       # fast subset, no Docker needed
ruff check .            # lint
ruff format --check .   # format check
```

Load testing (manual, not part of CI):
```bash
locust -f loadtests/locustfile.py --host http://localhost:8000
```

## Known Gaps & Limitations

Documented explicitly rather than silently faked:
- **Sector/industry classification and market capitalization**: not populated for any stock - no free bulk NSE source has been found. Sector Analysis is intentionally not built for this reason (see project notes).
- **Book Value, ROE, ROCE, Debt-to-Equity**: not present in XBRL filings for equity issuers; always returned as explicit `null`.
- **Live quotes / market status / indices**: cookie-gated NSE endpoints are blocked by Akamai bot-protection intermittently, even from residential networks - every dependent feature degrades gracefully rather than failing the request.
- **News symbol matching**: word-boundary regex against the symbol universe; a handful of NSE symbols are also common English words (e.g. RELIANCE, GLOBAL, TOTAL), producing occasional false-positive `related_symbols` tags. Treat it as a best-effort hint, not a reliable tag.
- **Alert types**: SMA/EMA cross and Golden/Death Cross are not implemented (need two consecutive days' indicator values, not just a snapshot threshold); portfolio-wide alerts are schema-ready (`alerts.stock_id` is nullable) but not yet evaluated.
- **Auth**: single implicit role - no RBAC. Every authenticated user has identical access; there's nothing in the app yet that needs differentiated permissions.

## Deployment

`Dockerfile` (repo root) is a multi-stage build (non-root user, stdlib-only healthcheck hitting `/healthz`). `docker/docker-compose.yml` brings up Postgres + Redis + the app together for a full local stack. See `.env.example` for every setting a production deployment needs to override.

Notes for deploying to a platform that doesn't host its own Postgres/Redis (e.g. Render + a managed Postgres like Neon/Supabase):

- **Port**: the image reads `$PORT` at container start (defaulting to `8000`) rather than hardcoding it. Render injects `$PORT` (default `10000`) into the container automatically - verified against Render's own docs - so this needs no manual configuration there.
- **Managed Postgres SSL**: Neon/Supabase dashboards hand you a connection string with `?sslmode=require`. That's the libpq/psycopg convention - this app's driver (`asyncpg`) rejects that exact parameter name (`TypeError: connect() got an unexpected keyword argument 'sslmode'`, verified directly against this app's asyncpg version). Rewrite it to `?ssl=require` in `DATABASE_URL` instead; see `.env.example` for a full example.
- **Cache**: default `CACHE_BACKEND=memory` needs no external service, avoiding a third account just to get a first deployment live. Switch to `redis` (with a managed `REDIS_URL`) once one is provisioned - in-memory cache state isn't shared across replicas, so it's a single-instance-only choice.

### Deploying to Render

1. Create a new **Web Service** at [dashboard.render.com](https://dashboard.render.com/) from this repo, environment **Docker** (Dockerfile at repo root is auto-detected). Render's free instance tier needs no credit card; it spins down after 15 min idle and takes 30-60s to cold-start on the next request.
2. In the service's **Environment** tab, add:
   - `DATABASE_URL` - your Neon/Supabase connection string, rewritten to use `postgresql+asyncpg://` and `?ssl=require` (not `?sslmode=require`; see above).
   - `JWT_SECRET_KEY` - a real random value (e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`), never the dev default.
   - `CACHE_BACKEND=memory`, `SCHEDULER_ENABLED=true`.
3. Apply migrations against the managed Postgres *before or right after* the first deploy: `DATABASE_URL=<your ssl=require URL> alembic upgrade head`, run from a machine with network access to it (Neon/Supabase both allow public connections by default).
4. Render builds and deploys on every push to the connected branch by default.
