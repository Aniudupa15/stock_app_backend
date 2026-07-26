# Phase 2 — Architecture (Indian AI Algo-Trading Platform)

> **Status:** Phase 2 of the mandated order. **No implementation.** This document defines structure, boundaries, and interfaces only. Builds on [phase1-research.md](phase1-research.md).
> **Confirmed context:** monorepo (two services); Zerodha-first; backend + Flutter; **operating model = self-hosted, single-user, bring-your-own-key**; **data = free NSE data-service for paper+scanner, Kite ₹500/mo only for live**.
> **Guiding constraint from Phase 1:** the design must make **paper → live a configuration change, not a rewrite**, and keep everything **white-box** (SEBI). The existing `stock_app_backend` is already clean hexagonal architecture (ports/adapters, services, repositories) — we **extend those patterns, not replace them**.

---

## 1. Architectural principles (inherited + new)

Inherited from the existing codebase (keep them):
- **Hexagonal / ports-and-adapters.** Business logic depends on ABCs in `domain/ports.py`; concretes are wired once in a `deps.py`. (Ref: `app/domain/ports.py`, `app/api/deps.py`.)
- **Repository pattern**, async SQLAlchemy 2.0, Alembic migrations.
- **Graceful degradation** — external failures raise typed errors and callers degrade, never 500. (Ref: circuit breaker → `ProviderUnavailableError`.)
- **Deterministic, templated "AI"** — no opaque LLM trade calls (also now a SEBI white-box requirement).

New principles for the trading domain:
- **P1 — One execution interface, two implementations.** Strategy/OMS/risk code talks to an `ExecutionVenue` port. `PaperExecutionVenue` and `BrokerExecutionVenue` are the only things that differ between paper and live. *This is the single most important design decision — it delivers the "minimal changes to go live" mandate.*
- **P2 — The broker is never trusted as authoritative-by-assumption.** Every reconnect triggers reconciliation; our OMS holds order-intent, the broker holds truth, and we converge them explicitly.
- **P3 — Safety is enforced at the gateway, not the UI.** Kill switch, rate limits, and risk checks sit *below* the strategy, in the order path, so nothing can bypass them.
- **P4 — Money-touching code is physically isolated.** The trading-service is a separate deployable with a static egress IP; the data-service never places orders.

---

## 2. Monorepo layout

```
trading-platform/                      # repo root (new)
├─ libs/                               # shared, service-agnostic Python packages
│  ├─ core/                            # config base, logging, exceptions, security primitives
│  │                                   #   (lifted from app/core/*)
│  ├─ market_domain/                   # Stock, OhlcvBar, Quote, indicators (from app/domain, app/indicators)
│  ├─ trading_domain/                  # NEW: Order, Position, Fill, Signal, Strategy, RiskProfile entities + ports
│  ├─ trading_calendar/                # NEW: IST-aware market-state (pre-open/open/square-off/closed),
│  │                                   #   holiday-master-driven (fetched, never hardcoded — Phase 1 §4)
│  ├─ charges/                         # NEW: NSE cost calculator (brokerage/STT/txn/SEBI/stamp/GST/DP),
│  │                                   #   broker-schedule-pluggable — shared by paper engine AND live P&L
│  └─ broker/                          # NEW: BrokerAdapter interface + Zerodha adapter (+ later adapters)
│
├─ services/
│  ├─ data_service/                    # = today's stock_app_backend, moved here ~as-is
│  │                                   #   (NSE provider, universe/price sync, indicators, screener,
│  │                                   #    news, dashboard, existing AI assistants). Free data source.
│  └─ trading_service/                 # NEW: OMS, risk engine, paper engine, strategy engine,
│                                      #   backtester, position manager, broker session mgmt,
│                                      #   the live trading engine process, analytics.
│
├─ apps/
│  └─ flutter/                         # NEW: Flutter app (Android/iOS/Web/Desktop)
│
├─ deploy/                             # docker-compose, per-service Dockerfiles, CI, infra notes
│  └─ (trading_service gets static-egress topology; data_service does not)
└─ docs/trading-platform/             # these phase docs move here
```

**Tooling decision:** Python workspace via **`uv`** (already the project's toolchain — note `.python-version`, `uv`-style layout) with `libs/*` as path dependencies of each service. Two independent FastAPI apps, two Dockerfiles, one compose file. Shared Postgres instance, **separate schemas** (`data`, `trading`) so migrations don't collide. Redis shared (separate key prefixes).

**Migration path (Phase 4 concern, noted here):** move existing `app/` → `services/data_service/`, extract `core/`, `domain/entities.py` (market half), `indicators/` into `libs/`. Mechanical, behavior-preserving; the 247 existing tests are the safety net.

---

## 3. Service boundaries & responsibilities

| Concern | data_service | trading_service |
|---|---|---|
| NSE universe / price / corp-actions / fundamentals sync | ✅ owns | consumes (read) |
| Indicators, screener, market movers, news, dashboard | ✅ owns | consumes |
| Existing rule-based AI assistants (intraday/long-term) | ✅ owns | consumes as *signal source* |
| Live quotes for **paper + scanner** (free NSE) | ✅ serves | consumes |
| Broker session (Kite login/token) | ✗ | ✅ owns |
| Live ticks + order-fill stream for **live** (Kite WS ₹500/mo) | ✗ | ✅ owns |
| Order Management (place/modify/cancel, brackets, reconcile) | ✗ | ✅ owns |
| Positions / holdings / P&L (paper + live) | ✗ | ✅ owns |
| Risk engine, kill switch, safety limits | ✗ | ✅ owns |
| Strategy engine, backtester, performance analytics | ✗ | ✅ owns |
| Static egress IP required | ✗ | ✅ |

**Inter-service contract:** trading_service reads market data from data_service over (a) internal REST for snapshots and (b) the existing WebSocket `Broadcaster` for live paper quotes. No shared DB tables across the boundary — only the shared `libs/`. Keeps the money-service deployable and scalable independently (P4).

---

## 4. The execution abstraction (P1) — core of the whole design

```
libs/trading_domain/ports.py
────────────────────────────
class ExecutionVenuePort(ABC):
    async def place(self, intent: OrderIntent) -> OrderAck          # returns venue order id + status
    async def modify(self, venue_order_id, changes) -> OrderAck
    async def cancel(self, venue_order_id) -> OrderAck
    async def positions(self) -> list[Position]
    async def holdings(self) -> list[Holding]
    async def available_margin(self) -> Margin
    async def required_margin(self, intents: list[OrderIntent]) -> Margin
    def order_events(self) -> AsyncIterator[OrderEvent]             # push stream of fills/rejections
```

Two implementations, selected by the account's `mode` (PAPER | LIVE):

- **`PaperExecutionVenue`** — fills `OrderIntent`s against live NSE quotes (from data_service) using a **slippage model**, applies the **`libs/charges` calculator**, simulates MIS square-off (~15:15–15:20) and circuit locks, emits synthetic `OrderEvent`s. Persists to `trading` schema paper tables. No network egress to a broker.
- **`BrokerExecutionVenue`** — wraps a `BrokerAdapter` (Zerodha first). Translates `OrderIntent` → Kite `/orders/:variety` params, subscribes to Kite's WebSocket order postbacks and re-emits them as `OrderEvent`s, uses `/margins/orders` for `required_margin`.

Everything above this line — strategy engine, OMS bracket logic, risk engine, position manager, analytics — is **written once** and runs identically in paper and live. **That is the "minimal changes to go live" guarantee, structurally enforced.**

### 4.1 BrokerAdapter port (auth-model-agnostic — Phase 1 §3)
```
libs/broker/base.py
───────────────────
class BrokerAdapter(ABC):
    # --- auth as a strategy, because Zerodha=daily-checksum, Angel=TOTP, Upstox/Fyers=OAuth ---
    def begin_login(self) -> LoginChallenge          # e.g. Kite login URL
    async def complete_login(self, callback: dict) -> BrokerSession   # request_token→access_token
    def session_valid(self, s: BrokerSession) -> bool
    def session_expiry(self, s: BrokerSession) -> datetime            # Kite: 6AM IST next day
    # --- trading surface (maps to ExecutionVenuePort needs) ---
    async def place_order(...); modify_order(...); cancel_order(...)
    async def orders(); trades(); positions(); holdings(); margins()
    async def stream(self, tokens) -> AsyncIterator[Tick | OrderUpdate]
```
Zerodha adapter wraps `pykiteconnect` (official) rather than hand-rolling HTTP — the official client already handles the binary WS parsing and checksum.

---

## 5. Order Management System (OMS)

Sits between strategy and `ExecutionVenuePort`. Responsibilities and design:

- **OrderIntent table** (our record of intent) → submitted → venue order id linked → lifecycle tracked from `OrderEvent`s. Statuses mirror Kite's lifecycle (Phase 1 §2.2) normalized to: `PENDING → SUBMITTED → OPEN → PARTIAL → COMPLETE / CANCELLED / REJECTED`.
- **Platform-managed brackets / OCO / trailing-SL** (Zerodha BO is deprecated — Phase 1 §2.2). A bracket = parent entry + child SL + child target as *managed* orders; on one leg filling, the OCO manager cancels the sibling. **Same code path for paper and live** (a big win of P1). Trailing-SL updates the SL child on favorable ticks.
- **Rate-limited order gateway** — token bucket per endpoint class (order ≤ safe fraction of 10/s, respect 25-modifies/order and 2000-MIS/day caps — Phase 1 §2.5). `aiolimiter` (already a dependency). Retries with backoff on transient broker errors.
- **Idempotency & dedupe** — client idempotency key per intent; `tag` field carries a correlation id; never submit two orders for one intent even across a socket drop (P2).
- **Reconciliation loop** — on (re)connect, pull broker `orders/positions/holdings`, diff against OMS, resolve orphans/unknowns before the engine resumes.
- **Kill switch (P3)** — a gateway-level circuit that hard-blocks all `place/modify` (cancels still allowed), independently testable, tripped by the safety engine or user.

---

## 6. Risk engine (pre-trade + continuous)

A **pre-trade gate** every `OrderIntent` passes through, plus a **continuous monitor**:
- Pre-trade: max position size, max open positions, max exposure, per-trade risk %, **margin sufficiency** (via `required_margin`), daily/weekly/monthly loss limits, cooldown-after-N-losses, symbol/segment allow-lists, market-open + not-near-square-off checks.
- Continuous: live drawdown vs limits, exposure drift, broker-disconnect detection → auto-pause, circuit-lock detection on held symbols.
- **Verdict object** (`ALLOW / REJECT(reason) / RESIZE(qty)`) is logged to the audit trail (SEBI order-level audit — Phase 1 §1). Capital preservation > trade frequency, per spec.

---

## 7. Strategy engine (white-box)

- **Rule DSL**: boolean tree of conditions (`AND/OR/NOT`) over indicator values and price/volume, multi-timeframe, reusing the existing `libs/market_domain` indicator engine (12 indicators already built). Serialized to JSON (save/share/import/export).
- **Signal source composition**: strategies can consume the existing rule-based intraday/long-term assistant outputs from data_service as inputs, *plus* their own conditions.
- **Deterministic & inspectable** → satisfies SEBI white-box; every signal carries templated reasoning referencing the actual values (matches the codebase's existing signal philosophy).
- A strategy emits `Signal(symbol, side, entry, sl, targets[], size_hint, reasoning)` → risk engine → OMS. Identical downstream in paper/live/backtest.

---

## 8. Backtesting & performance analytics

- **Backtester** = a **third `ExecutionVenue`-shaped harness** (`BacktestExecutionVenue`) replaying historical bars from `historical_prices` (data_service already stores ~213k bars) through the *same* strategy + risk + charges code. Reuses P1: what you backtest is what you paper-trade is what you trade live.
- **Metrics** (Phase 1 mandate): win rate, profit factor, Sharpe, Sortino, max drawdown, CAGR, avg R:R, consecutive win/loss, avg holding time, equity curve, trade distribution. Pure functions over the trade log (numpy — already a dep).
- **Strategy validation gate** before live: computed from backtest + accumulated paper performance; warns/blocks on insufficient sample or poor stability (Phase 1 SEBI + spec).

---

## 9. Runtime / process model

- **data_service**: unchanged runtime — FastAPI + APScheduler jobs + the WS `Broadcaster`.
- **trading_service**: FastAPI (REST + WS to Flutter) **+ a dedicated async "trading engine"** task group started in the app lifespan (mirrors how `Broadcaster.start()` is launched today):
  - *market-data intake loop* (paper: subscribe to data_service quotes; live: Kite WS ticks),
  - *strategy evaluation loop* (on each bar/tick for active strategies),
  - *order-event loop* (consume `ExecutionVenue.order_events()` → update positions/OMS → push to Flutter),
  - *safety monitor loop* (risk limits, square-off timer, disconnect watch).
- **Scheduling**: APScheduler for daily jobs (broker-relogin reminder at ~08:30 IST, EOD analytics, MIS square-off enforcement). Heavier fan-out (per-symbol scans across the universe) stays in data_service's existing snapshot jobs. **Celery deferred** unless load demands it — an asyncio task group is sufficient for single-user (matches operating model).

---

## 10. Data model (new — `trading` schema)

New tables (all user-scoped, following the existing `user_id`-scoping + JWT pattern):
`broker_sessions` (encrypted api_key/secret, short-lived access_token, expiry) · `trading_accounts` (mode PAPER|LIVE, virtual_balance, broker ref) · `strategies` (JSON rule tree, status, validation metrics) · `order_intents` · `orders` (venue-linked, lifecycle) · `fills` · `positions` · `holdings` · `bracket_groups` (parent/child OCO links) · `risk_profiles` (limits) · `risk_events` · `signals` · `trades` (closed round-trips for analytics) · `equity_snapshots` (curve) · `audit_log` (immutable, SEBI). Reuse existing `users`, `notifications`, `refresh_tokens`.

---

## 11. Flutter app architecture (high level)

- **State**: Riverpod (or Bloc) + repository layer mirroring backend ports; models generated from backend Pydantic schemas.
- **Transport**: REST for CRUD, **WebSocket** for live quotes / positions / order events / P&L.
- **Screens** (map to spec dashboard): Market Overview (indices/heatmap/movers), Watchlist, Scanner, AI Signals, Paper vs Live toggle (guarded), Positions, Orders, Portfolio, Strategy Builder (visual AND/OR/NOT), Backtest, Performance, Risk dashboard, Broker Connect, Settings/Kill-switch.
- **Charts**: candlesticks + indicators (fl_chart / syncfusion — decide in Phase 3).
- **Live-trading UX guardrails**: multi-step consent, broker-connected + risk-configured + validation-passed gates, prominent kill switch, daily broker re-login flow (Phase 1 §2.1). Material 3, light/dark.
- **Platforms**: Android/iOS/Web first; Desktop after.

---

## 12. Security architecture

- **Secrets**: `api_secret` (and other brokers' refresh tokens) envelope-encrypted at rest, per-user key; `access_token` short-lived in DB, never sent to Flutter. Reuse existing `libs/core/security`.
- **Static egress IP** for trading_service order calls (Kite allow-list — Phase 1 §0/§6).
- **AuthN/Z**: existing JWT + refresh (Phase 5 of data backend) reused across both services (shared `JWT_SECRET_KEY`).
- **Audit**: immutable `audit_log`, append-only, every signal→risk→order→broker-response.
- **Kill switch, idempotency, reconciliation** as in §5 (P2/P3).

---

## 13. Phase 2 exit checklist

- [x] Monorepo layout defined (`libs/ services/ apps/`), tooling = `uv` workspace, shared Postgres w/ separate schemas.
- [x] Service boundaries drawn (data_service = free data; trading_service = money, static IP).
- [x] **P1 execution abstraction** specified — the paper/live/backtest unification.
- [x] `BrokerAdapter` + `ExecutionVenue` port signatures drafted (auth-model-agnostic).
- [x] OMS, risk, strategy, backtest, runtime, data model, Flutter, security all sketched.
- [ ] **User review of this architecture** (esp. §2 monorepo layout, §4 execution abstraction, §9 asyncio-engine-vs-Celery) before Phase 3.

**Next phase:** Phase 3 — System Design (detailed component diagrams, full DB DDL, `OrderIntent`/`OrderEvent`/`Signal` schemas, the Zerodha adapter method-by-method mapping, sequence diagrams for place-order/reconnect-reconcile/square-off, and the paper-fill + slippage + charges algorithms). Then Phase 4 implementation begins with the mechanical monorepo restructure + `libs/` extraction.
