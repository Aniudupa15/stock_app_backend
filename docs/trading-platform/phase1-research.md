# Phase 1 — Research (Indian AI Algo-Trading Platform)

> **Status:** Phase 1 of the mandated order (Research → Architecture → System Design → Implementation → Testing → Paper Trading → Performance Validation → Live Trading). **No implementation has begun.**
> **Date compiled:** 2026-07-26. All facts below were verified against current (2025–2026) sources; citations at the end of each section. **Re-verify against official docs before implementation** — broker terms, fees, and rate limits change frequently.
>
> **Engagement scope (confirmed with user):** monorepo with two services (data-service + trading-service) sharing libraries; **Zerodha Kite Connect** as the first broker adapter; **FastAPI backend + Flutter frontend**, both in scope. Existing `stock_app_backend` (NSE-only data/analysis backend) is reused as the data layer.

---

## 0. Executive summary — what the research changes about the design

Three findings are architecture-shaping and must be settled before any code:

1. **SEBI's retail algo framework is LIVE (fully mandatory since 1 April 2026).** Auto-execution of orders through a broker API by anything other than the account holder's own manual action is regulated. Orders must carry an **exchange-issued unique algo ID**, algos above a turnover/frequency threshold must be **registered with the exchange via the broker**, and the **broker is the principal**. This is not a Phase 8 concern — it dictates how "Auto Trade" can legally work at all, and it is why **paper trading is the correct default and the safe long-term center of gravity** for a third-party platform.

2. **Zerodha Kite Connect requires a static IP for order placement (since 1 April 2025).** Any live-order deployment must run from a fixed egress IP registered with Zerodha. Data/WebSocket/read endpoints are not IP-restricted. → The trading-service needs a **static-egress deployment topology** (NAT gateway / reserved IP) distinct from the stateless data-service.

3. **Access tokens are daily-expiring and require an interactive login.** Kite `access_token` dies at **06:00 IST next day** and can only be regenerated through the user's browser login (request_token → checksum exchange). There is **no silent refresh-token** for Kite. → Live trading requires a **daily user-initiated broker re-login**; the platform cannot keep a Zerodha session alive unattended indefinitely. This is a fundamental UX + scheduling constraint.

Everything else (order types, risk engine, backtester, analytics) is conventional engineering. The two hard external constraints are **compliance** and **daily broker auth**.

---

## 1. SEBI regulatory framework for retail algo trading

**Primary instrument:** SEBI circular *"Safer Participation of Retail Investors in Algorithmic Trading"*, dated **4 February 2025**. Phased enforcement; **fully mandatory for all brokers from 1 April 2026** (interim broker-readiness milestones ran through late 2025).

### What it requires

| Requirement | Implication for our platform |
|---|---|
| **Order tagging** — every algo order routed via a broker API carries a **unique identifier issued by the exchange**. | Our order-placement layer must attach the broker/exchange-provided algo tag to each live order. Kite's `tag` field (≤20 chars) is the hook, but the *registered* algo ID comes from the broker↔exchange empanelment, not from us freely. |
| **White-box vs black-box classification.** White-box (logic disclosed/replicable by the user) vs black-box (opaque). **Black-box algos must be registered, and the provider registered as a Research Analyst.** | Keep our strategies **white-box**: deterministic, fully disclosed rules the user configures and can inspect. This avoids the RA-registration burden and matches the existing rule-based-AI philosophy of this codebase (no opaque LLM trade calls). |
| **Broker = principal; algo provider = agent**, only empanelled providers permitted; static-IP API connectivity, secure auth, **order-level audit trails**, vendor empanelment. | If the platform ever offers *hosted* auto-execution to third parties, it must be **empanelled with the broker/exchange**. For a **self-hosted, single-user (the account owner runs it against their own broker key)** deployment, the user is trading their own account via their own API key — the lightest-compliance path. **Decide the operating model early** (see §7). |
| **Per-second order-rate thresholds**: retail algos crossing a threshold (commonly cited ~10 orders/sec) are treated as algos requiring registration. | Risk engine must **throttle** and stay well under thresholds; this aligns with capital-preservation-first design anyway. |
| **Kill switch / audit / two-factor** expectations. | Already in our safety spec (kill switch, audit logs, MFA). |

### Design takeaways
- **Default and primary mode = paper trading.** No SEBI exposure; full realism.
- **Live mode = "bring your own broker key," white-box strategies only, single account, static IP, full audit log, explicit multi-step consent.** Treat hosted multi-tenant auto-execution as a separate, later, compliance-gated product — **not** in the initial build.
- Persist an **immutable audit trail** of every signal → risk-decision → order → broker-response with timestamps (already a stated requirement; now also a regulatory one).

*Sources:* [SEBI circular coverage – FinSec](https://www.finseclaw.com/article/finsec-tracker-on-sebi-issues-guidelines-on-retail-participation-in-algorithmic-trading) · [QuantInsti – Algo Trading India 2026](https://www.quantinsti.com/articles/algorithmic-trading-india/) · [Tradetron – SEBI Algo Rules 2025](https://tradetron.tech/blog/sebi-algo-trading-rules-in-india-2025) · [AlgoBulls – SEBI 2025-26](https://algobulls.com/blog/industry-insights-and-updates/sebi-new-algotrading-regulations-for-retail-investors-2026)

---

## 2. Zerodha Kite Connect (first broker adapter) — verified spec

Official docs: `https://kite.trade/docs/connect/v3/`. Official Python client: `pykiteconnect`.

### 2.1 Authentication (daily, interactive)
1. Redirect user to `https://kite.zerodha.com/connect/login?v=3&api_key=XXX`.
2. On success, Zerodha redirects to our registered redirect URL with `request_token` in the query string.
3. Exchange it: `POST https://api.kite.trade/session/token` with `api_key`, `request_token`, and `checksum = SHA-256(api_key + request_token + api_secret)`.
4. Receive `access_token`. Auth header for all calls: `Authorization: token api_key:access_token`.
5. **Expiry: 06:00 IST the next day** (regulatory). Invalidate via `DELETE /session/token`. **No refresh token** — daily re-login is mandatory.

> **Design:** a `BrokerSession` store per user holding encrypted `api_key`/`api_secret` (at rest) and the short-lived `access_token`; a **daily "connect your broker" prompt** (push/email) before market open; the trading engine refuses live orders when the session is stale.

### 2.2 Orders — endpoints
| Method | Path | Action |
|---|---|---|
| POST | `/orders/:variety` | Place |
| PUT | `/orders/:variety/:order_id` | Modify |
| DELETE | `/orders/:variety/:order_id` | Cancel |
| GET | `/orders` | All orders |
| GET | `/orders/:order_id` | Order history |
| GET | `/trades` | All trades |
| GET | `/orders/:order_id/trades` | Trades for an order |

**Varieties:** `regular`, `amo` (after-market), `co` (cover order), `iceberg`, `auction`.
**Params:** `tradingsymbol`, `exchange` (NSE, BSE, NFO, CDS, BCD, MCX), `transaction_type` (BUY/SELL), `order_type` (MARKET, LIMIT, SL, SL-M), `quantity`, `product` (CNC, NRML, MIS, MTF), `price`, `trigger_price`, `validity` (DAY, IOC, TTL) + `validity_ttl`, `disclosed_quantity`, `tag` (≤20 chars), `market_protection`, `autoslice`.
**Lifecycle statuses:** `PUT ORDER REQ RECEIVED` → `VALIDATION PENDING` → `OPEN PENDING` → `OPEN`/`TRIGGER PENDING` → terminal `COMPLETE` / `CANCELLED` / `REJECTED`. Modify path adds `MODIFY VALIDATION PENDING`/`MODIFY PENDING`.
**Response fields:** `order_id`, `parent_order_id` (multi-leg/CO), `status`, `filled_quantity`, `pending_quantity`, `cancelled_quantity`, `average_price`, `exchange_order_id`.

> **Note:** Zerodha **deprecated native Bracket Orders (BO)** for equity/F&O some time ago. Our OMS must implement **bracket/OCO logic itself** (entry + SL + target as managed child orders), not rely on a broker BO variety. Cover Orders (`co`) still exist. This is exactly why we need our own position-manager rather than leaning on broker-side brackets — and it makes the abstraction portable across brokers.

### 2.3 Margins
- `POST /margins/orders` — required margin (SPAN, exposure, option premium, additional, BO, cash, VAR, PNL, leverage, charges, total) for a list of orders.
- `POST /margins/basket?consider_positions=false` — basket margin with hedge-benefit netting (initial vs final margin).
- `GET /user/margins` / `GET /user/margins/:segment` — available funds/cash.

> Use these **pre-trade** in the risk engine to reject orders that would exceed available margin, and to size positions.

### 2.4 WebSocket streaming (live ticks + order updates)
- Endpoint: `wss://ws.kite.trade` with `api_key` + `access_token` query params.
- Subscribe: `{"a":"subscribe","v":[tokens]}`; unsubscribe; mode: `{"a":"mode","v":["full",[tokens]]}`.
- **Modes:** `ltp` (8 bytes), `quote` (44 bytes, no depth), `full` (184 bytes, incl. market depth + OI + timestamp). Binary; prices in paise (÷100; currencies ÷10^7).
- **Limits:** up to **3,000 instruments per connection**, **3 concurrent connections per API key** (→ ~9,000 instruments max — relevant for the scanner's universe coverage).
- **Order/trade updates** are pushed as **text JSON on the same socket**: `{"type":"order","data":{…}}` (also `error`, `message`). → We get near-real-time order state without polling.

### 2.5 Rate limits (hard constraints for the OMS)
- **GET (quote/read): <10 req/s** combined across all GET calls.
- **Order place/modify/cancel: 10 req/s** (community-cited; treat 10/s as the ceiling and stay well below).
- **Historical data: 3 req/s.**
- **Order modifications: max 25 per order.**
- **RMS account caps: 2,000 MIS orders/day and 2,000 CO/day** across segments.

> **Design:** a centralized **rate-limited, queued order gateway** (token-bucket per endpoint class) + retry/backoff. `aiolimiter` is already a dependency in this repo and fits.

### 2.6 Pricing (2025–2026)
- **Order/account APIs (place/modify/cancel, holdings, positions, margins): FREE** (since March 2025).
- **Market data (real-time + historical): ₹500/month per API key** (down from ₹2000+₹2000). Bundled together since Feb 2025.
- **Personal (free) plan:** trading APIs free but **no market data** (live or historical) — you'd source data elsewhere.

> **Implication:** we can source **live/historical market data from the existing NSE data-service (free)** and use Kite purely for **execution** — potentially avoiding the ₹500/mo data fee. But Kite's WebSocket gives true broker-grade ticks + order-fill events that NSE's public endpoints do not; for live trading the **₹500/mo Connect data subscription is recommended** for tick fidelity and the order-update stream. Paper trading can run entirely on the free NSE data-service.

*Sources:* [Kite Connect v3 docs](https://kite.trade/docs/connect/v3/) · [pykiteconnect](https://github.com/zerodha/pykiteconnect) · [Rate limits – Kite forum](https://kite.trade/forum/discussion/8577/api-rate-limits) · [Fee revision ₹2000→₹500](https://kite.trade/forum/discussion/15015/revising-kite-connect-fees-from-2000-to-500-per-month) · [Free personal APIs – Z-Connect](https://zerodha.com/z-connect/updates/free-personal-apis-from-kite-connect) · [Static IP requirement / FAQs](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/kite-connect-api-faqs)

---

## 3. Broker comparison (adapter roadmap after Zerodha)

The adapter interface must be designed against the *union* of these so brokers 2–5 drop in cleanly. Verify each against official docs at implementation time.

| Broker | Auth model | Order API cost | Market-data cost | Historical | WebSocket | Notes |
|---|---|---|---|---|---|---|
| **Zerodha Kite Connect** | Daily interactive login → request_token → access_token (checksum). Expires 6 AM. **Static IP for orders.** | **Free** | **₹500/mo** (bundled RT+historical) | Included w/ data plan (3 req/s) | `wss://ws.kite.trade`, 3k/conn ×3 | Largest broker; most mature; BO deprecated. **First adapter.** |
| **Angel One SmartAPI** | **TOTP-based** login (publishable key + TOTP secret) → JWT (access + refresh). | **Free** | **Free** | **Free, deep history** | Yes | No monthly fee; strong for algo. Good **second adapter** (free end-to-end for testing). |
| **Upstox** | **OAuth 2.0** (authorization code) → access token. | ~**₹10 per executed API order** (promo through 31 Mar 2026) | Free tier available | Yes | Yes | Cleanest OAuth; per-order fee. |
| **Dhan** | Access-token / consent model | **₹0 order placement** | **₹499/mo** data (RT+historical) | Paid w/ data | Yes | Modern, API-first. |
| **Fyers** | **OAuth 2.0** → access token | **Free** | **Free** | **Free** (minute-level, ~1–2 yrs back) | Yes | Analytics-friendly; free stack. |

**Recommended adapter order:** Zerodha (1st, per user) → **Angel One** (2nd — fully free, TOTP is automatable for CI/paper realism) → Fyers → Upstox → Dhan.

**Auth-model divergence is the key abstraction challenge:** Zerodha = daily manual checksum login; Angel One = TOTP (semi-automatable); Upstox/Fyers = OAuth redirect; Dhan = consent token. The `BrokerAdapter` interface must express auth as a **capability/strategy** (`begin_login()`, `complete_login(callback_params)`, `is_session_valid()`, `session_expiry()`), not assume one flow.

*Sources:* [Chittorgarh – Angel One API](https://www.chittorgarh.com/broker/angel-broking/api-for-algo-trading-review/14/) · [Stratzy – API cost comparison](https://stratzy.in/blog/cost-comparison-algo-trading-apis-india/) · [Fintegration – Top 5 broker APIs](https://www.fintegrationfs.com/post/top-5-apis-for-building-a-stock-trading-app-in-india-zerodha-angel-dhan-shoonya-fyers) · [SmartAPI](https://smartapi.angelbroking.com/)

---

## 4. Market mechanics (NSE equity) — engine constants

| Item | Value | Engine use |
|---|---|---|
| Pre-open session | **09:00–09:15 IST** (order collection 9:00–9:08, matching 9:08–9:12) | No continuous fills; scanner can warm up. |
| Continuous trading | **09:15–15:30 IST**, Mon–Fri | The only window for MARKET/LIMIT fills. |
| **Intraday (MIS) auto square-off** | Brokers force-close from **~15:20 IST** (equity ~15:20; varies 15:10–15:20 by broker/segment). No new MIS after square-off starts. | Risk engine must **exit all intraday positions by ~15:15** to keep control of the exit price — do not rely on the broker's forced square-off. |
| Post-close | 15:40–16:00 (AMO window varies) | AMO orders only. |
| **Circuit limits** | Per-stock daily bands **2% / 5% / 10% / 20%** by classification; index circuit breakers 10/15/20%. | If a stock is circuit-locked, **MIS square-off can fail → forced delivery + auction penalty.** Engine must detect circuit lock and flag/avoid; never assume an exit will fill. |
| Trading holidays 2026 | **~15 full-day equity holidays** + Muhurat session (Diwali, Sun **8 Nov 2026**). | **Do not hardcode.** Fetch the NSE holiday master at startup/daily (the data-service already talks to NSE); gate the trading calendar on it. |
| Settlement | **T+1** for equity cash. | Affects buying power / delivery accounting in paper P&L. |

> **Trading-calendar service** is a shared library concern (both services need "is the market open now?"). Build it once, holiday-aware, IST-timezone-correct (`pytz` already a dependency), with pre-open/continuous/square-off/closed states.

*Sources:* [NSE Market Timings](https://www.nseindia.com/static/market-data/market-timings) · [5paisa – Auto square-off](https://www.5paisa.com/stock-market-guide/online-trading/intraday-auto-square-off-time) · [Zerodha – MIS/CO square-off bulletin](https://zerodha.com/marketintel/bulletin/249809/latest-intraday-leverages-mis-bo-co) · [NSE Holidays 2026 – ClearTax](https://cleartax.in/s/nse-holidays-2026)

---

## 5. Paper-trading engine — realism requirements (mandated default mode)

Paper mode must be **execution-indistinguishable** from live except no broker call. Research-driven realism checklist:

- **Fills** against live/last NSE tick with **slippage model** (configurable bps; wider for illiquid/low-volume symbols; market vs limit handling; partial fills for large size vs available depth).
- **Costs replicated exactly** — this is where most paper engines cheat. For NSE equity intraday/delivery, model: **brokerage** (Zerodha: ₹0 delivery, 0.03% or ₹20 whichever lower for intraday), **STT/CTT**, **exchange transaction charges**, **SEBI turnover fee**, **stamp duty**, **GST (18% on brokerage+txn)**, **DP charges** on delivery sells. Encode as a pluggable **charges calculator** matching the selected broker's schedule so paper P&L ≈ live P&L.
- **Order types** parity: MARKET/LIMIT/SL/SL-M + platform-managed **bracket/OCO/trailing-SL** (since Zerodha BO is gone, paper and live share the *same* managed-bracket code — big consistency win).
- **MIS auto-square-off** simulated at the same ~15:15–15:20 cutoff.
- **Circuit-lock simulation** (reject fills when the simulated LTP is at a band).
- **T+1 settlement** accounting for delivery buying power.
- **Virtual portfolio/positions/balance** with realized + unrealized P&L, brokerage-inclusive.

The realism target: a strategy's paper equity curve should be a faithful, slightly-pessimistic proxy of live — never rosier.

---

## 6. Security & operational research findings

- **Token vaulting:** `api_secret` and (for other brokers) refresh tokens must be **encrypted at rest** (envelope encryption; per-user key). `access_token` is short-lived but still secret. Never send `api_secret` or `access_token` to the Flutter client.
- **Static-egress** deployment for the trading-service (Zerodha order IP allowlist). The data-service does not need it.
- **Idempotency / duplicate-order prevention:** client-generated idempotency key per intended order; reconcile via `GET /orders` on reconnect; the `tag` field + our own order-intent table prevent double-submits after a socket drop.
- **Reconciliation loop:** on every (re)connect, pull broker order/position/holding state and reconcile against our OMS before acting — never assume our view is authoritative after a disconnect.
- **Kill switch** must sever at the **gateway** layer (block all outbound order calls) and be independently testable, not just a UI flag.
- **Audit log** immutable + regulatory (SEBI order-level audit trail).

---

## 7. Open decisions to resolve in Phase 2 (Architecture)

These are genuine forks the research surfaced; flagging rather than silently choosing:

1. **Operating model (compliance-critical):** (a) **Self-hosted, single-user, bring-your-own-key** (lightest compliance — user trades own account) vs (b) **Hosted multi-tenant with auto-execution** (requires broker/exchange empanelment as an algo provider). Recommendation: **build (a) first**; treat (b) as a separate compliance-gated track. *This gates the entire live-trading design.*
2. **Live market data source:** Kite ₹500/mo data plan (broker-grade ticks + order stream) vs reuse free NSE data-service (no per-tick order-book fidelity). Recommendation: **NSE data-service for paper + scanner; Kite data plan for live execution fidelity.**
3. **Monorepo tooling:** how to physically restructure (`uv`/workspaces, shared `libs/` package for domain + trading-calendar + charges-calculator; `services/data-service`, `services/trading-service`; `apps/flutter`). To be designed in Phase 2.
4. **Async task/runtime:** current repo uses APScheduler; the trading engine's tick-loop + order-lifecycle may warrant Celery/Redis or a dedicated asyncio engine process. Decide in Phase 2.

---

## 8. Phase 1 exit checklist

- [x] SEBI retail algo framework understood and its constraints mapped to the design (compliance is live).
- [x] Zerodha Kite Connect verified end-to-end (auth, orders, margins, streaming, limits, pricing).
- [x] Four remaining brokers compared; adapter roadmap + auth-abstraction requirement identified.
- [x] Market mechanics (timings, square-off, circuits, holidays, settlement) captured as engine constants.
- [x] Paper-trading realism requirements defined (costs, slippage, managed brackets).
- [x] Security/ops constraints (static IP, token vaulting, idempotency, reconciliation, kill switch) captured.
- [ ] **User to confirm operating model (§7.1) and data-source choice (§7.2)** → unblocks Phase 2.

**Next phase:** Phase 2 — Architecture (monorepo layout, service boundaries, `BrokerAdapter` interface, OMS/risk/paper/strategy/backtest module design, data model). Begins only after §7.1–§7.2 are confirmed.
