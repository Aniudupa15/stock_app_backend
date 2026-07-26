# Deploying the web app + backend

Two pieces: the **backend** (data-service + trading-service, on Render) and the
**Flutter web app** (on Netlify) pointed at the backend URLs. Do the backend
first — the web app is useless until its APIs are reachable over HTTPS.

```
   Browser ──HTTPS──▶ Netlify (Flutter web)
                         │ calls
          ┌──────────────┴───────────────┐
          ▼                              ▼
  data-service (Render)          trading-service (Render)
   auth + NSE data                accounts/strategies/backtest/paper/broker
          └──────────── shared Postgres (Render) ───────────┘
```

## 1. Backend → Render (Blueprint)

1. Push this repo (done). In Render: **New + → Blueprint**, select this repo. It
   reads [`render.yaml`](../../render.yaml) and provisions: Postgres, a
   `data-service`, and a `trading-service` (both from the same Dockerfile).
2. First deploy runs `alembic upgrade head` automatically (creates the public +
   `trading` schemas). Note the two service URLs, e.g.
   `https://data-service-xxxx.onrender.com` and `https://trading-service-xxxx.onrender.com`.
3. **After** you have the Netlify URL (step 2), set `CORS_ALLOWED_ORIGINS` on
   **both** services to it (Render dashboard → each service → Environment), then
   redeploy. Without this the browser blocks the API calls.

`JWT_SECRET_KEY` is generated once and shared by both services (so tokens issued
by data-service validate on trading-service, and both derive the same
credential-encryption key). For production, also set a dedicated
`TRADING_ENCRYPTION_KEY` (a Fernet key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).

## 2. Web app → Netlify

1. Netlify: **Add new site → Import from Git**, select this repo.
2. Netlify reads [`apps/flutter/netlify.toml`](../../apps/flutter/netlify.toml)
   (base `apps/flutter`, build `netlify_build.sh`, publish `build/web`). The
   build script installs Flutter, runs `flutter create . --platforms=web` to
   generate the web scaffolding, and builds.
3. Set **Environment variables** (Site settings → Build & deploy → Environment):
   - `DATA_BASE_URL` = your Render data-service URL
   - `TRADING_BASE_URL` = your Render trading-service URL
4. Deploy. Copy the Netlify site URL and complete step 1.3 (CORS) with it.

> First build clones the Flutter SDK (~a few minutes). Subsequent builds reuse it if the cache persists.

## 3. Create a login

The seeded default user has no password. Register one against the deployed
data-service:

```bash
curl -X POST "$DATA_BASE_URL/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"choose-a-strong-one","display_name":"You"}'
```

Then log in from the web app.

## 4. Market data (for backtests / paper runs)

Backtests need `historical_prices`. NSE endpoints are often blocked from cloud
IPs (Akamai), so the scheduler is **disabled** by default in `render.yaml`.
Options: run the universe/price sync from a machine that can reach NSE and share
the DB, backfill Bhavcopy manually, or enable `SCHEDULER_ENABLED=true` and see if
your host's IP is allowed.

## Notes & safety

- **Free tier:** Render web services sleep after ~15 min idle (cold starts);
  free Postgres is deleted after 30 days. Use paid plans for anything real.
- **This is a money-touching system.** Keep the deployment private while
  testing. Live trading stays gated (needs `kiteconnect`, a static egress IP
  registered with Zerodha, and daily re-login) and is not enabled by this setup.
- **Local preview** without deploying:
  `cd apps/flutter && flutter run -d chrome --dart-define=DATA_BASE_URL=http://localhost:8000 --dart-define=TRADING_BASE_URL=http://localhost:8001`
