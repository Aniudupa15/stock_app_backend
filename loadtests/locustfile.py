"""Load test for the Indian Stock Market backend.

    locust -f loadtests/locustfile.py --host http://localhost:8000

Run headless for a fixed duration:

    locust -f loadtests/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 -t 30s

Each simulated user registers a real account once (on_start), then runs a
weighted mix of read-heavy tasks (search, stock detail, dashboard, market
movers - the newly-cached hot paths) and write tasks (watchlist, portfolio)
against real endpoints. Assumes the target server's `stocks`/
`historical_prices` are already populated (see scripts/run_universe_sync.py,
scripts/run_price_backfill.py) - these symbols are real NSE listings.
"""

import random
import uuid

from locust import HttpUser, between, task

_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "HINDUNILVR"]


class StockAppUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        email = f"loadtest-{uuid.uuid4()}@example.com"
        resp = self.client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "loadtestpassword123", "display_name": "Load Test User"},
            name="/api/v1/auth/register",
        )
        tokens = resp.json()
        self.client.headers["Authorization"] = f"Bearer {tokens['access_token']}"
        self.watchlist_id: str | None = None
        self.portfolio_id: str | None = None

    @task(5)
    def search_stocks(self) -> None:
        symbol = random.choice(_SYMBOLS)
        self.client.get(f"/api/v1/stocks/search?q={symbol[:3]}", name="/api/v1/stocks/search")

    @task(5)
    def stock_detail(self) -> None:
        symbol = random.choice(_SYMBOLS)
        self.client.get(f"/api/v1/stocks/{symbol}", name="/api/v1/stocks/[symbol]")

    @task(3)
    def dashboard(self) -> None:
        self.client.get("/api/v1/dashboard", name="/api/v1/dashboard")

    @task(3)
    def market_movers(self) -> None:
        self.client.get("/api/v1/market/gainers", name="/api/v1/market/gainers")
        self.client.get("/api/v1/market/losers", name="/api/v1/market/losers")

    @task(2)
    def watchlist_flow(self) -> None:
        if self.watchlist_id is None:
            resp = self.client.post(
                "/api/v1/watchlists", json={"name": "Load Test Watchlist"}, name="/api/v1/watchlists [POST]"
            )
            if resp.status_code == 201:
                self.watchlist_id = resp.json()["id"]
            return

        symbol = random.choice(_SYMBOLS)
        self.client.post(
            f"/api/v1/watchlists/{self.watchlist_id}/items",
            json={"symbol": symbol},
            name="/api/v1/watchlists/[id]/items [POST]",
        )
        self.client.get(f"/api/v1/watchlists/{self.watchlist_id}", name="/api/v1/watchlists/[id] [GET]")

    @task(1)
    def portfolio_flow(self) -> None:
        if self.portfolio_id is None:
            resp = self.client.post(
                "/api/v1/portfolios", json={"name": "Load Test Portfolio"}, name="/api/v1/portfolios [POST]"
            )
            if resp.status_code == 201:
                self.portfolio_id = resp.json()["id"]
            return

        symbol = random.choice(_SYMBOLS)
        self.client.post(
            f"/api/v1/portfolios/{self.portfolio_id}/transactions",
            json={
                "symbol": symbol,
                "transaction_type": "BUY",
                "quantity": "1",
                "price": "100",
                "transaction_date": "2026-01-01",
            },
            name="/api/v1/portfolios/[id]/transactions [POST]",
        )
