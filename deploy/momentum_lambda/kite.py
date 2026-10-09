"""Minimal Kite Connect v3 REST client (stdlib only - the kiteconnect package drags in
Twisted/autobahn for its websocket ticker, far too heavy for a Lambda zip).

Reads need no static IP; only order/GTT placement is IP-restricted, and this bot
places orders through the Kite basket UI on the user's phone instead.
"""

import csv
import hashlib
import io
import json
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.kite.trade"
LOGIN = "https://kite.zerodha.com/connect/login"
BASKET = "https://kite.zerodha.com/connect/basket"


class KiteError(Exception):
    def __init__(self, message: str, error_type: str = ""):
        super().__init__(message)
        self.error_type = error_type


def login_url(api_key: str, **redirect_params: str) -> str:
    q = {"v": "3", "api_key": api_key}
    if redirect_params:
        q["redirect_params"] = urllib.parse.urlencode(redirect_params)
    return f"{LOGIN}?{urllib.parse.urlencode(q)}"


class Kite:
    def __init__(self, api_key: str, access_token: str | None = None):
        self.api_key = api_key
        self.access_token = access_token

    def _call(self, method: str, path: str, data: dict | None = None) -> object:
        headers = {"X-Kite-Version": "3", "User-Agent": "momentum-bot/1.0"}
        if self.access_token:
            headers["Authorization"] = f"token {self.api_key}:{self.access_token}"
        body = None
        if data is not None:
            body = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(API + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                err = json.load(exc)
            except Exception:
                raise KiteError(f"HTTP {exc.code}") from exc
            raise KiteError(err.get("message", f"HTTP {exc.code}"), err.get("error_type", "")) from exc
        if payload.get("status") != "success":
            raise KiteError(payload.get("message", "unknown error"), payload.get("error_type", ""))
        return payload["data"]

    def create_session(self, request_token: str, api_secret: str) -> dict:
        checksum = hashlib.sha256((self.api_key + request_token + api_secret).encode()).hexdigest()
        data = self._call(
            "POST", "/session/token", {"api_key": self.api_key, "request_token": request_token, "checksum": checksum}
        )
        self.access_token = data["access_token"]
        return data

    def profile(self) -> dict:
        return self._call("GET", "/user/profile")

    def holdings(self) -> list[dict]:
        return self._call("GET", "/portfolio/holdings")

    def positions(self) -> dict:
        return self._call("GET", "/portfolio/positions")

    def equity_margin(self) -> float:
        return float(self._call("GET", "/user/margins/equity")["net"])

    def orders(self) -> list[dict]:
        return self._call("GET", "/orders")

    def place_gtt_stop(self, tsym: str, qty: int, trigger: float, limit: float, last_price: float) -> int:
        condition = {"exchange": "NSE", "tradingsymbol": tsym, "trigger_values": [trigger], "last_price": last_price}
        orders = [
            {
                "exchange": "NSE",
                "tradingsymbol": tsym,
                "transaction_type": "SELL",
                "quantity": qty,
                "order_type": "LIMIT",
                "product": "CNC",
                "price": limit,
            }
        ]
        data = self._call(
            "POST",
            "/gtt/triggers",
            {"type": "single", "condition": json.dumps(condition), "orders": json.dumps(orders)},
        )
        return int(data["trigger_id"])

    def delete_gtt(self, trigger_id: int) -> None:
        self._call("DELETE", f"/gtt/triggers/{trigger_id}")


def instruments_nse() -> dict[str, float]:
    """tradingsymbol -> tick size for every NSE equity instrument (public CSV, no auth)."""
    req = urllib.request.Request(f"{API}/instruments/NSE", headers={"X-Kite-Version": "3"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode()
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("segment") == "NSE" and row.get("instrument_type") == "EQ":
            out[row["tradingsymbol"]] = float(row["tick_size"] or 0.05)
    return out
