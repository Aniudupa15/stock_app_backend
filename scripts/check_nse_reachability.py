"""Standalone NSE reachability check - stdlib only, no project dependencies required.

Run this directly from a normal terminal on your machine (NOT through the
Claude Code tool sandbox, which may have different network egress):

    python scripts/check_nse_reachability.py

It checks three things Phase 1 depends on:
  1. NSE homepage (session/cookie bootstrap)
  2. NSE live quote API (needs the session cookies from step 1)
  3. NSE bulk equity list CSV (used for the stock universe sync)
"""

import http.cookiejar
import urllib.error
import urllib.request

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def check(name: str, url: str, extra_headers: dict | None = None) -> None:
    headers = {**HEADERS, **(extra_headers or {})}
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(req, timeout=15) as resp:
            body = resp.read(300)
            print(f"[PASS] {name}: HTTP {resp.status}")
            print(f"       body preview: {body[:150]!r}")
    except urllib.error.HTTPError as e:
        body = e.read(300)
        print(f"[FAIL] {name}: HTTP {e.code}")
        print(f"       body preview: {body[:150]!r}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"       cookies so far: {[c.name for c in cookie_jar]}")
    print()


if __name__ == "__main__":
    print("=== NSE reachability check ===\n")
    check("Homepage (cookie bootstrap)", "https://www.nseindia.com/")
    check(
        "Live quote API",
        "https://www.nseindia.com/api/quote-equity?symbol=RELIANCE",
        {"Accept": "application/json, text/plain, */*", "Referer": "https://www.nseindia.com/"},
    )
    check(
        "Equity master CSV",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        {"Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/"},
    )
    print("If all three show [PASS] with real content (JSON/CSV, not an error page),")
    print("your network can reach NSE and Phase 1's provider layer will work as built.")
