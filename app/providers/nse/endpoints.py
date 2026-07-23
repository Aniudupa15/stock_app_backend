"""NSE URL constants.

NSE endpoint paths and response shapes are known to drift over time and are
not officially documented - every path below has been verified against the
live site during this rebuild (Phase 1: equity master CSV, quote-equity;
Phase 2: Bhavcopy archive) EXCEPT where noted otherwise. Re-verify against
the live site if any of these start failing.
"""

# Bulk equity master list (symbol, name, ISIN, series, listing date, face value).
# Different host from the main API - this is a static file archive, not the
# cookie-gated JSON API, so it needs no session bootstrap. Verified working.
EQUITY_LIST_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

# Daily full Cash Market Bhavcopy (UDiFF format ZIP), covers ALL CM instruments
# (equities, SGBs, T-Bills, etc.) - callers must filter to equity rows. Same
# no-cookie static archive host as the equity master list. Verified working;
# format the date as YYYYMMDD.
BHAVCOPY_ZIP_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"

# Cookie-gated JSON API, all relative to settings.NSE_BASE_URL. Verified working.
QUOTE_EQUITY_PATH = "/api/quote-equity"

# Cookie-gated JSON API. Path corrected in Phase 3 after cross-referencing a
# well-maintained community NSE client (BennyThadikaran/NseIndiaApi) - the
# original Phase 2 guess (`/api/corporate-actions`) was never reachable from
# any test environment to verify, and was wrong. Still best-effort/degrades
# gracefully like the rest of the cookie-gated API.
CORPORATE_ACTIONS_PATH = "/api/corporates-corporateActions"

# Cookie-gated JSON API - filing index only (params: index=equities,
# period=Quarterly, symbol, from_date/to_date as DD-MM-YYYY). Each returned
# filing has an `xbrl` link into the reliable static archive - see
# FINANCIAL_RESULT_XBRL_PATTERN below. Path per the same community reference.
FINANCIAL_RESULTS_INDEX_PATH = "/api/corporates-financial-results"

# Cookie-gated JSON API. Response shapes cross-referenced against well-
# maintained community NSE clients (nsepython, BennyThadikaran/NseIndiaApi) -
# same approach that found the real corporate-actions/financial-results paths
# in Phase 3 - but NOT live-verified this session: this sandbox's cookie
# bootstrap was 403/blocked when checked (2026-07-22), consistent with the
# documented intermittent Akamai blocking (see project memory). Both methods
# degrade gracefully like get_quote, so a wrong field name just means an
# empty best-effort result, not a crash - but re-verify field names against
# the live site the first time this is exercised for real.
MARKET_STATUS_PATH = "/api/marketStatus"
ALL_INDICES_PATH = "/api/allIndices"

# Cookie-gated JSON API, confirmed LIVE and working this session (2026-07-22,
# Phase 6) - the generic homepage bootstrap was blocked at the time, but
# warming up the session against this specific page (instead of the
# homepage) succeeded, and both endpoints returned real data. Confirmed
# fields: upcoming/active issues -> companyName, symbol, series, status,
# issuePrice, issueSize, issueStartDate, issueEndDate (DD-Mon-YYYY); past/
# listed issues -> company, symbol, securityType, priceRange, issuePrice,
# ipoStartDate, ipoEndDate, listingDate (DD-Mon-YYYY, or "-" if not yet listed).
UPCOMING_IPO_PATH = "/api/all-upcoming-issues"  # ?category=ipo
PAST_IPO_PATH = "/api/public-past-issues"  # ?index=equities
IPO_WARMUP_URL = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"

# Reserved for later phases - not called by any Phase 4 code path.
HISTORICAL_EQUITY_PATH = "/api/historical/cm/equity"
LIVE_ANALYSIS_GAINERS_PATH = "/api/live-analysis-variations"  # ?index=gainers
LIVE_ANALYSIS_LOSERS_PATH = "/api/live-analysis-variations"  # ?index=loosers (NSE's own misspelling)
CORPORATE_ANNOUNCEMENTS_PATH = "/api/corporate-announcements"
