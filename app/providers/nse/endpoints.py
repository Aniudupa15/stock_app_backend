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

# Reserved for later phases - not called by any Phase 3 code path.
HISTORICAL_EQUITY_PATH = "/api/historical/cm/equity"
MARKET_STATUS_PATH = "/api/marketStatus"
ALL_INDICES_PATH = "/api/allIndices"
LIVE_ANALYSIS_GAINERS_PATH = "/api/live-analysis-variations"  # ?index=gainers
LIVE_ANALYSIS_LOSERS_PATH = "/api/live-analysis-variations"  # ?index=loosers (NSE's own misspelling)
CORPORATE_ANNOUNCEMENTS_PATH = "/api/corporate-announcements"
