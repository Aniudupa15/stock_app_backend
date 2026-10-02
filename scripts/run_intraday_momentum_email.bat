@echo off
REM Daily intraday (1-day lookback) momentum picks report - target of the IntradayMomentum_0800 scheduled task.
REM Syncs any missing price data first (no-op if already current) so the report doesn't repeat stale rankings.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_price_catchup.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\intraday_momentum_email.py" >> "E:\projects\stock_app_backend\scripts\intraday_momentum_email.log" 2>&1
