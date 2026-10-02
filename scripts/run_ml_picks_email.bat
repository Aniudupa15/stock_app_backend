@echo off
REM Daily ML picks report - target of the MLPicks_0800 scheduled task.
REM Syncs any missing price data first (no-op if already current) so the report doesn't score stale rankings.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_price_catchup.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\ml_picks_email.py" >> "E:\projects\stock_app_backend\scripts\ml_picks_email.log" 2>&1
