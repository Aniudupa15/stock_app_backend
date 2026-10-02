@echo off
REM Wrapper for the momentum email report - target of the two Windows scheduled tasks.
REM Syncs any missing price data first (no-op if already current) so the report doesn't repeat stale rankings.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_price_catchup.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\momentum_email_report.py" >> "E:\projects\stock_app_backend\scripts\momentum_email.log" 2>&1
