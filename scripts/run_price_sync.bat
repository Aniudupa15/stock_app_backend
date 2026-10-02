@echo off
REM Wrapper for the daily NSE price sync - target of the PriceSync_1800 scheduled task.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_daily_price_sync.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
