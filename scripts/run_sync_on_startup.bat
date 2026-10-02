@echo off
REM Runs on boot/logon - target of the SyncOnStartup scheduled task.
REM run_price_catchup.py already checks max(trade_date) and only fetches if something's missing.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_price_catchup.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
