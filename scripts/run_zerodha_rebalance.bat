@echo off
REM Target of the daily ZerodhaRebalance scheduled task (08:45). No-op once this month's rebalance is done.
REM Syncs missing price data first so the ranking uses the latest close.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_price_catchup.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\zerodha_rebalance.py" >> "E:\projects\stock_app_backend\scripts\zerodha_rebalance.log" 2>&1
