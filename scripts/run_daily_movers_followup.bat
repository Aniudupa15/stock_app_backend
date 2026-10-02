@echo off
REM 9:15 AM follow-up - target of the DailyMoversFollowup_0915 scheduled task.
REM Catches up any price data missed by the 18:00 sync, then resends the movers report.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_price_catchup.py" >> "E:\projects\stock_app_backend\scripts\price_sync.log" 2>&1
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\daily_movers_email.py" >> "E:\projects\stock_app_backend\scripts\daily_movers_email.log" 2>&1
