@echo off
REM Wrapper for the daily movers report - target of the DailyMovers_0900 scheduled task.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\daily_movers_email.py" >> "E:\projects\stock_app_backend\scripts\daily_movers_email.log" 2>&1
