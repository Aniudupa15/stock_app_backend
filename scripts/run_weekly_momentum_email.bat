@echo off
REM Daily 7-day momentum picks report - target of the WeeklyMomentumDaily_0930 scheduled task.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\weekly_momentum_email.py" >> "E:\projects\stock_app_backend\scripts\weekly_momentum_email.log" 2>&1
