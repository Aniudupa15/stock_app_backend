@echo off
REM Stop-loss check, run a few times a day - target of the StopLossCheck scheduled task.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\run_stoploss_check.py" >> "E:\projects\stock_app_backend\scripts\stoploss_check.log" 2>&1
