@echo off
REM Wrapper for the momentum email report - target of the two Windows scheduled tasks.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\momentum_email_report.py" >> "E:\projects\stock_app_backend\scripts\momentum_email.log" 2>&1
