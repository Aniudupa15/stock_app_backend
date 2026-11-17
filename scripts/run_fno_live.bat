@echo off
REM Daily LIVE NIFTY put trading (pe-trend rules). Scheduled task 09:00 Mon-Fri.
REM DRY RUN unless BOTH: --live below AND FNO_LIVE=true in .env (and pre-flight checks pass).
REM Emails the Kite login link (open on any device), then entries/exits/errors and the day summary.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\fno_live_trade.py" --email >> "E:\projects\stock_app_backend\scripts\fno_live.log" 2>&1
