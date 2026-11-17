@echo off
REM Daily F&O PAPER trading - NO real orders (Kite is used for market data only).
REM Target of a scheduled task at 09:00 Mon-Fri. Emails the Kite login link (open it on any device),
REM trades the backtested "pe-trend" put variant until 15:15, then emails the day's summary.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\fno_paper_trade.py" --preset pe-trend --email --login-wait-minutes 120 >> "E:\projects\stock_app_backend\scripts\fno_paper.log" 2>&1
