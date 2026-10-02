@echo off
REM Daily status of the live Zerodha bot -> WhatsApp + email. Target of the ZerodhaBotStatus task (18:45, after PriceSync_1800).
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\zerodha_bot_status.py" >> "E:\projects\stock_app_backend\scripts\zerodha_bot_status.log" 2>&1
