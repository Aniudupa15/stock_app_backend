#!/usr/bin/env bash
# Restart today's F&O bot on the server, e.g. after a failed login. Run from your PC:
#   "C:\Program Files\Git\bin\bash.exe" -c "ssh -i ~/.ssh/fno-bot.pem ubuntu@15.252.158.56 bash stock_app_backend/deploy/lightsail/restart_bot.sh"
# Stops any running bot by PID (never `pkill -f <text>`, which also matches the shell that
# runs it), starts a fresh one in the background and shows the start of its log.
set -uo pipefail
cd "$HOME/stock_app_backend"
for pid in $(pgrep -f "^$PWD/.venv/bin/python scripts/fno_live_trade.py|^.venv/bin/python scripts/fno_live_trade.py"); do
  [ "$pid" != "$$" ] && kill "$pid" && echo "stopped bot (pid $pid)"
done
sleep 2
setsid nohup .venv/bin/python scripts/fno_live_trade.py --email --login-wait-minutes "${1:-90}" \
  >> logs/fno_live.log 2>&1 < /dev/null &
echo "started bot (pid $!) - waiting for its first log lines..."
sleep 25
tail -8 logs/fno_live.log
