#!/usr/bin/env bash
# One-time setup of the Zerodha momentum bot on an Oracle Cloud Always Free VM
# (Ubuntu 24.04, ARM Ampere A1 or AMD micro). Idempotent - safe to re-run.
#
# Expects the project already copied to ~/stock_app_backend (deploy/oracle/push.sh does that).
# Runs as the default "ubuntu" user; uses sudo only for packages, timezone and firewall.
set -euo pipefail

APP="$HOME/stock_app_backend"
PY="$APP/.venv/bin/python"
PORT="${KITE_CALLBACK_PORT:-5010}"

echo "== packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv python3-dev build-essential iptables-persistent tzdata

echo "== timezone (cron times below are IST)"
sudo timedatectl set-timezone Asia/Kolkata

echo "== python venv"
python3 -m venv "$APP/.venv"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$APP/requirements.txt"

echo "== firewall: allow the Kite login callback on :$PORT"
# Oracle's Ubuntu images ship iptables rules that REJECT everything but SSH,
# in addition to the VCN security list (which you open in the console).
if ! sudo iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
  sudo iptables -I INPUT 6 -p tcp -m state --state NEW --dport "$PORT" -j ACCEPT
  sudo netfilter-persistent save
fi

echo "== cron"
mkdir -p "$APP/scripts/state" "$APP/logs"
CRON=$(cat <<EOF
# --- momentum bot (managed by deploy/oracle/setup.sh) ---
# Daily NSE Bhavcopy sync after close, and a catch-up before the bot runs.
0 18 * * 1-5  cd $APP && $PY scripts/run_daily_price_sync.py >> logs/price_sync.log 2>&1
30 8 * * 1-5  cd $APP && $PY scripts/run_price_catchup.py >> logs/price_sync.log 2>&1
# Monthly rebalance check (no-op unless due; emails you a login link when it is).
45 8 * * 1-5  cd $APP && $PY scripts/zerodha_rebalance.py >> logs/zerodha_rebalance.log 2>&1
EOF
)
( crontab -l 2>/dev/null | sed '/--- momentum bot/,/zerodha_rebalance.py/d'; echo "$CRON" ) | crontab -

echo "== smoke test"
"$PY" -c "import kiteconnect, sys; sys.path.insert(0, '$APP'); from app.core.config import get_settings; print('db host:', get_settings().DATABASE_URL.split('@')[-1].split('/')[0])"
echo "Public IP: $(curl -4 -s https://api.ipify.org)"
echo "Done. Crontab:"; crontab -l
