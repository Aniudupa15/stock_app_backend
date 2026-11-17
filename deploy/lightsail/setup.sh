#!/usr/bin/env bash
# One-time (idempotent) setup of the F&O bot on the Lightsail Ubuntu server.
# Runs ONLY the F&O bot - the monthly momentum rebalance stays on the Lambda (never schedule it
# here too, or it could rebalance twice). Expects the project in ~/stock_app_backend (push.sh).
set -euo pipefail
APP="$HOME/stock_app_backend"
PY="$APP/.venv/bin/python"

echo "== swap (512 MB servers need it for pip installs)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== packages + timezone (cron below is IST)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv python3-dev build-essential curl tzdata
sudo timedatectl set-timezone Asia/Kolkata

echo "== python venv"
[ -x "$PY" ] || python3 -m venv "$APP/.venv"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$APP/requirements.txt"

echo "== ngrok (phone login: Lambda -> https://NGROK_DOMAIN -> this server)"
if ! command -v ngrok >/dev/null; then
  curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq ngrok
fi
TOKEN=$(grep -E '^NGROK_AUTHTOKEN=' "$APP/.env" | cut -d= -f2- | tr -d '"' || true)
if [ -n "$TOKEN" ]; then ngrok config add-authtoken "$TOKEN" >/dev/null; else
  echo "!! NGROK_AUTHTOKEN not in .env - run once: ngrok config add-authtoken <token>"; fi

echo "== cron: F&O bot 09:00 Mon-Fri (DRY RUN unless FNO_LIVE=true in .env AND --live added below)"
mkdir -p "$APP/scripts/state" "$APP/logs"
CRON=$(cat <<CR
# --- fno bot (managed by deploy/lightsail/setup.sh) ---
0 9 * * 1-5  cd $APP && $PY scripts/fno_live_trade.py --email >> logs/fno_live.log 2>&1
# --- end fno bot ---
CR
)
# A fresh server has no crontab: `crontab -l` exits 1, which under pipefail silently killed
# this script here before - tolerate it explicitly.
EXISTING=$(crontab -l 2>/dev/null || true)
printf '%s\n%s\n' "$(printf '%s\n' "$EXISTING" | sed '/--- fno bot/,/--- end fno bot/d')" "$CRON" | crontab -

echo "== smoke test"
"$PY" -c "import kiteconnect; import sys; sys.path.insert(0, '$APP'); import services.trading_service.fno_live; print('imports ok')"
echo "Public IP (must be the static IP whitelisted in Kite): $(curl -4 -s https://api.ipify.org)"
echo "Done. Crontab:"; crontab -l
