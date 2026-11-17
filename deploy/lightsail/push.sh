#!/usr/bin/env bash
# Copy the project (code + .env + state) from this PC to the F&O bot server, then run setup.sh there.
#   bash deploy/lightsail/push.sh <static-ip> <path-to-private-key>
# Re-run any time to deploy code changes. Uses only ssh/scp (encrypted); nothing goes via git.
set -euo pipefail
IP="$1"; KEY="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$IP")

tar -C "$ROOT" -czf /tmp/fno-bot.tgz \
  --exclude=.venv --exclude=myenv --exclude=.git --exclude='__pycache__' --exclude='*.log' \
  --exclude=apps --exclude=loadtests --exclude=data/ml_models --exclude=research/data \
  --exclude=scripts/state/fno_backtest \
  .
scp -i "$KEY" -o StrictHostKeyChecking=accept-new /tmp/fno-bot.tgz "ubuntu@$IP:/tmp/fno-bot.tgz"
rm -f /tmp/fno-bot.tgz
"${SSH[@]}" 'mkdir -p ~/stock_app_backend && tar -xzf /tmp/fno-bot.tgz -C ~/stock_app_backend && rm /tmp/fno-bot.tgz && chmod 600 ~/stock_app_backend/.env'
"${SSH[@]}" 'bash ~/stock_app_backend/deploy/lightsail/setup.sh'
