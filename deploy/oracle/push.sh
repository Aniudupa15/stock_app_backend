#!/usr/bin/env bash
# Copy the project (code + .env + bot state) from this PC to the Oracle VM, then run setup.sh there.
#   bash deploy/oracle/push.sh <vm-public-ip> <path-to-private-key>
# Re-run any time to deploy code changes. Uses only ssh/scp (encrypted); nothing goes via git.
set -euo pipefail
IP="$1"; KEY="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$IP")

tar -C "$ROOT" -czf /tmp/bot.tgz \
  --exclude=.venv --exclude=myenv --exclude=.git --exclude='__pycache__' --exclude='*.log' \
  --exclude=apps --exclude=loadtests --exclude=tests --exclude=data/ml_models \
  .
scp -i "$KEY" /tmp/bot.tgz "ubuntu@$IP:/tmp/bot.tgz"
rm -f /tmp/bot.tgz
"${SSH[@]}" 'mkdir -p ~/stock_app_backend && tar -xzf /tmp/bot.tgz -C ~/stock_app_backend && rm /tmp/bot.tgz && chmod 600 ~/stock_app_backend/.env'
"${SSH[@]}" 'bash ~/stock_app_backend/deploy/oracle/setup.sh'
