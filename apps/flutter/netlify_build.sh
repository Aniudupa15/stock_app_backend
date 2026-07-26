#!/usr/bin/env bash
# Netlify (or any CI without Flutter) build script for the web app.
set -euo pipefail

FLUTTER_VERSION="${FLUTTER_VERSION:-3.24.5}"
FLUTTER_DIR="${HOME}/flutter"

if [ ! -x "${FLUTTER_DIR}/bin/flutter" ]; then
  echo "Installing Flutter ${FLUTTER_VERSION}..."
  git clone --depth 1 -b "${FLUTTER_VERSION}" https://github.com/flutter/flutter.git "${FLUTTER_DIR}"
fi
export PATH="${FLUTTER_DIR}/bin:${PATH}"

flutter --version
flutter config --enable-web

# Regenerate missing platform scaffolding (web/index.html, etc.) without
# touching lib/. Safe to run repeatedly; it only fills in absent files.
flutter create . --platforms=web --project-name algo_trading_app >/dev/null

flutter pub get
flutter build web --release \
  --dart-define=DATA_BASE_URL="${DATA_BASE_URL:-http://localhost:8000}" \
  --dart-define=TRADING_BASE_URL="${TRADING_BASE_URL:-http://localhost:8001}"

echo "Built web app -> build/web"
