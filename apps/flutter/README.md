# Algo Trading — Flutter client

Cross-platform (Android / iOS / Web / Desktop) client for the AI algo-trading
platform. Talks to two backend services over their shared JWT:

- **data-service** (`DATA_BASE_URL`, default `http://localhost:8000`) — auth + NSE data
- **trading-service** (`TRADING_BASE_URL`, default `http://localhost:8001`) — accounts, strategies, backtest, paper trading, broker

## Status

Foundation / vertical slice. Implemented: auth (login → shared JWT), account
list + create paper account, and an end-to-end **backtest** screen (symbol →
backend indicators + Backtester → metrics). Structured for extension:
`config` · `services/{api_client,auth_service,trading_api}` · `models` ·
`screens`.

> This scaffold has **not** been compiled in CI (no Flutter SDK in the backend
> dev environment). Run `flutter pub get` then `flutter analyze` locally before
> extending.

## Run

```bash
flutter pub get
flutter run --dart-define=DATA_BASE_URL=http://localhost:8000 --dart-define=TRADING_BASE_URL=http://localhost:8001
```

## Next screens to build

- Visual strategy builder (AND/OR/NOT over the 12 indicators)
- Strategy list + paper-run + live trade journal / equity curve (fl_chart)
- Positions & orders, risk dashboard, kill switch
- Broker connect flow (Kite login webview) + live-trading multi-confirmation gate
