/// App configuration. Override at build time with
/// `--dart-define=DATA_BASE_URL=... --dart-define=TRADING_BASE_URL=...`.
///
/// Two services: the data-service (auth + NSE market data) and the
/// trading-service (accounts, strategies, backtest, paper trading, broker).
class AppConfig {
  static const String dataBaseUrl = String.fromEnvironment(
    'DATA_BASE_URL',
    defaultValue: 'http://localhost:8000',
  );

  static const String tradingBaseUrl = String.fromEnvironment(
    'TRADING_BASE_URL',
    defaultValue: 'http://localhost:8001',
  );
}
