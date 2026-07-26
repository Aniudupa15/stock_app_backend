import '../config.dart';
import '../models/models.dart';
import 'api_client.dart';

/// Typed wrapper over the trading-service endpoints.
class TradingApi {
  TradingApi() : _client = ApiClient(AppConfig.tradingBaseUrl);
  final ApiClient _client;

  Future<List<Account>> listAccounts() async {
    final data = await _client.get('/trading/accounts') as List;
    return data.map((e) => Account.fromJson((e as Map).cast<String, dynamic>())).toList();
  }

  Future<Account> createAccount({String mode = 'PAPER', double startingBalance = 1000000}) async {
    final data = await _client.post('/trading/accounts', body: {'mode': mode, 'starting_balance': startingBalance.toString()});
    return Account.fromJson((data as Map).cast<String, dynamic>());
  }

  Future<List<Strategy>> listStrategies() async {
    final data = await _client.get('/trading/strategies') as List;
    return data.map((e) => Strategy.fromJson((e as Map).cast<String, dynamic>())).toList();
  }

  Future<Strategy> createStrategy({
    required String name,
    required Map<String, dynamic> ruleTree,
    String side = 'BUY',
    String product = 'MIS',
    int quantity = 1,
    double? stopLossPct,
    double? targetPct,
  }) async {
    final data = await _client.post('/trading/strategies', body: {
      'name': name,
      'rule_tree': ruleTree,
      'side': side,
      'product': product,
      'quantity': quantity,
      if (stopLossPct != null) 'stop_loss_pct': stopLossPct.toString(),
      if (targetPct != null) 'target_pct': targetPct.toString(),
    });
    return Strategy.fromJson((data as Map).cast<String, dynamic>());
  }

  Future<BacktestResult> runBacktest({
    required String symbol,
    String? strategyId,
    Map<String, dynamic>? ruleTree,
    String product = 'CNC',
    int quantity = 10,
    double? stopLossPct,
    double? targetPct,
    double startingCash = 1000000,
  }) async {
    final data = await _client.post('/trading/backtest', body: {
      'symbol': symbol,
      if (strategyId != null) 'strategy_id': strategyId,
      if (ruleTree != null) 'rule_tree': ruleTree,
      'product': product,
      'quantity': quantity,
      if (stopLossPct != null) 'stop_loss_pct': stopLossPct.toString(),
      if (targetPct != null) 'target_pct': targetPct.toString(),
      'starting_cash': startingCash.toString(),
    });
    return BacktestResult.fromJson((data as Map).cast<String, dynamic>());
  }

  Future<Map<String, dynamic>> paperRun({required String accountId, required String strategyId, required String symbol}) async {
    final data = await _client.post('/trading/accounts/$accountId/paper-run', body: {'strategy_id': strategyId, 'symbol': symbol});
    return (data as Map).cast<String, dynamic>();
  }

  Future<List<Trade>> listTrades(String accountId) async {
    final data = await _client.get('/trading/accounts/$accountId/trades') as List;
    return data.map((e) => Trade.fromJson((e as Map).cast<String, dynamic>())).toList();
  }

  Future<List<EquityPoint>> equityCurve(String accountId) async {
    final data = await _client.get('/trading/accounts/$accountId/equity') as List;
    return data.map((e) => EquityPoint.fromJson((e as Map).cast<String, dynamic>())).toList();
  }

  Future<void> setKillSwitch(String accountId, bool on) async {
    await _client.post('/trading/accounts/$accountId/kill-switch', body: {'on': on});
  }

  Future<BrokerStatus> brokerStatus() async {
    final data = await _client.get('/trading/broker/status');
    return BrokerStatus.fromJson((data as Map).cast<String, dynamic>());
  }

  Future<String> brokerConnect({required String apiKey, required String apiSecret}) async {
    final data = await _client.post('/trading/broker/connect', body: {'api_key': apiKey, 'api_secret': apiSecret});
    return (data as Map)['login_url'] as String;
  }
}
