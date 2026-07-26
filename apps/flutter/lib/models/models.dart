/// Plain data models mirroring the trading-service API responses.

class Account {
  Account({required this.id, required this.mode, required this.virtualBalance, required this.startingBalance});
  final String id;
  final String mode;
  final double? virtualBalance;
  final double? startingBalance;

  factory Account.fromJson(Map<String, dynamic> j) => Account(
        id: j['id'] as String,
        mode: j['mode'] as String,
        virtualBalance: _d(j['virtual_balance']),
        startingBalance: _d(j['starting_balance']),
      );
}

class Strategy {
  Strategy({required this.id, required this.name, required this.side, required this.product, required this.status, required this.quantity});
  final String id;
  final String name;
  final String side;
  final String product;
  final String status;
  final int quantity;

  factory Strategy.fromJson(Map<String, dynamic> j) => Strategy(
        id: j['id'] as String,
        name: j['name'] as String,
        side: j['side'] as String,
        product: j['product'] as String,
        status: j['status'] as String,
        quantity: j['quantity'] as int,
      );
}

class BacktestResult {
  BacktestResult({required this.symbol, required this.finalEquity, required this.bars, required this.metrics});
  final String symbol;
  final double finalEquity;
  final int bars;
  final Map<String, dynamic> metrics;

  factory BacktestResult.fromJson(Map<String, dynamic> j) => BacktestResult(
        symbol: j['symbol'] as String,
        finalEquity: _d(j['final_equity']) ?? 0,
        bars: j['bars'] as int,
        metrics: (j['metrics'] as Map).cast<String, dynamic>(),
      );
}

class Trade {
  Trade({required this.symbol, required this.qty, required this.entryPrice, required this.exitPrice, required this.pnlNet, required this.exitReason});
  final String symbol;
  final int qty;
  final double entryPrice;
  final double exitPrice;
  final double pnlNet;
  final String? exitReason;

  factory Trade.fromJson(Map<String, dynamic> j) => Trade(
        symbol: j['symbol'] as String,
        qty: j['qty'] as int,
        entryPrice: _d(j['entry_price']) ?? 0,
        exitPrice: _d(j['exit_price']) ?? 0,
        pnlNet: _d(j['pnl_net']) ?? 0,
        exitReason: j['exit_reason'] as String?,
      );
}

class EquityPoint {
  EquityPoint({required this.ts, required this.equity});
  final DateTime ts;
  final double equity;

  factory EquityPoint.fromJson(Map<String, dynamic> j) => EquityPoint(
        ts: DateTime.parse(j['ts'] as String),
        equity: _d(j['equity']) ?? 0,
      );
}

class BrokerStatus {
  BrokerStatus({required this.broker, required this.connected, required this.status});
  final String broker;
  final bool connected;
  final String status;

  factory BrokerStatus.fromJson(Map<String, dynamic> j) => BrokerStatus(
        broker: j['broker'] as String,
        connected: j['connected'] as bool,
        status: j['status'] as String,
      );
}

double? _d(dynamic v) => v == null ? null : (v is num ? v.toDouble() : double.tryParse(v.toString()));
