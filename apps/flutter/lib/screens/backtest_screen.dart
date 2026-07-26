import 'package:flutter/material.dart';

import '../models/models.dart';
import '../services/trading_api.dart';
import '../theme.dart';

/// Runs a demo EMA/SMA-cross backtest for a symbol. A full visual strategy
/// builder (AND/OR/NOT over the 12 indicators) is a follow-up; this proves the
/// end-to-end path: symbol -> backend indicators + Backtester -> metrics.
class BacktestScreen extends StatefulWidget {
  const BacktestScreen({super.key});
  @override
  State<BacktestScreen> createState() => _BacktestScreenState();
}

class _BacktestScreenState extends State<BacktestScreen> {
  final _api = TradingApi();
  final _symbol = TextEditingController(text: 'RELIANCE');
  final _target = TextEditingController(text: '2');
  final _stop = TextEditingController(text: '3');
  bool _busy = false;
  String? _error;
  BacktestResult? _result;

  // close > SMA_20 -> long. Deterministic, white-box (SEBI-compliant).
  static const _rule = {
    'op': 'GT',
    'left': {'feature': 'close'},
    'right': {'feature': 'SMA_20'},
  };

  Future<void> _run() async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
    });
    try {
      final result = await _api.runBacktest(
        symbol: _symbol.text.trim().toUpperCase(),
        ruleTree: _rule,
        product: 'CNC',
        quantity: 10,
        targetPct: double.tryParse(_target.text),
        stopLossPct: double.tryParse(_stop.text),
      );
      setState(() => _result = result);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Backtest')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Strategy: close > SMA(20)', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 12),
          TextField(controller: _symbol, decoration: const InputDecoration(labelText: 'Symbol (NSE)')),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(child: TextField(controller: _target, decoration: const InputDecoration(labelText: 'Target %'), keyboardType: TextInputType.number)),
            const SizedBox(width: 12),
            Expanded(child: TextField(controller: _stop, decoration: const InputDecoration(labelText: 'Stop-loss %'), keyboardType: TextInputType.number)),
          ]),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _busy ? null : _run,
            icon: _busy ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.play_arrow),
            label: const Text('Run backtest'),
          ),
          if (_error != null) Padding(padding: const EdgeInsets.only(top: 16), child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error))),
          if (_result != null) ...[
            const SizedBox(height: 24),
            _MetricsCard(result: _result!),
          ],
        ],
      ),
    );
  }
}

class _MetricsCard extends StatelessWidget {
  const _MetricsCard({required this.result});
  final BacktestResult result;

  @override
  Widget build(BuildContext context) {
    final m = result.metrics;
    final netPnl = (m['net_pnl'] as num?)?.toDouble() ?? 0;
    Widget row(String k, String v) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [Text(k), Text(v, style: const TextStyle(fontWeight: FontWeight.w600))]),
        );
    String pct(dynamic v) => v == null ? '-' : '${((v as num) * 100).toStringAsFixed(1)}%';
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${result.symbol} · ${result.bars} bars', style: Theme.of(context).textTheme.titleMedium),
            const Divider(),
            row('Total trades', '${m['total_trades']}'),
            row('Win rate', pct(m['win_rate'])),
            row('Profit factor', '${m['profit_factor'] ?? '-'}'),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                const Text('Net P&L'),
                Text(netPnl.toStringAsFixed(2), style: TextStyle(fontWeight: FontWeight.w700, color: pnlColor(context, netPnl))),
              ]),
            ),
            row('Max drawdown', pct(m['max_drawdown_pct'])),
            row('Sharpe', '${m['sharpe']}'),
            row('Sortino', '${m['sortino']}'),
            row('CAGR', pct(m['cagr'])),
            const SizedBox(height: 8),
            Text('Final equity: ${result.finalEquity.toStringAsFixed(2)}', style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    );
  }
}
