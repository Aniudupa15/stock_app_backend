import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/trading_api.dart';
import '../theme.dart';

class AccountDetailScreen extends StatefulWidget {
  const AccountDetailScreen({super.key, required this.account});
  final Account account;
  @override
  State<AccountDetailScreen> createState() => _AccountDetailScreenState();
}

class _AccountDetailScreenState extends State<AccountDetailScreen> {
  final _api = TradingApi();
  final _money = NumberFormat.currency(locale: 'en_IN', symbol: '₹');
  final _symbol = TextEditingController(text: 'RELIANCE');
  List<Strategy> _strategies = [];
  String? _selectedStrategyId;
  List<Trade> _trades = [];
  List<EquityPoint> _equity = [];
  bool _killSwitch = false;
  bool _busy = false;
  String? _message;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final results = await Future.wait([
        _api.listStrategies(),
        _api.listTrades(widget.account.id),
        _api.equityCurve(widget.account.id),
      ]);
      setState(() {
        _strategies = results[0] as List<Strategy>;
        _trades = results[1] as List<Trade>;
        _equity = results[2] as List<EquityPoint>;
        _selectedStrategyId ??= _strategies.isNotEmpty ? _strategies.first.id : null;
      });
    } catch (e) {
      setState(() => _message = '$e');
    }
  }

  Future<void> _runPaper() async {
    if (_selectedStrategyId == null) {
      setState(() => _message = 'Create a strategy first.');
      return;
    }
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      final res = await _api.paperRun(accountId: widget.account.id, strategyId: _selectedStrategyId!, symbol: _symbol.text.trim().toUpperCase());
      setState(() => _message = 'Paper run: ${res['trades']} trades, net ₹${res['net_pnl']}');
      await _load();
    } catch (e) {
      setState(() => _message = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _toggleKill(bool on) async {
    setState(() => _killSwitch = on);
    try {
      await _api.setKillSwitch(widget.account.id, on);
    } catch (e) {
      setState(() => _message = '$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('${widget.account.mode} account')),
      body: RefreshIndicator(
        onRefresh: _load,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text('Balance: ${_money.format(widget.account.virtualBalance ?? 0)}', style: Theme.of(context).textTheme.titleLarge),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Kill switch'),
              subtitle: const Text('Block all new entry orders'),
              value: _killSwitch,
              onChanged: _toggleKill,
            ),
            const Divider(),
            Text('Run paper session', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              value: _selectedStrategyId,
              isExpanded: true,
              decoration: const InputDecoration(labelText: 'Strategy'),
              items: _strategies.map((s) => DropdownMenuItem(value: s.id, child: Text(s.name))).toList(),
              onChanged: (v) => setState(() => _selectedStrategyId = v),
            ),
            const SizedBox(height: 8),
            TextField(controller: _symbol, decoration: const InputDecoration(labelText: 'Symbol (NSE)')),
            const SizedBox(height: 8),
            FilledButton.icon(
              onPressed: _busy ? null : _runPaper,
              icon: _busy ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.play_arrow),
              label: const Text('Run over history (paper)'),
            ),
            if (_message != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_message!)),
            const SizedBox(height: 16),
            if (_equity.length >= 2) ...[
              Text('Equity curve', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              SizedBox(height: 200, child: _EquityChart(points: _equity)),
            ],
            const SizedBox(height: 16),
            Text('Trade journal (${_trades.length})', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ..._trades.map((t) => Card(
                  child: ListTile(
                    dense: true,
                    title: Text('${t.symbol}  ×${t.qty}'),
                    subtitle: Text('${t.entryPrice.toStringAsFixed(2)} → ${t.exitPrice.toStringAsFixed(2)}  (${t.exitReason ?? "-"})'),
                    trailing: Text(t.pnlNet.toStringAsFixed(2), style: TextStyle(fontWeight: FontWeight.w700, color: pnlColor(context, t.pnlNet))),
                  ),
                )),
          ],
        ),
      ),
    );
  }
}

class _EquityChart extends StatelessWidget {
  const _EquityChart({required this.points});
  final List<EquityPoint> points;

  @override
  Widget build(BuildContext context) {
    final spots = <FlSpot>[
      for (var i = 0; i < points.length; i++) FlSpot(i.toDouble(), points[i].equity),
    ];
    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: false),
        titlesData: const FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: false,
            dotData: const FlDotData(show: false),
            color: Theme.of(context).colorScheme.primary,
            belowBarData: BarAreaData(show: true, color: Theme.of(context).colorScheme.primary.withOpacity(0.15)),
          ),
        ],
      ),
    );
  }
}
