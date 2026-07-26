import 'package:flutter/material.dart';

import '../models/models.dart';
import '../services/trading_api.dart';

const _features = ['close', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14', 'VWAP_20', 'MACD', 'MACD_SIGNAL'];
const _operators = {
  'GT': '>',
  'LT': '<',
  'GTE': '>=',
  'LTE': '<=',
  'CROSS_ABOVE': 'crosses above',
  'CROSS_BELOW': 'crosses below',
};

class StrategiesScreen extends StatefulWidget {
  const StrategiesScreen({super.key});
  @override
  State<StrategiesScreen> createState() => _StrategiesScreenState();
}

class _StrategiesScreenState extends State<StrategiesScreen> {
  final _api = TradingApi();
  late Future<List<Strategy>> _strategies;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  void _reload() => setState(() => _strategies = _api.listStrategies());

  Future<void> _openCreate() async {
    final created = await showDialog<bool>(context: context, builder: (_) => const _CreateStrategyDialog());
    if (created == true) _reload();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Strategies')),
      floatingActionButton: FloatingActionButton.extended(onPressed: _openCreate, icon: const Icon(Icons.add), label: const Text('New strategy')),
      body: FutureBuilder<List<Strategy>>(
        future: _strategies,
        builder: (context, snap) {
          if (snap.connectionState == ConnectionState.waiting) return const Center(child: CircularProgressIndicator());
          if (snap.hasError) return Center(child: Text('${snap.error}'));
          final items = snap.data ?? [];
          if (items.isEmpty) return const Center(child: Text('No strategies yet.'));
          return ListView.builder(
            itemCount: items.length,
            itemBuilder: (context, i) {
              final s = items[i];
              return ListTile(
                leading: const Icon(Icons.rule),
                title: Text(s.name),
                subtitle: Text('${s.side} · ${s.product} · qty ${s.quantity}'),
                trailing: Chip(label: Text(s.status)),
              );
            },
          );
        },
      ),
    );
  }
}

class _CreateStrategyDialog extends StatefulWidget {
  const _CreateStrategyDialog();
  @override
  State<_CreateStrategyDialog> createState() => _CreateStrategyDialogState();
}

class _CreateStrategyDialogState extends State<_CreateStrategyDialog> {
  final _api = TradingApi();
  final _name = TextEditingController();
  final _rightConst = TextEditingController(text: '55');
  final _target = TextEditingController(text: '2');
  final _stop = TextEditingController(text: '3');
  String _left = 'close';
  String _op = 'GT';
  String _rightFeature = 'SMA_20';
  bool _compareToNumber = false;
  String _side = 'BUY';
  String _product = 'MIS';
  bool _busy = false;
  String? _error;

  Map<String, dynamic> _buildRule() => {
        'op': _op,
        'left': {'feature': _left},
        'right': _compareToNumber ? {'const': double.tryParse(_rightConst.text) ?? 0} : {'feature': _rightFeature},
      };

  Future<void> _save() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await _api.createStrategy(
        name: _name.text.trim(),
        ruleTree: _buildRule(),
        side: _side,
        product: _product,
        quantity: 10,
        targetPct: double.tryParse(_target.text),
        stopLossPct: double.tryParse(_stop.text),
      );
      if (mounted) Navigator.of(context).pop(true);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('New strategy'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: _name, decoration: const InputDecoration(labelText: 'Name')),
            const SizedBox(height: 12),
            const Align(alignment: Alignment.centerLeft, child: Text('Entry rule')),
            Row(children: [
              Expanded(child: _dropdown(_left, _features, (v) => setState(() => _left = v))),
              const SizedBox(width: 8),
              Expanded(child: _dropdown(_op, _operators.keys.toList(), (v) => setState(() => _op = v), labels: _operators)),
            ]),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Compare to a number'),
              value: _compareToNumber,
              onChanged: (v) => setState(() => _compareToNumber = v),
            ),
            _compareToNumber
                ? TextField(controller: _rightConst, decoration: const InputDecoration(labelText: 'Value'), keyboardType: TextInputType.number)
                : _dropdown(_rightFeature, _features, (v) => setState(() => _rightFeature = v)),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: _dropdown(_side, const ['BUY', 'SELL'], (v) => setState(() => _side = v))),
              const SizedBox(width: 8),
              Expanded(child: _dropdown(_product, const ['MIS', 'CNC'], (v) => setState(() => _product = v))),
            ]),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: TextField(controller: _target, decoration: const InputDecoration(labelText: 'Target %'), keyboardType: TextInputType.number)),
              const SizedBox(width: 8),
              Expanded(child: TextField(controller: _stop, decoration: const InputDecoration(labelText: 'Stop %'), keyboardType: TextInputType.number)),
            ]),
            if (_error != null) Padding(padding: const EdgeInsets.only(top: 12), child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error))),
          ],
        ),
      ),
      actions: [
        TextButton(onPressed: _busy ? null : () => Navigator.of(context).pop(false), child: const Text('Cancel')),
        FilledButton(onPressed: _busy ? null : _save, child: _busy ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Create')),
      ],
    );
  }

  Widget _dropdown(String value, List<String> items, ValueChanged<String> onChanged, {Map<String, String>? labels}) {
    return DropdownButtonFormField<String>(
      value: value,
      isExpanded: true,
      items: items.map((e) => DropdownMenuItem(value: e, child: Text(labels?[e] ?? e))).toList(),
      onChanged: (v) => v == null ? null : onChanged(v),
    );
  }
}
