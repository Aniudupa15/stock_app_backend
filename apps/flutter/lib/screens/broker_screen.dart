import 'package:flutter/material.dart';

import '../models/models.dart';
import '../services/trading_api.dart';

class BrokerScreen extends StatefulWidget {
  const BrokerScreen({super.key});
  @override
  State<BrokerScreen> createState() => _BrokerScreenState();
}

class _BrokerScreenState extends State<BrokerScreen> {
  final _api = TradingApi();
  final _apiKey = TextEditingController();
  final _apiSecret = TextEditingController();
  BrokerStatus? _status;
  String? _loginUrl;
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    try {
      final status = await _api.brokerStatus();
      setState(() => _status = status);
    } catch (e) {
      setState(() => _error = '$e');
    }
  }

  Future<void> _connect() async {
    setState(() {
      _busy = true;
      _error = null;
      _loginUrl = null;
    });
    try {
      final url = await _api.brokerConnect(apiKey: _apiKey.text.trim(), apiSecret: _apiSecret.text.trim());
      setState(() => _loginUrl = url);
      await _refresh();
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final connected = _status?.connected ?? false;
    return Scaffold(
      appBar: AppBar(title: const Text('Broker (Zerodha)')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: ListTile(
              leading: Icon(connected ? Icons.check_circle : Icons.cancel, color: connected ? Colors.green : Colors.grey),
              title: Text('Status: ${_status?.status ?? "…"}'),
              subtitle: Text(connected ? 'Connected' : 'Not connected'),
            ),
          ),
          const SizedBox(height: 16),
          Text('Connect API credentials', style: Theme.of(context).textTheme.titleMedium),
          const Text('Stored encrypted at rest. Never shared back to the app.', style: TextStyle(fontSize: 12)),
          const SizedBox(height: 8),
          TextField(controller: _apiKey, decoration: const InputDecoration(labelText: 'API key')),
          const SizedBox(height: 8),
          TextField(controller: _apiSecret, decoration: const InputDecoration(labelText: 'API secret'), obscureText: true),
          const SizedBox(height: 12),
          FilledButton(onPressed: _busy ? null : _connect, child: const Text('Save & get login link')),
          if (_loginUrl != null) ...[
            const SizedBox(height: 16),
            const Text('Open this URL to log in to Zerodha, then paste the request_token to complete login:'),
            const SizedBox(height: 8),
            SelectableText(_loginUrl!, style: const TextStyle(fontFamily: 'monospace')),
            const SizedBox(height: 8),
            const Text('Note: live order placement also requires a static egress IP registered with Zerodha, '
                'and each session expires at ~06:00 IST the next day (re-login daily).', style: TextStyle(fontSize: 12)),
          ],
          if (_error != null) Padding(padding: const EdgeInsets.only(top: 12), child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error))),
        ],
      ),
    );
  }
}
