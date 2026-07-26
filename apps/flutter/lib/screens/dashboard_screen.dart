import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/auth_service.dart';
import '../services/trading_api.dart';
import 'account_detail_screen.dart';
import 'backtest_screen.dart';
import 'broker_screen.dart';
import 'login_screen.dart';
import 'strategies_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});
  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final _api = TradingApi();
  final _auth = AuthService();
  final _money = NumberFormat.currency(locale: 'en_IN', symbol: '₹');
  late Future<List<Account>> _accounts;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  void _reload() => setState(() => _accounts = _api.listAccounts());

  Future<void> _createAccount() async {
    try {
      await _api.createAccount();
      _reload();
    } catch (e) {
      _snack('Create failed: $e');
    }
  }

  Future<void> _logout() async {
    await _auth.logout();
    if (mounted) {
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const LoginScreen()));
    }
  }

  void _snack(String msg) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Trading Accounts'),
        actions: [
          IconButton(icon: const Icon(Icons.science_outlined), tooltip: 'Backtest', onPressed: () {
            Navigator.of(context).push(MaterialPageRoute(builder: (_) => const BacktestScreen()));
          }),
          IconButton(icon: const Icon(Icons.logout), tooltip: 'Log out', onPressed: _logout),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          children: [
            const DrawerHeader(child: Center(child: Text('Algo Trading'))),
            ListTile(
              leading: const Icon(Icons.rule),
              title: const Text('Strategies'),
              onTap: () {
                Navigator.of(context).pop();
                Navigator.of(context).push(MaterialPageRoute(builder: (_) => const StrategiesScreen()));
              },
            ),
            ListTile(
              leading: const Icon(Icons.science_outlined),
              title: const Text('Backtest'),
              onTap: () {
                Navigator.of(context).pop();
                Navigator.of(context).push(MaterialPageRoute(builder: (_) => const BacktestScreen()));
              },
            ),
            ListTile(
              leading: const Icon(Icons.account_balance),
              title: const Text('Broker'),
              onTap: () {
                Navigator.of(context).pop();
                Navigator.of(context).push(MaterialPageRoute(builder: (_) => const BrokerScreen()));
              },
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _createAccount,
        icon: const Icon(Icons.add),
        label: const Text('Paper account'),
      ),
      body: RefreshIndicator(
        onRefresh: () async => _reload(),
        child: FutureBuilder<List<Account>>(
          future: _accounts,
          builder: (context, snap) {
            if (snap.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snap.hasError) {
              return _ErrorView(message: '${snap.error}', onRetry: _reload);
            }
            final accounts = snap.data ?? [];
            if (accounts.isEmpty) {
              return const Center(child: Text('No accounts yet. Create a paper account to begin.'));
            }
            return ListView.separated(
              padding: const EdgeInsets.all(12),
              itemCount: accounts.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (context, i) {
                final a = accounts[i];
                return Card(
                  child: ListTile(
                    leading: CircleAvatar(child: Text(a.mode == 'LIVE' ? 'L' : 'P')),
                    title: Text('${a.mode} account'),
                    subtitle: Text('Balance: ${_money.format(a.virtualBalance ?? 0)}'),
                    trailing: a.mode == 'LIVE'
                        ? const Chip(label: Text('LIVE'), backgroundColor: Color(0x22FF0000))
                        : const Chip(label: Text('PAPER')),
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => AccountDetailScreen(account: a)),
                    ),
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.message, required this.onRetry});
  final String message;
  final VoidCallback onRetry;
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.cloud_off, size: 48),
          const SizedBox(height: 8),
          Padding(padding: const EdgeInsets.all(16), child: Text(message, textAlign: TextAlign.center)),
          FilledButton(onPressed: onRetry, child: const Text('Retry')),
        ],
      ),
    );
  }
}
