import 'package:algo_trading_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

// Committed so `flutter create` in CI doesn't drop in a template test that
// references the non-existent `MyApp`. Analysis-only in CI (no `flutter test`
// step); it just needs to reference real symbols.
void main() {
  testWidgets('app widget constructs', (tester) async {
    const app = AlgoTradingApp();
    expect(app, isA<AlgoTradingApp>());
  });
}
