import 'package:flutter/material.dart';

const _seed = Color(0xFF1565C0);

ThemeData buildLightTheme() => ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: _seed),
      useMaterial3: true,
    );

ThemeData buildDarkTheme() => ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: _seed, brightness: Brightness.dark),
      useMaterial3: true,
    );

/// Green for profit, red for loss - consistent across the app.
Color pnlColor(BuildContext context, double value) =>
    value >= 0 ? Colors.green.shade600 : Colors.red.shade600;
