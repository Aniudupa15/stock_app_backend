import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Thin HTTP client that injects the JWT access token and decodes JSON,
/// raising [ApiException] on non-2xx responses. Shared by the data-service
/// (auth) and trading-service calls.
class ApiException implements Exception {
  ApiException(this.statusCode, this.message);
  final int statusCode;
  final String message;
  @override
  String toString() => 'ApiException($statusCode): $message';
}

class ApiClient {
  ApiClient(this.baseUrl);
  final String baseUrl;

  static const _tokenKey = 'access_token';

  Future<void> setToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token);
  }

  Future<void> clearToken() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
  }

  Future<String?> token() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey);
  }

  Future<Map<String, String>> _headers({bool auth = true}) async {
    final headers = {'Content-Type': 'application/json'};
    if (auth) {
      final t = await token();
      if (t != null) headers['Authorization'] = 'Bearer $t';
    }
    return headers;
  }

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  dynamic _decode(http.Response resp) {
    if (resp.statusCode >= 200 && resp.statusCode < 300) {
      return resp.body.isEmpty ? null : jsonDecode(resp.body);
    }
    String message = resp.body;
    try {
      final decoded = jsonDecode(resp.body);
      if (decoded is Map && decoded['detail'] != null) {
        message = decoded['detail'].toString();
      }
    } catch (_) {}
    throw ApiException(resp.statusCode, message);
  }

  Future<dynamic> get(String path, {bool auth = true}) async =>
      _decode(await http.get(_uri(path), headers: await _headers(auth: auth)));

  Future<dynamic> post(String path, {Object? body, bool auth = true}) async => _decode(
        await http.post(_uri(path), headers: await _headers(auth: auth), body: jsonEncode(body ?? {})),
      );

  Future<dynamic> patch(String path, {Object? body}) async => _decode(
        await http.patch(_uri(path), headers: await _headers(), body: jsonEncode(body ?? {})),
      );

  Future<void> delete(String path) async => _decode(await http.delete(_uri(path), headers: await _headers()));
}
