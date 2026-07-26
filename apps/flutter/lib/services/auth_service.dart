import '../config.dart';
import 'api_client.dart';

/// Authenticates against the data-service (which issues the shared JWT the
/// trading-service also accepts). Token is persisted by ApiClient.
class AuthService {
  AuthService() : _client = ApiClient(AppConfig.dataBaseUrl);
  final ApiClient _client;

  Future<bool> isLoggedIn() async => (await _client.token()) != null;

  Future<void> login(String email, String password) async {
    final data = await _client.post(
      '/api/v1/auth/login',
      body: {'email': email, 'password': password},
      auth: false,
    );
    final token = (data as Map)['access_token'] as String;
    await _client.setToken(token);
  }

  /// Registers a new user and logs them in (the endpoint returns a token pair).
  Future<void> register(String email, String password, String displayName) async {
    final data = await _client.post(
      '/api/v1/auth/register',
      body: {'email': email, 'password': password, 'display_name': displayName},
      auth: false,
    );
    final token = (data as Map)['access_token'] as String;
    await _client.setToken(token);
  }

  Future<void> logout() => _client.clearToken();
}
