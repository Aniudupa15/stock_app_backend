from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_NAME: str = "Indian Stock Market Backend"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ALLOWED_ORIGINS: str = "*"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://stockapp:stockapp@localhost:5432/stockapp"

    # NSE provider
    NSE_BASE_URL: str = "https://www.nseindia.com"
    NSE_REQUEST_TIMEOUT_SECONDS: float = 15.0
    NSE_MAX_RETRIES: int = 3
    NSE_RATE_LIMIT_PER_SECOND: float = 3.0
    NSE_COOKIE_TTL_SECONDS: int = 300
    NSE_CIRCUIT_FAIL_MAX: int = 5
    NSE_CIRCUIT_RESET_TIMEOUT_SECONDS: int = 60

    # Cache
    CACHE_BACKEND: str = "memory"  # "memory" or "redis"
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_SEARCH_TTL_SECONDS: int = 60
    CACHE_QUOTE_TTL_SECONDS: int = 15
    CACHE_DASHBOARD_TTL_SECONDS: int = 30

    # Auth
    # Dev-only placeholder - MUST be overridden via .env in any real deployment.
    JWT_SECRET_KEY: str = "dev-only-insecure-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Scheduler
    SCHEDULER_ENABLED: bool = True
    UNIVERSE_SYNC_HOUR_IST: int = 8
    UNIVERSE_SYNC_MINUTE_IST: int = 0
    PRICE_SYNC_HOUR_IST: int = 18
    PRICE_SYNC_MINUTE_IST: int = 0
    CORPORATE_ACTIONS_SYNC_HOUR_IST: int = 7
    CORPORATE_ACTIONS_SYNC_MINUTE_IST: int = 30
    FINANCIAL_RESULTS_SYNC_HOUR_IST: int = 9
    FINANCIAL_RESULTS_SYNC_MINUTE_IST: int = 0
    NEWS_SYNC_INTERVAL_MINUTES: int = 30
    ALERT_EVALUATION_INTERVAL_MINUTES: int = 15
    INDICATOR_SNAPSHOT_SYNC_HOUR_IST: int = 18
    INDICATOR_SNAPSHOT_SYNC_MINUTE_IST: int = 30
    IPO_SYNC_HOUR_IST: int = 7
    IPO_SYNC_MINUTE_IST: int = 45

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
