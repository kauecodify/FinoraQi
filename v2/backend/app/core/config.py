from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FinoraQi API"
    app_version: str = "2.0.0"
    env: str = "development"

    database_url: str = "sqlite:///./finoraqi.db"

    jwt_secret_key: str = Field(default="change-me-in-production-0123456789abcdef", min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    cors_origins: list[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]

    models_dir: str = "./models_store"
    frontend_dir: str = "../frontend"
    auto_create_tables: bool = True
    fx_provider: str = "static"
    max_upload_mb: int = 20

    default_feature_flags: dict[str, bool] = {
        "cashpool.enabled": True,
        "cashpool.sweep": True,
        "cashpool.netting": True,
        "cashpool.yield": True,
        "automl.enabled": True,
    }

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
