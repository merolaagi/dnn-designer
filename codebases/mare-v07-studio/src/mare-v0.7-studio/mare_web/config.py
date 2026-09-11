from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MARE_", env_file=".env", extra="ignore")
    environment: Literal["development", "production", "test"] = "development"
    database_url: str = "sqlite+aiosqlite:///./mare-web.db"
    auth_token: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver", "api"]
    rate_limit_per_minute: int = Field(default=120, ge=1)
    max_body_bytes: int = Field(default=2000000, ge=1024)
    lease_seconds: int = Field(default=30, ge=10)
    round_timeout_seconds: int = Field(default=600, ge=1, le=3600)
    max_attempts: int = Field(default=3, ge=1, le=10)
    poll_seconds: float = Field(default=1, ge=0.05)
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    planner_capture: bool = False
    planner_checkpoint: str | None = None
    planner_mode: Literal["shadow", "blend"] = "shadow"
    openai_enabled: bool = False
    openai_model: str = ""

    @model_validator(mode="after")
    def production_safety(self):
        if self.environment == "production":
            if not self.auth_token or len(self.auth_token.get_secret_value()) < 32:
                raise ValueError("Production requires MARE_AUTH_TOKEN with at least 32 characters")
            if "*" in self.cors_origins or "*" in self.allowed_hosts:
                raise ValueError("Production requires explicit origins and hosts")
        if self.openai_enabled and not self.openai_model:
            raise ValueError("Set MARE_OPENAI_MODEL explicitly when enabling OpenAI")
        return self
