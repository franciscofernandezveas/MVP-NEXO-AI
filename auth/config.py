from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    DATABASE_URL: str = ""

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_JWT_SECRET: str

    # Emails que serán admin al sincronizarse (separados por coma)
    ADMIN_EMAILS: str = ""

    FRONTEND_URL: str = "http://localhost:3000"
    API_URL: str = "http://localhost:8001"
    SECURE_COOKIES: bool = False
    COOKIE_DOMAIN: str | None = None
    ACCESS_TOKEN_COOKIE_NAME: str = "access_token"

    @property
    def admin_email_set(self) -> Set[str]:
        return {e.strip().lower() for e in self.ADMIN_EMAILS.split(",") if e.strip()}


settings = AuthSettings()
