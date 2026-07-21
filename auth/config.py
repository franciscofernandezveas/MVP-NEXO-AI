from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # Ignora variables de entorno que no estén definidas aquí
    )

    DATABASE_URL: str = ""

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    FRONTEND_URL: str = "http://localhost:3000"
    API_URL: str = "http://localhost:8001"
    SECURE_COOKIES: bool = False
    COOKIE_DOMAIN: str | None = None
    ACCESS_TOKEN_COOKIE_NAME: str = "access_token"
    REFRESH_TOKEN_COOKIE_NAME: str = "refresh_token"
    PASSWORD_MIN_LENGTH: int = 8


settings = AuthSettings()
