"""Управление конфигурацией сервиса через переменные окружения."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса кредитного скоринга.

    Загружает параметры из переменных окружения и файла configs/.env.
    Используется для вынесения путей, портов и флагов из кода.
    """

    # Сервис
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    app_title: str = "Credit Scoring API"
    app_version: str = "1.0.0"

    # Модель и артефакты
    # Путь к сериализованному пайплайну. По умолчанию относительно корня проекта.
    model_artifact_path: Path = Path("artifacts/modeling/models/scorer_pipeline_v1.0.joblib")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / "configs" / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )


settings = Settings()