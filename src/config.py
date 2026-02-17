"""Configuration management for the energy forecast pipeline."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    EIA_API_KEY: str = ""
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    DEFAULT_REGION: str = "CISO"

    model_config = {"env_file": PROJECT_ROOT / ".env", "env_file_encoding": "utf-8"}


settings = Settings()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


setup_logging()
