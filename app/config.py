import os
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    app_name: str = "Resume Screening AI"
    version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'app.db')}"

    # Model paths
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = os.path.join(BASE_DIR, "data", "models", "faiss_index")
    classifier_path: str = os.path.join(BASE_DIR, "data", "models", "classifier.joblib")
    tfidf_path: str = os.path.join(BASE_DIR, "data", "models", "tfidf.joblib")
    label_encoder_path: str = os.path.join(BASE_DIR, "data", "models", "label_encoder.joblib")

    # LLM configuration
    llm_provider: LLMProvider = LLMProvider.GEMINI
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "llama3"
    ollama_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Scoring weights — must sum to 1.0
    semantic_weight: float = 0.40
    skill_weight: float = 0.30
    experience_weight: float = 0.15
    education_weight: float = 0.15

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def data_dir(self) -> str:
        return os.path.join(BASE_DIR, "data")

    @property
    def raw_data_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.data_dir, "models")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_dir, "processed")


settings = Settings()
