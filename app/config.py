import os
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import model_validator

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
    ohe_path: str = os.path.join(BASE_DIR, "data", "models", "ohe.joblib")
    scaler_path: str = os.path.join(BASE_DIR, "data", "models", "scaler.joblib")

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
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    cors_origins: str = "*"
    api_key: Optional[str] = None
    max_upload_size_mb: int = 10

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def validate_weights(self):
        total = self.semantic_weight + self.skill_weight + self.experience_weight + self.education_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0 (current sum: {total:.2f})")
        return self

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
