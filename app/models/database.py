from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
from app.config import settings
from app.utils.helpers import ensure_dir
import os

# Make sure the data directory exists before connecting
ensure_dir(os.path.dirname(settings.database_url.replace("sqlite:///", "")))

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=settings.debug,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)


# ── ORM Models ────────────────────────────────────────────────

class ResumeRecord(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    raw_text = Column(Text, nullable=False)
    skills = Column(Text, default="[]")         # JSON array
    education = Column(String(100), nullable=True)
    years_experience = Column(Float, nullable=True)
    predicted_role = Column(String(100), nullable=True)
    role_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class JobDescriptionRecord(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    required_skills = Column(Text, default="[]")
    preferred_skills = Column(Text, default="[]")
    min_experience = Column(Float, default=0)
    education_required = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class ScreeningRecord(Base):
    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, nullable=False)
    job_id = Column(Integer, nullable=False)
    overall_score = Column(Float, nullable=False)
    semantic_score = Column(Float)
    skill_score = Column(Float)
    experience_score = Column(Float)
    education_score = Column(Float)
    matched_skills = Column(Text, default="[]")
    missing_skills = Column(Text, default="[]")
    explanation = Column(Text, nullable=True)
    suggestions = Column(Text, default="[]")
    created_at = Column(DateTime, default=_utcnow)


class ChatRecord(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True)
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


# ── Database lifecycle ────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency-injectable session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
