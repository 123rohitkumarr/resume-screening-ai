from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


# ── Resume Input ──────────────────────────────────────────────

class ResumeInput(BaseModel):
    name: str = Field(..., min_length=1, examples=["Jane Doe"])
    raw_text: Optional[str] = None
    skills: Optional[list[str]] = None
    education: Optional[str] = None
    years_experience: Optional[float] = None


class ResumeFromFile(BaseModel):
    """Metadata sent alongside a file upload."""
    name: Optional[str] = None


# ── Job Description ───────────────────────────────────────────

class JobDescriptionInput(BaseModel):
    title: str = Field(..., examples=["Senior Data Scientist"])
    description: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience: float = 0
    education_required: Optional[str] = None


class ScreeningRequest(BaseModel):
    name: str = Field(..., examples=["Jane Doe"])
    resume_text: str
    jd_title: str
    jd_description: str
    required_skills: list[str]
    preferred_skills: Optional[list[str]] = Field(default_factory=list)
    min_experience: float = 0.0
    education_required: Optional[str] = None
    save_to_db: bool = True


# ── Scoring & Analysis ───────────────────────────────────────

class SkillAnalysis(BaseModel):
    matched: list[str]
    missing: list[str]
    bonus: list[str]
    match_percentage: float
    preferred_skills_match_count: int = 0


class ScoreBreakdown(BaseModel):
    overall: float = Field(..., ge=0, le=100)
    semantic: float = Field(..., ge=0, le=100)
    skill: float = Field(..., ge=0, le=100)
    experience: float = Field(..., ge=0, le=100)
    education: float = Field(..., ge=0, le=100)


class ExplanationItem(BaseModel):
    factor: str
    score: float
    impact: str  # positive / negative / neutral
    detail: str


class ScreeningResult(BaseModel):
    resume_id: Optional[int] = None
    job_id: Optional[int] = None
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    candidate_phone: Optional[str] = None
    extracted_experience: Optional[float] = None
    extracted_education: Optional[str] = None
    scores: ScoreBreakdown
    skill_analysis: SkillAnalysis
    predicted_role: Optional[str] = None
    explanation: list[ExplanationItem]
    suggestions: list[str]
    summary: str


# ── Chatbot ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    screening_context: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list[str] = Field(default_factory=list)


# ── System & Configuration ────────────────────────────────────

class HealthCheck(BaseModel):
    status: str
    version: str
    models_loaded: dict


class FeedbackRequest(BaseModel):
    screening_result: ScreeningResult
    focus_areas: Optional[list[str]] = None


class FeedbackResponse(BaseModel):
    overall_feedback: str
    skill_recommendations: list[dict]
    career_tips: list[str]
    learning_resources: list[dict]
    priority_actions: list[str]


class SettingsResponse(BaseModel):
    semantic_weight: float
    skill_weight: float
    experience_weight: float
    education_weight: float
    llm_provider: str
    gemini_model: str
    ollama_model: str
    ollama_url: str
    openai_model: str


class SettingsUpdateRequest(BaseModel):
    semantic_weight: Optional[float] = None
    skill_weight: Optional[float] = None
    experience_weight: Optional[float] = None
    education_weight: Optional[float] = None
    llm_provider: Optional[str] = None
    gemini_model: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_url: Optional[str] = None
    openai_model: Optional[str] = None
