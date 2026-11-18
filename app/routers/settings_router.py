from fastapi import APIRouter, HTTPException, status
from app.config import settings, LLMProvider
from app.models.schemas import SettingsResponse, SettingsUpdateRequest
from app.utils.helpers import setup_logger

log = setup_logger(__name__)
router = APIRouter(prefix="/api/settings", tags=["Settings"])


@router.get("/", response_model=SettingsResponse)
def get_settings():
    """Retrieve current runtime application scoring weights and LLM configurations."""
    return SettingsResponse(
        semantic_weight=settings.semantic_weight,
        skill_weight=settings.skill_weight,
        experience_weight=settings.experience_weight,
        education_weight=settings.education_weight,
        llm_provider=settings.llm_provider.value if isinstance(settings.llm_provider, LLMProvider) else str(settings.llm_provider),
        gemini_model=settings.gemini_model,
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        openai_model=settings.openai_model,
    )


@router.put("/", response_model=SettingsResponse)
def update_settings(update: SettingsUpdateRequest):
    """Update runtime scoring weights and LLM configurations with validation."""
    # Determine new weights
    sem = update.semantic_weight if update.semantic_weight is not None else settings.semantic_weight
    skill = update.skill_weight if update.skill_weight is not None else settings.skill_weight
    exp = update.experience_weight if update.experience_weight is not None else settings.experience_weight
    edu = update.education_weight if update.education_weight is not None else settings.education_weight

    total = sem + skill + exp + edu
    if abs(total - 1.0) > 0.01:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Scoring weights must sum to 1.0 (current sum: {total:.2f})",
        )

    # Apply updates
    settings.semantic_weight = sem
    settings.skill_weight = skill
    settings.experience_weight = exp
    settings.education_weight = edu

    if update.llm_provider is not None:
        try:
            settings.llm_provider = LLMProvider(update.llm_provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid LLM provider '{update.llm_provider}'. Allowed: gemini, ollama, openai",
            )

    if update.gemini_model:
        settings.gemini_model = update.gemini_model
    if update.ollama_model:
        settings.ollama_model = update.ollama_model
    if update.ollama_url:
        settings.ollama_url = update.ollama_url
    if update.openai_model:
        settings.openai_model = update.openai_model

    log.info(f"Updated settings: weights=({sem}, {skill}, {exp}, {edu}), provider={settings.llm_provider}")

    return get_settings()
