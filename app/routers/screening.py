import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import Optional, List

from app.models.database import get_db, ResumeRecord, JobDescriptionRecord, ScreeningRecord
from app.models.schemas import ScreeningResult, ScoreBreakdown, SkillAnalysis, ExplanationItem, ScreeningRequest
from app.services.parser import parse_resume
from app.services.preprocessor import extract_experience_years, extract_education
from app.services.skill_extractor import extract_skills, match_skills
from app.services.classifier import classifier
from app.services.embeddings import embedding_engine
from app.services.matcher import matching_engine
from app.services.explainer import explainer
from app.utils.helpers import setup_logger, clean_text

log = setup_logger(__name__)
router = APIRouter(prefix="/api/screening", tags=["Screening"])

def _run_screening_pipeline(
    name: str,
    resume_text: str,
    jd_title: str,
    jd_description: str,
    required_skills: List[str],
    preferred_skills: Optional[List[str]] = None,
    min_experience: float = 0.0,
    education_required: Optional[str] = None,
    save_to_db: bool = True,
    db: Session = None
) -> ScreeningResult:
    """Internal helper to execute the screening pipeline."""
    # Preprocess / Extract fields if not present
    candidate_skills = extract_skills(resume_text)
    candidate_exp = extract_experience_years(resume_text) or 0.0
    candidate_edu = extract_education(resume_text) or "Bachelors"

    # Classification predicted role
    pred_role, confidence = classifier.predict(candidate_skills, candidate_edu, candidate_exp)

    # Match skills
    preferred_skills = preferred_skills or []
    skill_analysis = match_skills(candidate_skills, required_skills, preferred_skills)

    # Calculate scores
    scores = matching_engine.compute_overall_score(
        resume_text=resume_text,
        jd_text=jd_description,
        skill_analysis=skill_analysis,
        candidate_exp=candidate_exp,
        required_exp=min_experience,
        candidate_edu=candidate_edu,
        required_edu=education_required
    )

    # Generate XAI explanations and suggestions
    explanation = explainer.explain(
        scores=scores,
        skill_analysis=skill_analysis,
        candidate_exp=candidate_exp,
        required_exp=min_experience,
        candidate_edu=candidate_edu,
        required_edu=education_required
    )
    suggestions = explainer.generate_suggestions(
        scores=scores,
        skill_analysis=skill_analysis,
        candidate_exp=candidate_exp,
        required_exp=min_experience
    )
    summary = explainer.generate_summary(
        scores=scores,
        skill_analysis=skill_analysis,
        predicted_role=pred_role
    )

    resume_id = None
    job_id = None

    if save_to_db and db:
        # Save Resume
        res_rec = ResumeRecord(
            name=name,
            raw_text=resume_text,
            skills=json.dumps(candidate_skills),
            education=candidate_edu,
            years_experience=candidate_exp,
            predicted_role=pred_role,
            role_confidence=confidence
        )
        db.add(res_rec)
        db.commit()
        db.refresh(res_rec)
        resume_id = res_rec.id

        # Save Job Description
        jd_rec = JobDescriptionRecord(
            title=jd_title,
            description=jd_description,
            required_skills=json.dumps(required_skills),
            preferred_skills=json.dumps(preferred_skills),
            min_experience=min_experience,
            education_required=education_required
        )
        db.add(jd_rec)
        db.commit()
        db.refresh(jd_rec)
        job_id = jd_rec.id

        # Save Screening Result
        screen_rec = ScreeningRecord(
            resume_id=resume_id,
            job_id=job_id,
            overall_score=scores.overall,
            semantic_score=scores.semantic,
            skill_score=scores.skill,
            experience_score=scores.experience,
            education_score=scores.education,
            matched_skills=json.dumps(skill_analysis.matched),
            missing_skills=json.dumps(skill_analysis.missing),
            explanation=json.dumps([item.model_dump() for item in explanation]),
            suggestions=json.dumps(suggestions)
        )
        db.add(screen_rec)
        db.commit()

    return ScreeningResult(
        resume_id=resume_id,
        job_id=job_id,
        scores=scores,
        skill_analysis=skill_analysis,
        predicted_role=pred_role,
        explanation=explanation,
        suggestions=suggestions,
        summary=summary
    )


@router.post("/", response_model=ScreeningResult)
def screen_resume_text(
    request: ScreeningRequest,
    db: Session = Depends(get_db)
):
    """
    Screen a candidate's resume text against a job description.
    """
    try:
        return _run_screening_pipeline(
            name=request.name,
            resume_text=request.resume_text,
            jd_title=request.jd_title,
            jd_description=request.jd_description,
            required_skills=request.required_skills,
            preferred_skills=request.preferred_skills,
            min_experience=request.min_experience,
            education_required=request.education_required,
            save_to_db=request.save_to_db,
            db=db
        )
    except Exception as e:
        log.error(f"Error screening resume text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screening pipeline error: {str(e)}"
        )


@router.post("/file", response_model=ScreeningResult)
async def screen_resume_file(
    name: str = Form(...),
    jd_title: str = Form(...),
    jd_description: str = Form(...),
    required_skills_json: str = Form(...), # Comma separated or json array
    preferred_skills_json: Optional[str] = Form(None),
    min_experience: float = Form(0.0),
    education_required: Optional[str] = Form(None),
    save_to_db: bool = Form(True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Screen a candidate's uploaded resume file (PDF, DOCX, or text) against a job description.
    """
    try:
        # Load and parse file
        file_bytes = await file.read()
        resume_text = parse_resume(file_bytes=file_bytes, filename=file.filename)
        
        # Parse required/preferred skills
        try:
            required_skills = json.loads(required_skills_json)
        except json.JSONDecodeError:
            required_skills = [s.strip() for s in required_skills_json.split(",") if s.strip()]

        preferred_skills = []
        if preferred_skills_json:
            try:
                preferred_skills = json.loads(preferred_skills_json)
            except json.JSONDecodeError:
                preferred_skills = [s.strip() for s in preferred_skills_json.split(",") if s.strip()]

        return _run_screening_pipeline(
            name=name,
            resume_text=resume_text,
            jd_title=jd_title,
            jd_description=jd_description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience=min_experience,
            education_required=education_required,
            save_to_db=save_to_db,
            db=db
        )

    except Exception as e:
        log.error(f"Error screening resume file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screening file pipeline error: {str(e)}"
        )
