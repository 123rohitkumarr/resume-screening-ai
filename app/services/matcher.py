import math
import numpy as np
from typing import Optional, List, Dict
from app.config import settings
from app.models.schemas import ScoreBreakdown, SkillAnalysis
from app.utils.helpers import setup_logger, clamp

log = setup_logger(__name__)


class MatchingEngine:
    def __init__(self, embedding_engine=None):
        if embedding_engine is None:
            from app.services.embeddings import embedding_engine as default_engine
            self.embedding_engine = default_engine
        else:
            self.embedding_engine = embedding_engine

    def compute_semantic_score(self, resume_text: str, jd_text: str) -> float:
        """Compute cosine similarity score between resume text and job description."""
        if not resume_text.strip() or not jd_text.strip():
            return 0.0

        res_emb = self.embedding_engine.encode_single(resume_text)
        jd_emb = self.embedding_engine.encode_single(jd_text)

        res_norm = res_emb / (np.linalg.norm(res_emb) or 1.0)
        jd_norm = jd_emb / (np.linalg.norm(jd_emb) or 1.0)

        similarity = float(np.dot(res_norm, jd_norm))
        score = max(0.0, similarity) * 100.0
        return clamp(round(score, 2), 0.0, 100.0)

    def compute_skill_score(self, skill_analysis: SkillAnalysis) -> float:
        """Compute skill score with preferred skill bonus."""
        base_score = getattr(skill_analysis, "match_percentage", 0.0)
        bonus = getattr(skill_analysis, "preferred_skills_match_count", 0) * 5.0
        return clamp(round(base_score + bonus, 2), 0.0, 100.0)

    def compute_experience_score(self, candidate_years: Optional[float], required_years: float) -> float:
        """
        Compute experience compatibility score using Gaussian penalty.
        Clearly handles:
        - candidate_years is None (missing/unknown experience)
        - candidate_years == 0.0 (actual zero experience / fresher)
        - candidate_years >= required_years
        """
        if candidate_years is None:
            # Experience not specified in resume
            return 100.0 if required_years <= 0 else 50.0

        req = max(0.0, float(required_years))
        cand = max(0.0, float(candidate_years))

        if cand < req:
            gap = req - cand
            penalty = 100.0 * math.exp(-0.5 * ((gap / max(req, 1.0)) ** 2))
            return clamp(round(penalty, 2), 0.0, 100.0)
        else:
            over = cand - req
            if over <= 3.0:
                return 100.0
            else:
                score = 100.0 - 5.0 * (over - 3.0)
                return clamp(round(score, 2), 0.0, 100.0)

    def compute_education_score(self, candidate_edu: Optional[str], required_edu: Optional[str]) -> float:
        """
        Compute ordinal education match score.
        Differentiates between unknown education (cand_level = 0) vs fulfilled requirements.
        """
        edu_order = {
            "high school": 1,
            "associate": 2,
            "bachelors": 3,
            "bachelor": 3,
            "masters": 4,
            "master": 4,
            "phd": 5,
            "doctorate": 5
        }

        cand_str = str(candidate_edu).lower().strip() if candidate_edu else ""
        req_str = str(required_edu).lower().strip() if required_edu else ""

        cand_level = edu_order.get(cand_str, 0)
        req_level = edu_order.get(req_str, 0)

        # If job does not specify education or says None/Any
        if req_level == 0 or req_str in ["none", "any", "unspecified", "unknown"]:
            return 100.0

        # Candidate education missing/unknown
        if cand_level == 0:
            return 30.0

        diff = cand_level - req_level
        if diff >= 0:
            return 100.0
        elif diff == -1:
            return 70.0
        elif diff == -2:
            return 40.0
        else:
            return 20.0

    def compute_overall_score(
        self,
        resume_text: str,
        jd_text: str,
        skill_analysis: SkillAnalysis,
        candidate_exp: Optional[float],
        required_exp: float,
        candidate_edu: Optional[str],
        required_edu: Optional[str]
    ) -> ScoreBreakdown:
        """Calculate weighted composite score adhering to configured weights (40/30/15/15)."""
        semantic = self.compute_semantic_score(resume_text, jd_text)
        skills = self.compute_skill_score(skill_analysis)
        experience = self.compute_experience_score(candidate_exp, required_exp)
        education = self.compute_education_score(candidate_edu, required_edu)

        w_sem = getattr(settings, "semantic_weight", 0.40)
        w_skill = getattr(settings, "skill_weight", 0.30)
        w_exp = getattr(settings, "experience_weight", 0.15)
        w_edu = getattr(settings, "education_weight", 0.15)

        overall = (semantic * w_sem) + (skills * w_skill) + (experience * w_exp) + (education * w_edu)
        overall = clamp(round(overall, 2), 0.0, 100.0)

        return ScoreBreakdown(
            overall=overall,
            semantic=semantic,
            skill=skills,
            experience=experience,
            education=education
        )

    def rank_candidates(self, resumes: List[dict], job_description: dict) -> List[dict]:
        """Rank multiple resumes against a job description in descending order of overall score."""
        jd_text = job_description.get("text", "")
        req_exp = job_description.get("required_experience", 0.0)
        req_edu = job_description.get("required_education", "Bachelors")

        results = []
        for res in resumes:
            res_text = res.get("text", "")
            skill_analysis = res.get("skill_analysis")
            if not skill_analysis:
                from app.services.skill_extractor import match_skills
                skills = res.get("skills", [])
                req_skills = job_description.get("required_skills", [])
                pref_skills = job_description.get("preferred_skills", [])
                skill_analysis = match_skills(skills, req_skills, pref_skills)

            cand_exp = res.get("experience_years")
            cand_edu = res.get("education_level")

            score_breakdown = self.compute_overall_score(
                res_text, jd_text, skill_analysis, cand_exp, req_exp, cand_edu, req_edu
            )

            res_copy = dict(res)
            res_copy["score_breakdown"] = score_breakdown
            res_copy["overall_score"] = score_breakdown.overall
            results.append(res_copy)

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return results


matching_engine = MatchingEngine()
