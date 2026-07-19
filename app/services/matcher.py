import math
import numpy as np
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
        res_emb = self.embedding_engine.encode_single(resume_text)
        jd_emb = self.embedding_engine.encode_single(jd_text)
        
        res_norm = res_emb / (np.linalg.norm(res_emb) or 1)
        jd_norm = jd_emb / (np.linalg.norm(jd_emb) or 1)
        
        similarity = float(np.dot(res_norm, jd_norm))
        score = max(0, similarity) * 100
        return clamp(score, 0, 100)

    def compute_skill_score(self, skill_analysis: SkillAnalysis) -> float:
        base_score = getattr(skill_analysis, 'match_percentage', 0.0)
        bonus = getattr(skill_analysis, 'preferred_skills_match_count', 0) * 5
        return clamp(base_score + bonus, 0, 100)

    def compute_experience_score(self, candidate_years: float, required_years: float) -> float:
        if candidate_years < required_years:
            gap = required_years - candidate_years
            penalty = 100 * math.exp(-0.5 * ((gap / max(required_years, 1))**2))
            return clamp(penalty, 0, 100)
        else:
            over = candidate_years - required_years
            if over <= 3:
                return 100.0
            else:
                score = 100.0 - 5 * (over - 3)
                return clamp(score, 0, 100)

    def compute_education_score(self, candidate_edu: str, required_edu: str) -> float:
        edu_order = {
            'high school': 1,
            'associate': 2,
            'bachelors': 3,
            'masters': 4,
            'phd': 5
        }
        
        cand_level = edu_order.get(str(candidate_edu).lower(), 0)
        req_level = edu_order.get(str(required_edu).lower(), 0)
        
        if req_level == 0:
            return 100.0
            
        if cand_level == 0:
            return 20.0
            
        diff = cand_level - req_level
        if diff == 0:
            return 100.0
        elif diff > 0:
            return 100.0
        elif diff == -1:
            return 70.0
        elif diff == -2:
            return 40.0
        else:
            return 20.0

    def compute_overall_score(self, resume_text: str, jd_text: str, skill_analysis: SkillAnalysis, 
                              candidate_exp: float, required_exp: float, 
                              candidate_edu: str, required_edu: str) -> ScoreBreakdown:
                              
        semantic = self.compute_semantic_score(resume_text, jd_text)
        skills = self.compute_skill_score(skill_analysis)
        experience = self.compute_experience_score(candidate_exp, required_exp)
        education = self.compute_education_score(candidate_edu, required_edu)
        
        w_sem = getattr(settings, 'semantic_weight', 0.25)
        w_skill = getattr(settings, 'skill_weight', 0.25)
        w_exp = getattr(settings, 'experience_weight', 0.25)
        w_edu = getattr(settings, 'education_weight', 0.25)
        
        overall = (semantic * w_sem) + (skills * w_skill) + (experience * w_exp) + (education * w_edu)
        
        return ScoreBreakdown(
            overall=overall,
            semantic=semantic,
            skill=skills,
            experience=experience,
            education=education
        )

    def rank_candidates(self, resumes: list[dict], job_description: dict) -> list[dict]:
        jd_text = job_description.get('text', '')
        req_exp = job_description.get('required_experience', 0.0)
        req_edu = job_description.get('required_education', 'bachelors')
        
        results = []
        for res in resumes:
            res_text = res.get('text', '')
            skill_analysis = res.get('skill_analysis', SkillAnalysis())
            cand_exp = res.get('experience_years', 0.0)
            cand_edu = res.get('education_level', '')
            
            score_breakdown = self.compute_overall_score(
                res_text, jd_text, skill_analysis, cand_exp, req_exp, cand_edu, req_edu
            )
            
            res['score_breakdown'] = score_breakdown
            res['overall_score'] = score_breakdown.overall
            results.append(res)
            
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        return results

matching_engine = MatchingEngine()
