import pytest
from app.services.matcher import matching_engine
from app.models.schemas import SkillAnalysis

def test_compute_experience_score_exact_match():
    score = matching_engine.compute_experience_score(5.0, 5.0)
    assert score == 100.0

def test_compute_experience_score_slightly_over():
    score = matching_engine.compute_experience_score(6.0, 5.0)
    assert score == 100.0

def test_compute_experience_score_overqualified():
    score = matching_engine.compute_experience_score(15.0, 5.0)
    assert score < 100.0

def test_compute_experience_score_under_qualified():
    score = matching_engine.compute_experience_score(2.0, 5.0)
    assert score < 100.0

def test_compute_education_score_exact():
    score = matching_engine.compute_education_score("Bachelors", "Bachelors")
    assert score == 100.0

def test_compute_education_score_exceeds():
    score = matching_engine.compute_education_score("Masters", "Bachelors")
    assert score == 100.0

def test_compute_education_score_below():
    score = matching_engine.compute_education_score("High School", "Bachelors")
    assert score < 100.0

def test_compute_education_score_unknown_requirement():
    score = matching_engine.compute_education_score("Bachelors", "Unknown")
    assert score == 100.0

def test_compute_skill_score_with_bonus():
    analysis = SkillAnalysis(
        matched=["python", "java"],
        missing=["docker"],
        bonus=["kubernetes"],
        match_percentage=66.67,
        preferred_skills_match_count=1
    )
    score = matching_engine.compute_skill_score(analysis)
    # 66.67 + 5 bonus points from preferred/extra = 71.67
    assert score > 66.67

def test_compute_skill_score_full_match():
    analysis = SkillAnalysis(
        matched=["python", "java"],
        missing=[],
        bonus=[],
        match_percentage=100.0
    )
    score = matching_engine.compute_skill_score(analysis)
    assert score == 100.0
