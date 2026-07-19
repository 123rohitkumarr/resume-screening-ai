import pytest
from app.services.skill_extractor import extract_skills, resolve_alias, match_skills, categorize_skills, get_skill_importance

def test_extract_skills_basic():
    skills = extract_skills("I know python and javascript")
    assert "python" in skills
    assert "javascript" in skills

def test_resolve_alias():
    assert resolve_alias("js") == "javascript"
    assert resolve_alias("ml") == "machine learning"

def test_match_skills_full_match():
    analysis = match_skills(["python", "java"], ["python", "java"])
    assert "python" in analysis.matched
    assert "java" in analysis.matched
    assert len(analysis.missing) == 0
    assert analysis.match_percentage == 100.0

def test_match_skills_partial_match():
    analysis = match_skills(["python"], ["python", "java"])
    assert "python" in analysis.matched
    assert "java" in analysis.missing
    assert analysis.match_percentage == 50.0

def test_match_skills_with_preferred():
    analysis = match_skills(["python", "docker"], ["python"], preferred_skills=["docker"])
    assert "python" in analysis.matched
    assert "docker" in analysis.bonus

def test_categorize_skills():
    categories = categorize_skills(["python", "agile"])
    assert "python" in categories.get("programming_languages", [])
    assert "agile" in categories.get("soft_skills", [])

def test_get_skill_importance():
    assert get_skill_importance("python") == "high"
    assert get_skill_importance("communication") == "low"
