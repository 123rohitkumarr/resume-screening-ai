import pytest
from app.services.preprocessor import extract_experience_years, extract_education, tokenize_skills, preprocess_text

def test_extract_experience_years_basic():
    assert extract_experience_years("5 years of experience") == 5.0

def test_extract_experience_years_plus():
    assert extract_experience_years("3+ years") == 3.0

def test_extract_experience_years_range():
    assert extract_experience_years("2-4 years") == 4.0

def test_extract_experience_years_none():
    assert extract_experience_years("no mention of time") is None

def test_extract_education_phd():
    assert extract_education("PhD in Computer Science") == "PhD"

def test_extract_education_masters():
    assert extract_education("Masters in Data Science") == "Masters"

def test_extract_education_bachelors():
    assert extract_education("BS in Engineering") == "Bachelors"

def test_extract_education_none():
    assert extract_education("self-taught developer") is None

def test_tokenize_skills():
    assert tokenize_skills("python, java, sql") == ["python", "java", "sql"]

def test_preprocess_text():
    text = preprocess_text("Hello World! This is a TEST.")
    # Stopwords are removed, lowercase, lemmatized
    assert "hello" in text
    assert "world" in text
    assert "test" in text
    assert "this" not in text
    assert "is" not in text
