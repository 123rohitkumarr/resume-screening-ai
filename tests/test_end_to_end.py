import io
import json
import pytest
from fastapi.testclient import TestClient
import fitz
from docx import Document

from app.main import app
from app.services.classifier import classifier
from app.services.matcher import matching_engine
from app.services.preprocessor import (
    extract_email,
    extract_phone,
    extract_candidate_name,
    extract_experience_years,
    extract_education,
)
from app.services.embeddings import embedding_engine
from app.models.schemas import SkillAnalysis

client = TestClient(app)


# ── Health & System Tests ─────────────────────────────────────

def test_health_check_complete():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["classifier"] is True
    assert data["models_loaded"]["tfidf"] is True
    assert data["models_loaded"]["faiss_index"] is True
    assert data["models_loaded"]["database"] is True


# ── Settings API Tests ────────────────────────────────────────

def test_settings_get_and_update():
    # 1. Fetch settings
    res = client.get("/api/settings/")
    assert res.status_code == 200
    cfg = res.json()
    assert "semantic_weight" in cfg

    # 2. Update with valid sum (0.40 + 0.30 + 0.15 + 0.15 = 1.0)
    update_payload = {
        "semantic_weight": 0.40,
        "skill_weight": 0.30,
        "experience_weight": 0.15,
        "education_weight": 0.15,
        "llm_provider": "gemini"
    }
    res_up = client.put("/api/settings/", json=update_payload)
    assert res_up.status_code == 200
    assert res_up.json()["semantic_weight"] == 0.40

    # 3. Reject invalid sum (e.g. 0.80 != 1.0)
    bad_payload = {
        "semantic_weight": 0.50,
        "skill_weight": 0.10,
        "experience_weight": 0.10,
        "education_weight": 0.10
    }
    res_bad = client.put("/api/settings/", json=bad_payload)
    assert res_bad.status_code == 400


# ── Candidate Information Extraction Tests ────────────────────

def test_candidate_extraction_email_and_phone():
    text = (
        "Sarah Jenkins\n"
        "Email: sarah.jenkins@example.com | Phone: (555) 234-5678\n"
        "Senior Python Engineer with 6 years of experience in FastAPI and Docker."
    )
    email = extract_email(text)
    phone = extract_phone(text)
    name = extract_candidate_name(text)
    exp = extract_experience_years(text)

    assert email == "sarah.jenkins@example.com"
    assert phone is not None and "555" in phone
    assert name == "Sarah Jenkins"
    assert exp == 6.0


def test_zero_experience_vs_missing_experience():
    # Explicit zero experience
    zero_text = "Recent Computer Science graduate. Fresher with 0 years of experience seeking entry-level software role."
    assert extract_experience_years(zero_text) == 0.0

    # Unmentioned / missing experience
    missing_text = "Proficient in Python and JavaScript. Worked on personal projects."
    assert extract_experience_years(missing_text) is None


# ── XGBoost Role Classifier Tests ─────────────────────────────

def test_role_classifier_prediction():
    skills = ["python", "machine learning", "pandas", "numpy", "scikit-learn"]
    role, conf = classifier.predict(skills, education="Masters", experience=4.0)
    assert isinstance(role, str)
    assert role != "Unknown"
    assert conf > 0.0

    top_k = classifier.predict_top_k(skills, education="Masters", experience=4.0, k=3)
    assert len(top_k) == 3
    assert all(isinstance(r, str) and c >= 0.0 for r, c in top_k)


# ── End-to-End Screening via JSON ─────────────────────────────

def test_screen_resume_text_endpoint():
    payload = {
        "name": "Alex Mercer",
        "resume_text": (
            "Alex Mercer\n"
            "Email: alex.mercer@example.com\n"
            "Education: BS in Computer Science\n"
            "Experience: 4 years of software development\n"
            "Skills: Python, FastAPI, Docker, PostgreSQL, Redis, Git, Unit Testing"
        ),
        "jd_title": "Backend Python Developer",
        "jd_description": "We need a Backend Developer proficient in Python, FastAPI, Docker, and PostgreSQL.",
        "required_skills": ["python", "fastapi", "docker", "postgresql"],
        "preferred_skills": ["redis", "git"],
        "min_experience": 3.0,
        "education_required": "Bachelors",
        "save_to_db": True
    }
    response = client.post("/api/screening/", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["candidate_name"] == "Alex Mercer"
    assert data["candidate_email"] == "alex.mercer@example.com"
    assert data["scores"]["overall"] >= 70.0
    assert data["scores"]["skill"] >= 80.0
    assert "python" in data["skill_analysis"]["matched"]
    assert len(data["explanation"]) == 4
    assert len(data["suggestions"]) > 0


# ── Screening via Uploaded Files (PDF, DOCX, TXT) ─────────────

def test_screen_file_pdf_synthetic():
    # Build a real in-memory synthetic PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 72),
        "Elena Rostova\n"
        "elena.rostova@example.com\n"
        "Education: Masters in Artificial Intelligence\n"
        "Experience: 5 years of machine learning engineering\n"
        "Skills: Python, PyTorch, TensorFlow, Scikit-Learn, Docker, SQL\n"
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"file": ("elena_resume.pdf", pdf_bytes, "application/pdf")}
    data = {
        "name": "Elena Rostova",
        "jd_title": "AI Engineer",
        "jd_description": "Seeking AI Engineer with PyTorch, TensorFlow, and Python expertise.",
        "required_skills_json": json.dumps(["python", "pytorch", "tensorflow"]),
        "preferred_skills_json": json.dumps(["docker"]),
        "min_experience": 3.0,
        "education_required": "Masters",
        "save_to_db": True
    }
    response = client.post("/api/screening/file", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["candidate_name"] == "Elena Rostova"
    assert res_json["candidate_email"] == "elena.rostova@example.com"
    assert res_json["scores"]["overall"] >= 70.0


def test_screen_file_docx_synthetic():
    # Build a real in-memory synthetic DOCX with paragraphs and tables
    doc = Document()
    doc.add_heading("David Kim", level=1)
    doc.add_paragraph("david.kim@example.com | 555-432-1098")
    doc.add_paragraph("Education: Bachelors in Software Engineering")
    doc.add_paragraph("Experience: 3 years building web apps")
    # Table of skills
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Frontend"
    table.rows[0].cells[1].text = "React, TypeScript, JavaScript, HTML, CSS"

    docx_buf = io.BytesIO()
    doc.save(docx_buf)
    docx_bytes = docx_buf.getvalue()

    files = {"file": ("david_resume.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    data = {
        "name": "David Kim",
        "jd_title": "Frontend React Engineer",
        "jd_description": "Frontend developer proficient in React, TypeScript, and HTML.",
        "required_skills_json": json.dumps(["react", "typescript", "javascript"]),
        "min_experience": 2.0,
        "education_required": "Bachelors",
        "save_to_db": True
    }
    response = client.post("/api/screening/file", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "react" in res_json["skill_analysis"]["matched"]
    assert res_json["scores"]["overall"] >= 60.0


def test_screen_file_txt():
    txt_content = (
        "Marcus Vance\n"
        "marcus.vance@example.com\n"
        "High School Diploma\n"
        "Experience: 1 year\n"
        "Skills: Python, Git"
    ).encode("utf-8")

    files = {"file": ("marcus.txt", txt_content, "text/plain")}
    data = {
        "name": "Marcus Vance",
        "jd_title": "Junior Python Dev",
        "jd_description": "Entry level Python dev with Git knowledge.",
        "required_skills_json": json.dumps(["python", "git"]),
        "min_experience": 1.0,
        "education_required": "High School",
        "save_to_db": True
    }
    response = client.post("/api/screening/file", files=files, data=data)
    assert response.status_code == 200


def test_empty_file_rejected():
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    data = {
        "name": "Empty Test",
        "jd_title": "Any",
        "jd_description": "Any description",
        "required_skills_json": json.dumps(["python"]),
    }
    response = client.post("/api/screening/file", files=files, data=data)
    assert response.status_code == 400


def test_invalid_extension_rejected():
    files = {"file": ("malicious.exe", b"binary_content", "application/octet-stream")}
    data = {
        "name": "Bad File",
        "jd_title": "Any",
        "jd_description": "Any description",
        "required_skills_json": json.dumps(["python"]),
    }
    response = client.post("/api/screening/file", files=files, data=data)
    assert response.status_code == 400


# ── AI Feedback Generation Endpoint ───────────────────────────

def test_feedback_endpoint():
    # Build screening result
    payload = {
        "name": "Feedback Test",
        "resume_text": "Skills: Python, SQL. Experience: 3 years. Education: Bachelors.",
        "jd_title": "Backend Dev",
        "jd_description": "Backend engineer with Python, SQL, Docker, Kubernetes.",
        "required_skills": ["python", "sql", "docker", "kubernetes"],
        "save_to_db": False
    }
    screen_resp = client.post("/api/screening/", json=payload)
    assert screen_resp.status_code == 200
    screening_result = screen_resp.json()

    fb_resp = client.post("/api/feedback/", json={"screening_result": screening_result})
    assert fb_resp.status_code == 200
    fb_data = fb_resp.json()
    assert "overall_feedback" in fb_data
    assert len(fb_data["skill_recommendations"]) > 0
    assert len(fb_data["priority_actions"]) > 0


# ── RAG Chatbot Endpoint ──────────────────────────────────────

def test_chatbot_endpoint_and_history():
    chat_payload = {
        "message": "What skills should I prioritize to improve my match score?",
        "session_id": "test-session-e2e",
        "screening_context": {
            "candidate_name": "Alex Mercer",
            "job_title": "Backend Engineer",
            "overall_score": 75.0,
            "matched_skills": ["python", "fastapi"],
            "missing_skills": ["docker", "kubernetes"],
            "predicted_role": "Backend Developer",
            "summary": "Solid foundation with missing cloud infrastructure skills."
        }
    }
    response = client.post("/api/chat/", json=chat_payload)
    assert response.status_code == 200
    chat_data = response.json()
    assert "response" in chat_data
    assert len(chat_data["response"]) > 20
    assert chat_data["session_id"] == "test-session-e2e"

    # Fetch history
    hist_resp = client.get("/api/chat/history/test-session-e2e")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) >= 2  # user + assistant

    # Clear history
    del_resp = client.delete("/api/chat/history/test-session-e2e")
    assert del_resp.status_code == 204


# ── Candidate Ranking Engine ──────────────────────────────────

def test_candidate_ranking():
    resumes = [
        {"name": "Low Match", "text": "Basic skills only.", "skills": ["html"], "experience_years": 0.5, "education_level": "High School"},
        {"name": "High Match", "text": "Senior Python and FastAPI specialist with Docker.", "skills": ["python", "fastapi", "docker"], "experience_years": 5.0, "education_level": "Masters"}
    ]
    job_desc = {
        "text": "Senior Python and FastAPI engineer with Docker.",
        "required_skills": ["python", "fastapi", "docker"],
        "required_experience": 4.0,
        "required_education": "Masters"
    }
    ranked = matching_engine.rank_candidates(resumes, job_desc)
    assert len(ranked) == 2
    assert ranked[0]["name"] == "High Match"
    assert ranked[0]["overall_score"] > ranked[1]["overall_score"]
