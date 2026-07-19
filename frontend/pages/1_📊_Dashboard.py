import streamlit as st
import requests
import json

from frontend.styles.theme import apply_custom_theme
from frontend.components.score_card import render_score_gauge
from frontend.components.skill_gap import render_radar_breakdown, render_skills_bar_chart
from frontend.components.explanation import render_xai_explanation

apply_custom_theme()

st.markdown('<h1 class="gradient-text">📊 Resume Screening Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Screen candidate resumes against target job specifications using NLP semantic alignment, fuzzy skill mapping, and machine learning models.")

# Preset Job Descriptions to simplify testing
JD_PRESETS = {
    "Custom": {
        "title": "",
        "description": "",
        "required_skills": "",
        "preferred_skills": "",
        "min_experience": 0.0,
        "education_required": "Bachelors"
    },
    "Data Scientist": {
        "title": "Senior Data Scientist",
        "description": "We are seeking a senior data scientist to build machine learning models, neural networks, and analyze large datasets. You will write clean Python code using pandas, numpy, and deploy models using Docker and cloud services.",
        "required_skills": "python, machine learning, sql, pandas, numpy, scikit-learn",
        "preferred_skills": "deep learning, tensorflow, docker, aws",
        "min_experience": 5.0,
        "education_required": "Masters"
    },
    "Web Developer": {
        "title": "Frontend React Developer",
        "description": "Looking for a Frontend Developer with deep React experience. Build responsive layouts, design reusable UI elements using Tailwind CSS, and write modern ES6+ Javascript/Typescript.",
        "required_skills": "javascript, html, css, react, typescript, git",
        "preferred_skills": "webpack, responsive design, tailwind css",
        "min_experience": 2.0,
        "education_required": "Bachelors"
    },
    "DevOps Engineer": {
        "title": "DevOps Architect",
        "description": "Build high availability CI/CD pipelines, configure Terraform infrastructure as code on AWS, manage Kubernetes clusters, and Docker containers for microservices architecture.",
        "required_skills": "docker, kubernetes, aws, terraform, ci/cd, git",
        "preferred_skills": "jenkins, linux, ansible, bash",
        "min_experience": 4.0,
        "education_required": "Bachelors"
    }
}

# Input Form
with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Step 1: Candidate Resume")
        cand_name = st.text_input("Candidate Name", value="Jane Doe")
        
        upload_mode = st.radio("Resume Input Mode", ["File Upload", "Paste Raw Text"])
        
        file_bytes = None
        file_name = None
        raw_resume_text = ""
        
        if upload_mode == "File Upload":
            uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
            if uploaded_file:
                file_bytes = uploaded_file.read()
                file_name = uploaded_file.name
        else:
            raw_resume_text = st.text_area("Paste Resume Content", height=230, placeholder="Paste clean resume details here...")

    with col2:
        st.markdown("### 💼 Step 2: Job Description")
        preset_choice = st.selectbox("Load Job Preset", list(JD_PRESETS.keys()))
        preset_data = JD_PRESETS[preset_choice]
        
        jd_title = st.text_input("Job Title", value=preset_data["title"])
        jd_desc = st.text_area("Job Description Details", value=preset_data["description"], height=100)
        
        req_skills_input = st.text_input("Required Skills (comma-separated)", value=preset_data["required_skills"])
        pref_skills_input = st.text_input("Preferred Skills (comma-separated)", value=preset_data["preferred_skills"])
        
        exp_col, edu_col = st.columns(2)
        with exp_col:
            min_exp = st.number_input("Min Experience (years)", min_value=0.0, max_value=20.0, value=float(preset_data["min_experience"]), step=1.0)
        with edu_col:
            edu_req = st.selectbox("Required Education", ["None", "High School", "Associate", "Bachelors", "Masters", "PhD"], index=["None", "High School", "Associate", "Bachelors", "Masters", "PhD"].index(preset_data["education_required"]))

# Trigger Screening
if st.button("🚀 Analyze Compatibility"):
    api_url = st.session_state.api_url
    
    # Validation checks
    if upload_mode == "File Upload" and not uploaded_file:
        st.error("Please upload a resume file.")
    elif upload_mode == "Paste Raw Text" and not raw_resume_text.strip():
        st.error("Please paste resume text.")
    elif not jd_desc.strip():
        st.error("Please enter a job description.")
    elif not req_skills_input.strip():
        st.error("Please enter required skills.")
    else:
        with st.spinner("Processing pipeline: parsing, extracting NLP skills, executing ML classifier & scoring..."):
            try:
                # Prepare payload
                req_skills = [s.strip() for s in req_skills_input.split(",") if s.strip()]
                pref_skills = [s.strip() for s in pref_skills_input.split(",") if s.strip()]
                
                # File upload path
                if upload_mode == "File Upload":
                    files = {
                        "file": (file_name, file_bytes)
                    }
                    data = {
                        "name": cand_name,
                        "jd_title": jd_title,
                        "jd_description": jd_desc,
                        "required_skills_json": json.dumps(req_skills),
                        "preferred_skills_json": json.dumps(pref_skills),
                        "min_experience": min_exp,
                        "education_required": edu_req,
                        "save_to_db": True
                    }
                    resp = requests.post(f"{api_url}/api/screening/file", files=files, data=data)
                
                # Raw text path
                else:
                    payload = {
                        "name": cand_name,
                        "resume_text": raw_resume_text,
                        "jd_title": jd_title,
                        "jd_description": jd_desc,
                        "required_skills": req_skills,
                        "preferred_skills": pref_skills,
                        "min_experience": min_exp,
                        "education_required": edu_req,
                        "save_to_db": True
                    }
                    resp = requests.post(f"{api_url}/api/screening/", json=payload)

                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.screening_result = result
                    st.session_state.last_resume_name = cand_name
                    st.session_state.last_jd_title = jd_title
                    st.success("Screening completed successfully!")
                else:
                    st.error(f"Error during screening: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")

# Display Results
if st.session_state.screening_result:
    result = st.session_state.screening_result
    scores = result.get("scores", {})
    skill_analysis = result.get("skill_analysis", {})
    predicted_role = result.get("predicted_role", "Unknown")
    explanation = result.get("explanation", [])
    suggestions = result.get("suggestions", [])
    summary = result.get("summary", "")

    st.markdown("---")
    st.markdown(f"## 📋 Screening Results: **{st.session_state.last_resume_name}** ↔ **{st.session_state.last_jd_title}**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Gauge Metric
        render_score_gauge(scores.get("overall", 0.0), "Overall Compatibility")
        
        # Details metadata
        st.markdown(
            f"""
            <div class="glass-card">
                <h4>🎯 Profile Verdict</h4>
                <p><b>Predicted Professional Role</b>: <span style="color:#818cf8; font-weight:600;">{predicted_role}</span></p>
                <p style="font-size:0.95rem; line-height:1.5; color:#cbd5e1;">{summary}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Radar breakdown chart
        st.markdown("### 📊 Metrics Breakdown")
        render_radar_breakdown(scores)

    # Skills Analysis Cards
    st.markdown("### 🛡️ Skills Analysis")
    sc1, sc2 = st.columns([2, 1])
    
    with sc1:
        matched = skill_analysis.get("matched", [])
        missing = skill_analysis.get("missing", [])
        bonus = skill_analysis.get("bonus", [])
        
        st.markdown("**Matched Skills**")
        if matched:
            st.markdown(" ".join([f'<span class="badge badge-matched">{s}</span>' for s in matched]), unsafe_allow_html=True)
        else:
            st.markdown("None detected.")
            
        st.markdown("<br>**Missing Required Skills**", unsafe_allow_html=True)
        if missing:
            st.markdown(" ".join([f'<span class="badge badge-missing">{s}</span>' for s in missing]), unsafe_allow_html=True)
        else:
            st.markdown("None!")
            
        if bonus:
            st.markdown("<br>**Extra/Bonus Skills**", unsafe_allow_html=True)
            st.markdown(" ".join([f'<span class="badge badge-bonus">{s}</span>' for s in bonus]), unsafe_allow_html=True)
            
    with sc2:
        render_skills_bar_chart(matched, missing)

    # Explanation and recommendations
    render_xai_explanation(explanation)

    # Recommendations panel
    st.markdown("### 💡 Recommended Improvement Roadmap")
    for sug in suggestions:
        st.markdown(f"• {sug}")
        
    st.markdown("---")
    st.markdown("💬 **Tip**: Go to the **Chatbot** in the sidebar to ask questions about how this candidate can prepare or bridge these specific skill gaps.")
