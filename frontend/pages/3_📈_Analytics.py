import streamlit as st
import pandas as pd
import plotly.express as px
import json

from frontend.styles.theme import apply_custom_theme
from frontend.utils import init_session_state
from app.models.database import SessionLocal, ScreeningRecord, ResumeRecord, JobDescriptionRecord

apply_custom_theme()
init_session_state()

st.markdown('<h1 class="gradient-text">📈 Screening & Candidate Analytics</h1>', unsafe_allow_html=True)
st.markdown("Monitor historical screenings, candidate distributions, and aggregate performance insights.")

# Query historical data
screenings = []
resumes = []
jds = []

try:
    db = SessionLocal()
    try:
        screenings = db.query(ScreeningRecord).all()
        resumes = db.query(ResumeRecord).all()
        jds = db.query(JobDescriptionRecord).all()
    finally:
        db.close()
except Exception as e:
    st.error(f"Error accessing database: {e}")

if not screenings:
    st.info("No historical screenings found in the database. Run resume evaluations on the Dashboard first!")
else:
    # Build DataFrames
    df_scr = pd.DataFrame([{
        "id": s.id,
        "resume_id": s.resume_id,
        "job_id": s.job_id,
        "overall": s.overall_score,
        "semantic": s.semantic_score,
        "skill": s.skill_score,
        "experience": s.experience_score,
        "education": s.education_score,
        "missing": json.loads(s.missing_skills or "[]"),
        "created_at": s.created_at
    } for s in screenings])

    df_res = pd.DataFrame([{
        "id": r.id,
        "name": r.name,
        "predicted_role": r.predicted_role or "Unknown",
        "experience": r.years_experience if r.years_experience is not None else 0.0,
        "education": r.education or "Unknown"
    } for r in resumes])

    # Join DataFrames
    df_merge = df_scr.merge(df_res, left_on="resume_id", right_on="id", how="inner")

    # Metrics Summary Row
    st.markdown("### 📊 High-level Aggregates")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Resumes Screened", len(resumes))
    with col2:
        st.metric("Average Match Score", f"{df_scr['overall'].mean():.1f}%")
    with col3:
        st.metric("Avg Years of Experience", f"{df_res['experience'].mean():.1f} Years")
    with col4:
        top_edu = df_res["education"].mode().iloc[0] if not df_res.empty else "N/A"
        st.metric("Top Education Tier", top_edu)

    # Double columns charts
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("### 🎭 Match Score Distribution")
        fig_dist = px.histogram(
            df_scr,
            x="overall",
            nbins=10,
            labels={"overall": "Overall Match Score (%)"},
            color_discrete_sequence=["#6366f1"]
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with ch2:
        st.markdown("### 🏷️ Predicted Candidate Roles")
        role_counts = df_res["predicted_role"].value_counts().reset_index()
        role_counts.columns = ["Role", "Count"]
        fig_pie = px.pie(
            role_counts,
            values="Count",
            names="Role",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Missing Skills Frequency Analysis
    st.markdown("### ⚠️ Common Missing Skills Across Candidates")
    all_missing = []
    for item in df_scr["missing"]:
        if isinstance(item, list):
            all_missing.extend(item)

    if all_missing:
        missing_counts = pd.Series(all_missing).value_counts().reset_index()
        missing_counts.columns = ["Skill", "Count"]

        fig_skills = px.bar(
            missing_counts.head(10),
            x="Count",
            y="Skill",
            orientation="h",
            labels={"Count": "Frequency", "Skill": "Missing Skill"},
            color_discrete_sequence=["#ef4444"]
        )
        fig_skills.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_skills, use_container_width=True)
    else:
        st.info("No missing skills recorded yet.")

    # Historical screenings log table
    st.markdown("### 📜 Screening Logs")
    if not df_merge.empty and "created_at" in df_merge.columns:
        log_df = df_merge[["name", "predicted_role", "experience", "overall", "created_at"]].sort_values(by="created_at", ascending=False)
        log_df.columns = ["Candidate Name", "Predicted Role", "Experience (Years)", "Compatibility Score (%)", "Screening Date"]
        st.dataframe(log_df, use_container_width=True)
