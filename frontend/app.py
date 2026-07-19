import streamlit as st
import os
import requests

from frontend.styles.theme import apply_custom_theme
from app.config import settings

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Feedback System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Styled Theme
apply_custom_theme()

# Set up defaults
if "api_url" not in st.session_state:
    st.session_state.api_url = f"http://localhost:{settings.api_port}"
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "screening_result" not in st.session_state:
    st.session_state.screening_result = None
if "last_resume_name" not in st.session_state:
    st.session_state.last_resume_name = None
if "last_jd_title" not in st.session_state:
    st.session_state.last_jd_title = None

# Sidebar branding & info
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 24px;">
        <h2 class="gradient-text" style="font-size: 1.8rem; margin-bottom: 0;">Antigravity AI</h2>
        <p style="color: #94a3b8; font-size: 0.85rem;">Resume Screener & Career Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 System Status")

# Fetch API health dynamically
try:
    health_resp = requests.get(f"{st.session_state.api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        models = health_data.get("models_loaded", {})
        
        st.sidebar.markdown("🟢 **API Status**: Online")
        st.sidebar.markdown(f"• **Classifier**: {'🟢 Loaded' if models.get('classifier') else '🔴 Missing'}")
        st.sidebar.markdown(f"• **Vector DB**: {'🟢 Active' if models.get('faiss_index') else '🔴 Inactive'}")
    else:
        st.sidebar.markdown("🔴 **API Status**: Offline")
except Exception:
    st.sidebar.markdown("🔴 **API Status**: Offline (Cannot reach API)")

st.sidebar.markdown("---")

# Navigation helper links
st.sidebar.markdown("### 🧭 Quick Guide")
st.sidebar.markdown(
    """
    1. **Go to 📊 Dashboard** to upload a resume file and copy-paste a job description.
    2. **View match ratings**, skill gaps, and explanations.
    3. **Go to 💬 Chatbot** to ask questions about your profile or interview suggestions.
    """
)

# Landing Main View
st.markdown('<h1 class="gradient-text">AI Resume Screening & Career System</h1>', unsafe_allow_html=True)
st.markdown(
    """
    Welcome to a complete, production-ready Resume Screening, Profiling, and Explainable AI feedback assistant.
    This system parses files, calculates compatibility scores using ML, Deep Learning, and NLP semantic similarity,
    visualizes feature attribution (XAI), and supports interactive career coaching using RAG.
    """
)

# Render interactive visual blocks
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="glass-card animate-fade-in">
            <h3 style="margin-top: 0; color: #38bdf8;">📊 Dynamic Compatibility Screening</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Drop candidate PDF or DOCX resumes and job descriptions. Our engine evaluates semantic similarity,
                fuzzy matching skills, experience thresholds, and education levels to output a weighted compatibility score.
            </p>
            <p style="font-size: 0.9rem; color: #94a3b8;">➡️ Click <b>Dashboard</b> in the sidebar to begin.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card animate-fade-in">
            <h3 style="margin-top: 0; color: #c084fc;">💬 Contextual RAG Chatbot</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Chat with an intelligent assistant grounded in your resume profile. Get detailed interview preparation tips,
                actionable skill development advice, and course recommendations tailored to matching job criteria.
            </p>
            <p style="font-size: 0.9rem; color: #94a3b8;">➡️ Click <b>Chatbot</b> in the sidebar to ask questions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
