import streamlit as st
import requests

from frontend.styles.theme import apply_custom_theme
from frontend.utils import init_session_state, get_api_base_url, get_api_headers

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Career System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Styled Theme & Initialize State
apply_custom_theme()
init_session_state()

api_url = get_api_base_url()
headers = get_api_headers()

# Sidebar branding & info
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 24px;">
        <h2 class="gradient-text" style="font-size: 1.8rem; margin-bottom: 0;">AI Resume Screener</h2>
        <p style="color: #94a3b8; font-size: 0.85rem;">Intelligent Talent & Candidate Screening</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 System Status")

# Fetch API health dynamically
try:
    health_resp = requests.get(f"{api_url}/health", headers=headers, timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        models = health_data.get("models_loaded", {})

        st.sidebar.markdown("🟢 **API Status**: Online")
        st.sidebar.markdown(f"• **Classifier**: {'🟢 Loaded' if models.get('classifier') else '🟡 Missing'}")
        st.sidebar.markdown(f"• **Vector DB**: {'🟢 Active' if models.get('faiss_index') else '🟡 Inactive'}")
        st.sidebar.markdown(f"• **Database**: {'🟢 Connected' if models.get('database') else '🔴 Error'}")
    else:
        st.sidebar.markdown(f"🔴 **API Status**: HTTP {health_resp.status_code}")
except Exception:
    st.sidebar.markdown("🔴 **API Status**: Offline (Backend not reachable)")

st.sidebar.markdown("---")

# Navigation helper links
st.sidebar.markdown("### 🧭 Quick Guide")
st.sidebar.markdown(
    """
    1. **📊 Dashboard**: Upload a resume (PDF/DOCX/TXT) and match with a job description.
    2. **💬 Chatbot**: Ask questions about the candidate's profile, interview questions, or skill gaps.
    3. **📈 Analytics**: Review historical screening distributions and candidate metrics.
    4. **⚙️ Settings**: Adjust 4-dimension scoring weights (must sum to 100%) and LLM settings.
    """
)

# Landing Main View
st.markdown('<h1 class="gradient-text">AI Resume Screening & Career System</h1>', unsafe_allow_html=True)
st.markdown(
    """
    Welcome to an end-to-end Resume Screening and Explainable AI feedback system.
    Upload resumes (PDF, DOCX, TXT), evaluate multi-dimension candidate compatibility 
    (Semantic similarity 40%, Skill match 30%, Experience 15%, Education 15%), 
    explore SHAP-style Explainable AI attribution, and receive personalized career coaching via RAG.
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
                Upload candidate resumes in PDF, DOCX, or plain text. The system extracts NLP skills,
                predicts likely job roles using trained XGBoost models, and scores four weighted dimensions:
                semantic alignment, skill overlap, experience fit, and education.
            </p>
            <p style="font-size: 0.9rem; color: #94a3b8;">➡️ Navigate to <b>Dashboard</b> in the sidebar to begin.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card animate-fade-in">
            <h3 style="margin-top: 0; color: #c084fc;">💬 Contextual Career & Recruiter Chatbot</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Converse with an AI assistant grounded in the candidate's resume and job requirements.
                Generate tailored interview questions, uncover skill gaps, and explore concrete learning
                resources without hallucinations.
            </p>
            <p style="font-size: 0.9rem; color: #94a3b8;">➡️ Navigate to <b>Chatbot</b> in the sidebar to chat.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
