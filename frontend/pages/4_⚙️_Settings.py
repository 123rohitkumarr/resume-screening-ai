import streamlit as st
import requests

from frontend.styles.theme import apply_custom_theme
from frontend.utils import init_session_state, get_api_base_url, get_api_headers
from app.config import settings
from app.services.embeddings import embedding_engine

apply_custom_theme()
init_session_state()

st.markdown('<h1 class="gradient-text">⚙️ Configuration & Weights Settings</h1>', unsafe_allow_html=True)
st.markdown("Customize model scoring dimension weights and LLM providers. Changes synchronize directly with the FastAPI backend engine.")

api_url = get_api_base_url()
headers = get_api_headers()

# Fetch active configuration from FastAPI backend
curr_sem = settings.semantic_weight
curr_skill = settings.skill_weight
curr_exp = settings.experience_weight
curr_edu = settings.education_weight
curr_prov = settings.llm_provider.value if hasattr(settings.llm_provider, "value") else str(settings.llm_provider)

try:
    resp = requests.get(f"{api_url}/api/settings/", headers=headers, timeout=2)
    if resp.status_code == 200:
        backend_cfg = resp.json()
        curr_sem = backend_cfg.get("semantic_weight", curr_sem)
        curr_skill = backend_cfg.get("skill_weight", curr_skill)
        curr_exp = backend_cfg.get("experience_weight", curr_exp)
        curr_edu = backend_cfg.get("education_weight", curr_edu)
        curr_prov = backend_cfg.get("llm_provider", curr_prov)
except Exception:
    pass

# Configuration form
with st.form("settings_form"):
    st.markdown("### 🎛️ Scoring Dimension Weights")
    st.markdown("Adjust the importance of each metric. **The sum must equal 1.0 (100%).**")

    col1, col2 = st.columns(2)
    with col1:
        sem_w = st.slider("Semantic Alignment Weight", 0.0, 1.0, float(curr_sem), 0.05)
        skill_w = st.slider("Skills Match Weight", 0.0, 1.0, float(curr_skill), 0.05)
    with col2:
        exp_w = st.slider("Experience Match Weight", 0.0, 1.0, float(curr_exp), 0.05)
        edu_w = st.slider("Education Fit Weight", 0.0, 1.0, float(curr_edu), 0.05)

    current_sum = sem_w + skill_w + exp_w + edu_w
    if abs(current_sum - 1.0) > 0.01:
        st.warning(f"⚠️ Current weight sum is **{current_sum:.2f}**. Must sum to **1.0** to save.")
    else:
        st.success("✅ Weights correctly sum to 1.0 (100%).")

    st.markdown("---")
    st.markdown("### 🧠 LLM Configurations")
    prov_options = ["gemini", "ollama", "openai"]
    prov_index = prov_options.index(curr_prov) if curr_prov in prov_options else 0
    llm_prov = st.selectbox("LLM Provider", prov_options, index=prov_index)

    g_model = st.text_input("Gemini Model", value=settings.gemini_model)
    o_model = st.text_input("Ollama Model", value=settings.ollama_model)
    o_url = st.text_input("Ollama Endpoint URL", value=settings.ollama_url)
    op_model = st.text_input("OpenAI Model", value=settings.openai_model)

    submitted = st.form_submit_button("💾 Save Configurations")

if submitted:
    total_w = sem_w + skill_w + exp_w + edu_w
    if abs(total_w - 1.0) > 0.01:
        st.error(f"Error: Weights must sum to 1.0. Current sum is: {total_w:.2f}")
    else:
        payload = {
            "semantic_weight": round(sem_w, 2),
            "skill_weight": round(skill_w, 2),
            "experience_weight": round(exp_w, 2),
            "education_weight": round(edu_w, 2),
            "llm_provider": llm_prov,
            "gemini_model": g_model,
            "ollama_model": o_model,
            "ollama_url": o_url,
            "openai_model": op_model,
        }
        try:
            resp = requests.put(f"{api_url}/api/settings/", json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                st.success("Configurations successfully synchronized with backend engine!")
            else:
                st.error(f"Failed to update backend settings: {resp.text}")
        except Exception as e:
            # Fallback to local settings
            settings.semantic_weight = round(sem_w, 2)
            settings.skill_weight = round(skill_w, 2)
            settings.experience_weight = round(exp_w, 2)
            settings.education_weight = round(edu_w, 2)
            st.warning(f"Could not reach backend API ({e}). Applied locally in frontend.")

# Model Engine Statistics
st.markdown("---")
st.markdown("### 📊 Vector Knowledge Base Status")
try:
    if embedding_engine.index is None:
        embedding_engine.load_index()
    if embedding_engine.index is not None:
        st.write(f"• **FAISS Vector Dimension**: `{embedding_engine.index.d}`")
        st.write(f"• **Indexed Vectors**: `{embedding_engine.index_size}` Chunks")
    else:
        st.info("FAISS vector index is starting or empty.")
except Exception as e:
    st.error(f"Error reading vector database status: {str(e)}")
