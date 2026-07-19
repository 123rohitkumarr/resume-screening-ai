import streamlit as st
import os

from frontend.styles.theme import apply_custom_theme
from app.config import settings
from app.services.embeddings import embedding_engine

apply_custom_theme()

st.markdown('<h1 class="gradient-text">⚙️ Configuration & Weights Settings</h1>', unsafe_allow_html=True)
st.markdown("Customize model hyperparameters, LLM providers, and individual scoring dimension weights.")

# Configuration form
with st.form("settings_form"):
    st.markdown("### 🎛️ Scoring Weights")
    st.markdown("Adjust the importance of each metric. Note that the sum **must equal 1.0** (100%).")
    
    col1, col2 = st.columns(2)
    with col1:
        sem_w = st.slider("Semantic Alignment Weight", 0.0, 1.0, settings.semantic_weight, 0.05)
        skill_w = st.slider("Skills Match Weight", 0.0, 1.0, settings.skill_weight, 0.05)
    with col2:
        exp_w = st.slider("Experience Match Weight", 0.0, 1.0, settings.experience_weight, 0.05)
        edu_w = st.slider("Education Fit Weight", 0.0, 1.0, settings.education_weight, 0.05)
        
    st.markdown("---")
    st.markdown("### 🧠 LLM Configurations")
    llm_prov = st.selectbox("LLM Provider", ["gemini", "ollama", "openai"], index=["gemini", "ollama", "openai"].index(settings.llm_provider))
    
    g_model = st.text_input("Gemini Model", value=settings.gemini_model)
    o_model = st.text_input("Ollama Model", value=settings.ollama_model)
    o_url = st.text_input("Ollama Endpoint URL", value=settings.ollama_url)
    op_model = st.text_input("OpenAI Model", value=settings.openai_model)
    
    st.markdown("---")
    st.markdown("### 📂 Paths Configuration")
    db_url = st.text_input("Database Connection URI", value=settings.database_url)
    faiss_path = st.text_input("FAISS Vector Index Path", value=settings.faiss_index_path)

    submitted = st.form_submit_with_ui_modifications("Save Configurations") if hasattr(st, "form_submit_with_ui_modifications") else st.form_submit_button("Save Configurations")

if submitted:
    total_w = sem_w + skill_w + exp_w + edu_w
    if not (0.99 <= total_w <= 1.01):
        st.error(f"Error: Weights must sum to 1.0. Current sum is: {total_w:.2f}")
    else:
        # Save configurations directly to global singleton state
        settings.semantic_weight = sem_w
        settings.skill_weight = skill_w
        settings.experience_weight = exp_w
        settings.education_weight = edu_w
        settings.llm_provider = llm_prov
        settings.gemini_model = g_model
        settings.ollama_model = o_model
        settings.ollama_url = o_url
        settings.openai_model = op_model
        settings.database_url = db_url
        settings.faiss_index_path = faiss_path
        
        st.success("Configurations updated successfully for this session!")

# Model Engine Statistics
st.markdown("---")
st.markdown("### 📊 Embedding Vector Index Stats")
try:
    if embedding_engine.index is None:
        embedding_engine.load_index()
    if embedding_engine.index is not None:
        st.write(f"• **FAISS Vector Dimension**: `{embedding_engine.index.d}`")
        st.write(f"• **Indexed Resumes (Knowledge Chunks)**: `{embedding_engine.index_size}` Vectors")
    else:
        st.warning("FAISS vector index is not initialized or empty.")
except Exception as e:
    st.error(f"Error reading vector database status: {str(e)}")
