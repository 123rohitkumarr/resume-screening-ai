import os
import uuid
import streamlit as st
from app.config import settings


def get_api_base_url() -> str:
    """Retrieve backend API base URL from environment, session state, or settings."""
    url = os.getenv("API_BASE_URL", "").strip()
    if not url and "api_url" in st.session_state and st.session_state.api_url:
        url = str(st.session_state.api_url).strip()
    if not url:
        url = getattr(settings, "api_base_url", f"http://127.0.0.1:{settings.api_port}")
    return url.rstrip("/")


def get_api_headers() -> dict:
    """Get HTTP headers including optional API key."""
    api_key = os.getenv("API_KEY") or getattr(settings, "api_key", None)
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def init_session_state():
    """Ensure all required Streamlit session state variables exist safely across all pages."""
    if "api_url" not in st.session_state:
        st.session_state.api_url = get_api_base_url()
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "screening_result" not in st.session_state:
        st.session_state.screening_result = None
    if "last_resume_name" not in st.session_state:
        st.session_state.last_resume_name = None
    if "last_jd_title" not in st.session_state:
        st.session_state.last_jd_title = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
