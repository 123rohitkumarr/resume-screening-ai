import streamlit as st
import requests

from frontend.styles.theme import apply_custom_theme
from frontend.utils import init_session_state, get_api_base_url, get_api_headers

apply_custom_theme()
init_session_state()

st.markdown('<h1 class="gradient-text">💬 AI Candidate & Recruiter Assistant</h1>', unsafe_allow_html=True)
st.markdown("Ask the assistant questions about candidate compatibility, tailored interview questions, skill roadmaps, or career coaching grounded in the screening profile.")

api_url = get_api_base_url()
headers = get_api_headers()
session_id = st.session_state.chat_session_id

# Fetch history from DB if memory is empty
if not st.session_state.chat_messages:
    try:
        resp = requests.get(f"{api_url}/api/chat/history/{session_id}", headers=headers, timeout=3)
        if resp.status_code == 200:
            for item in resp.json():
                st.session_state.chat_messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })
    except Exception:
        pass

# Display current candidate context banner if available
if st.session_state.get("screening_result"):
    name = st.session_state.get("last_resume_name", "Candidate")
    title = st.session_state.get("last_jd_title", "Position")
    st.info(f"💡 **Active Profile Grounding**: Questions are tailored specifically to **{name}** evaluating for **{title}**.")

# Render chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question (e.g. 'What interview questions should I ask?', 'How can I bridge the skill gaps?')"):
    # Display user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Bundle screening context if present
    screening_context = None
    if st.session_state.get("screening_result"):
        res = st.session_state.screening_result
        screening_context = {
            "candidate_name": st.session_state.get("last_resume_name", "Candidate"),
            "job_title": st.session_state.get("last_jd_title", "Position"),
            "overall_score": res.get("scores", {}).get("overall", 0.0),
            "matched_skills": res.get("skill_analysis", {}).get("matched", []),
            "missing_skills": res.get("skill_analysis", {}).get("missing", []),
            "predicted_role": res.get("predicted_role", "Unknown"),
            "summary": res.get("summary", "")
        }

    with st.spinner("Analyzing profile & formulating response..."):
        try:
            payload = {
                "message": prompt,
                "session_id": session_id,
                "screening_context": screening_context
            }
            resp = requests.post(f"{api_url}/api/chat/", json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                reply = data["response"]

                # Append assistant reply
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)
                    if data.get("sources"):
                        st.caption(f"Sources utilized: {', '.join(data['sources'])}")
            else:
                err_detail = resp.text
                try:
                    err_detail = resp.json().get("detail", resp.text)
                except Exception:
                    pass
                st.error(f"Assistant API returned error: {err_detail}")
        except Exception as e:
            st.error(f"Error connecting to chatbot endpoint: {str(e)}")

# Clear Chat Option
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🗑️ Clear Chat History"):
    try:
        requests.delete(f"{api_url}/api/chat/history/{session_id}", headers=headers, timeout=3)
    except Exception:
        pass
    st.session_state.chat_messages = []
    st.rerun()
