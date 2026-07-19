import streamlit as st
import requests
import uuid

from frontend.styles.theme import apply_custom_theme

apply_custom_theme()

st.markdown('<h1 class="gradient-text">💬 AI Candidate & Recruiter Assistant</h1>', unsafe_allow_html=True)
st.markdown("Ask the chatbot questions about candidate compatibility, interview questions, resume improvements, or general career advice.")

# Ensure unique session ID for chatbot
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Fetch history from DB if empty
if not st.session_state.chat_messages:
    try:
        api_url = st.session_state.api_url
        session_id = st.session_state.chat_session_id
        resp = requests.get(f"{api_url}/api/chat/history/{session_id}", timeout=2)
        if resp.status_code == 200:
            for item in resp.json():
                st.session_state.chat_messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })
    except Exception:
        pass

# Display current context if available
if st.session_state.screening_result:
    name = st.session_state.last_resume_name
    title = st.session_state.last_jd_title
    st.info(f"💡 **Context Enabled**: Asking questions specifically grounded in the screening of **{name}** for **{title}**.")

# Render chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question (e.g. 'What interview questions should I ask Jane?', 'How can I learn SQL fast?')"):
    # Display user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call Backend Chat Endpoint
    api_url = st.session_state.api_url
    session_id = st.session_state.chat_session_id
    
    # Bundle screening context if present
    screening_context = None
    if st.session_state.screening_result:
        res = st.session_state.screening_result
        screening_context = {
            "candidate_name": st.session_state.last_resume_name,
            "job_title": st.session_state.last_jd_title,
            "overall_score": res.get("scores", {}).get("overall", 0.0),
            "matched_skills": res.get("skill_analysis", {}).get("matched", []),
            "missing_skills": res.get("skill_analysis", {}).get("missing", []),
            "predicted_role": res.get("predicted_role", "Unknown"),
            "summary": res.get("summary", "")
        }

    with st.spinner("Thinking..."):
        try:
            payload = {
                "message": prompt,
                "session_id": session_id,
                "screening_context": screening_context
            }
            resp = requests.post(f"{api_url}/api/chat/", json=payload)
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
                st.error("Failed to fetch response from assistant API.")
        except Exception as e:
            st.error(f"Error connecting to chatbot endpoint: {str(e)}")

# Clear Chat Option
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🗑️ Clear Chat History"):
    try:
        api_url = st.session_state.api_url
        session_id = st.session_state.chat_session_id
        requests.delete(f"{api_url}/api/chat/history/{session_id}")
    except Exception:
        pass
    st.session_state.chat_messages = []
    st.experimental_rerun()
