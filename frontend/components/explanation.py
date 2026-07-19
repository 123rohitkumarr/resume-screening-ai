import streamlit as st

def render_xai_explanation(explanation_items: list):
    """
    Renders a custom SHAP-style feature attribution panel using styled HTML.
    Each explanation item contains:
    - factor: name of the score dimension
    - score: value between 0-100
    - impact: 'positive', 'neutral', 'negative'
    - detail: custom feedback rationale
    """
    st.markdown("### 🔍 Model Feature Attribution (Explainable AI)")
    st.markdown(
        "This panel breaks down how each evaluation criteria influenced "
        "the final compatibility score. Green indicates a positive contribution, "
        "yellow indicates neutral, and red indicates negative."
    )

    for item in explanation_items:
        # Load variables
        factor = item.get('factor', 'Unknown')
        score = item.get('score', 0.0)
        impact = item.get('impact', 'neutral')
        detail = item.get('detail', '')

        # Set impact color configurations
        if impact == "positive":
            bar_color = "linear-gradient(90deg, #059669 0%, #10b981 100%)"
            badge_color = "rgba(16, 185, 129, 0.15)"
            text_color = "#10b981"
            sign = "+"
        elif impact == "negative":
            bar_color = "linear-gradient(90deg, #dc2626 0%, #ef4444 100%)"
            badge_color = "rgba(239, 68, 68, 0.15)"
            text_color = "#ef4444"
            sign = "-"
        else:
            bar_color = "linear-gradient(90deg, #d97706 0%, #f59e0b 100%)"
            badge_color = "rgba(245, 158, 11, 0.15)"
            text_color = "#f59e0b"
            sign = ""

        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom: 12px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 1.05rem; color: #ffffff;">{factor}</div>
                    <div style="background-color: {badge_color}; color: {text_color}; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">
                        {sign}{score:.0f}% Impact
                    </div>
                </div>
                <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 9999px; height: 8px; margin-bottom: 8px; overflow: hidden;">
                    <div style="background: {bar_color}; width: {score}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.4;">
                    {detail}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
