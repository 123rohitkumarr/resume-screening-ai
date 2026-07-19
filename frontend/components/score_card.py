import plotly.graph_objects as go
import streamlit as st

def render_score_gauge(score: float, title: str = "Overall Match"):
    """
    Renders an interactive, high-fidelity circular gauge chart using Plotly.
    Color ranges match typical recruiter assessment bands:
    - 0-40: Low Match (Red)
    - 40-70: Mid Match (Orange/Yellow)
    - 70-100: Excellent Match (Teal/Green)
    """
    # Select color based on score band
    if score >= 75:
        accent_color = "#10b981"  # Emerald
    elif score >= 50:
        accent_color = "#f59e0b"  # Amber
    else:
        accent_color = "#ef4444"  # Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 20, 'color': '#ffffff', 'family': 'Outfit'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 50, 'color': '#ffffff', 'family': 'Outfit'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': accent_color, 'thickness': 0.25},
            'bgcolor': "rgba(15, 23, 42, 0.5)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    # Update layout to be transparent and fit nicely in Streamlit columns
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': "Outfit"},
        margin=dict(l=10, r=10, t=40, b=10),
        height=220,
    )

    st.plotly_chart(fig, use_container_width=True)
