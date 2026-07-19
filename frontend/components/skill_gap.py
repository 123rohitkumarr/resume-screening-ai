import plotly.graph_objects as go
import streamlit as st

def render_radar_breakdown(scores_dict: dict):
    """
    Renders a radar chart comparing the score dimensions:
    Semantic, Skills, Experience, and Education.
    """
    categories = ['Semantic Match', 'Skill Fit', 'Experience Match', 'Education Fit']
    
    # Map from ScoreBreakdown fields
    values = [
        scores_dict.get('semantic', 0),
        scores_dict.get('skill', 0),
        scores_dict.get('experience', 0),
        scores_dict.get('education', 0)
    ]
    
    # Complete the circular loop
    categories_loop = categories + [categories[0]]
    values_loop = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_loop,
        theta=categories_loop,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',  # Indigo semi-transparent fill
        line=dict(color='#818cf8', width=2),
        name='Candidate Metrics'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='#94a3b8',
                gridcolor='rgba(255, 255, 255, 0.1)',
                angle=45
            ),
            angularaxis=dict(
                color='#f8fafc',
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            bgcolor='rgba(15, 23, 42, 0.4)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=250,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def render_skills_bar_chart(matched_skills: list, missing_skills: list):
    """
    Renders a side-by-side comparison of Matched vs Missing skills.
    """
    labels = ['Matched Skills', 'Missing Skills']
    counts = [len(matched_skills), len(missing_skills)]
    colors = ['#10b981', '#ef4444']

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=counts,
        marker_color=colors,
        width=0.4
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc', 'family': 'Outfit'},
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            color='#94a3b8',
            title='Count'
        ),
        xaxis=dict(
            color='#f8fafc'
        ),
        margin=dict(l=20, r=20, t=10, b=10),
        height=200,
    )

    st.plotly_chart(fig, use_container_width=True)
