"""
Helper utilities for UI rendering, Plotly chart generation, and data export.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def create_radar_score_chart(
    correctness: float,
    relevance: float,
    completeness: float,
    clarity: float,
    overall: float,
) -> go.Figure:
    """
    Generate an interactive Plotly radar chart displaying multi-criteria viva performance.
    """
    categories = ["Correctness", "Relevance", "Completeness", "Clarity", "Overall"]
    values = [correctness, relevance, completeness, clarity, overall]
    # Close the radar polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.35)",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=8, color="#4f46e5"),
            name="Your Performance",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                ticktext=["2", "4", "6", "8", "10"],
                gridcolor="rgba(255, 255, 255, 0.15)",
                linecolor="rgba(255, 255, 255, 0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.15)",
                linecolor="rgba(255, 255, 255, 0.2)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        showlegend=False,
    )

    return fig


def create_questions_bar_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Generate an interactive Plotly bar chart showing score per question.
    """
    if not history:
        return go.Figure()

    q_labels = [f"Q{item.get('question_index', idx+1)}" for idx, item in enumerate(history)]
    scores = [item.get("overall_score", 0.0) for item in history]

    colors = []
    for s in scores:
        if s >= 8.0:
            colors.append("#10b981")  # Green
        elif s >= 6.0:
            colors.append("#f59e0b")  # Amber
        else:
            colors.append("#ef4444")  # Red

    fig = go.Figure(
        data=[
            go.Bar(
                x=q_labels,
                y=scores,
                marker_color=colors,
                text=[f"{s}/10" for s in scores],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        yaxis=dict(range=[0, 10], title="Score (/10)", gridcolor="rgba(255, 255, 255, 0.1)"),
        xaxis=dict(title="Viva Questions"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=20, b=30),
        height=260,
    )

    return fig


def export_questions_to_markdown(
    questions: List[Dict[str, Any]],
    subject: str = "Viva Prep",
    topic: str = "General",
) -> str:
    """
    Convert generated viva questions to structured Markdown for downloading.
    """
    md_lines = [
        f"# AI Viva Preparation Assistant — Question Bank",
        f"**Subject:** {subject} | **Topic:** {topic}",
        f"**Generated Count:** {len(questions)} Questions",
        "",
        "---",
        "",
    ]

    for idx, q in enumerate(questions, start=1):
        md_lines.append(f"### Q{idx}. {q.get('question')}")
        md_lines.append(f"- **Difficulty:** `{q.get('difficulty', 'Medium')}`")
        md_lines.append(f"- **Topic:** {q.get('topic', 'General')}")
        if q.get("ideal_answer"):
            md_lines.append(f"- **Model Answer:** {q.get('ideal_answer')}")
        if q.get("key_points"):
            md_lines.append("- **Key Evaluation Points:**")
            for pt in q.get("key_points", []):
                md_lines.append(f"  * {pt}")
        md_lines.append("")

    return "\n".join(md_lines)


def export_chat_history_to_markdown(messages: List[Dict[str, Any]]) -> str:
    """
    Convert chat session history into readable Markdown.
    """
    lines = [
        "# AI Viva Preparation Assistant — Chat Transcript",
        "",
        "---",
        "",
    ]

    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        lines.append(f"### {role}:")
        lines.append(content)
        if sources:
            lines.append("\n**Sources Referenced:**")
            for s in sources:
                lines.append(f"- 📄 `{s.get('filename')}` (Page {s.get('page')})")
        lines.append("\n---\n")

    return "\n".join(lines)
