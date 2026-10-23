"""
AI Viva Preparation Assistant — Main Application Entrypoint
A College Teacher Assessment (TA) RAG application built with Streamlit, FAISS,
LangChain, and OpenAI GPT-5.6 Luna.
"""


import streamlit as st
import config
from ui.styles import apply_custom_styles
from ui.upload import render_upload_page
from ui.chatbot import render_chatbot_page
from ui.question_generator import render_question_generator_page
from ui.mock_viva import render_mock_viva_page
from ui.settings import render_settings_page
from rag.faiss_store import faiss_manager


def main():
    """
    Main application setup and page routing.
    """
    # Streamlit Page Configuration
    st.set_page_config(
        page_title="AI Viva Preparation Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom modern glassmorphic styling
    apply_custom_styles()

    # Sidebar Header and Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 12px 0;">
                <h2 style="font-weight: 800; background: linear-gradient(90deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
                    🎓 AI VIVA ASSISTANT
                </h2>
                <span style="font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">
                    College Viva Preparation
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Knowledge Base Status Indicator in Sidebar
        stats = faiss_manager.get_index_stats()
        if stats["index_exists"]:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; margin-bottom: 12px;">
                    <span class="status-dot-ready"></span>
                    <strong style="color: #34d399;">Knowledge Base Ready</strong>
                    <div style="color: #94a3b8; margin-top: 2px;">{stats['total_vectors']} vectors indexed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; margin-bottom: 12px;">
                    <span class="status-dot-empty"></span>
                    <strong style="color: #f87171;">Knowledge Base Empty</strong>
                    <div style="color: #94a3b8; margin-top: 2px;">Upload study material to begin</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Navigation Options
        nav_options = [
            "📄 Upload Material",
            "💬 AI Viva Chat",
            "🎯 Generate Questions",
            "🎤 Mock Viva",
            "⚙ Settings",
        ]

        # Sync active nav with programmatic triggers if present
        default_index = 0
        if "active_nav" in st.session_state:
            target = st.session_state.pop("active_nav")
            for idx, opt in enumerate(nav_options):
                if target in opt:
                    default_index = idx
                    break

        selected_page = st.radio(
            "Navigation",
            options=nav_options,
            index=default_index,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System Specs in Sidebar Footer
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: #64748b; line-height: 1.6;">
                <strong>Engine:</strong> {config.get_active_model(st.session_state.get('custom_model'))}<br/>
                <strong>Vector Store:</strong> FAISS (Local)<br/>
                <strong>RAG Pipeline:</strong> LangChain<br/>
                <strong>Status:</strong> Active
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Page Routing
    if "Upload Material" in selected_page:
        render_upload_page()
    elif "AI Viva Chat" in selected_page:
        render_chatbot_page()
    elif "Generate Questions" in selected_page:
        render_question_generator_page()
    elif "Mock Viva" in selected_page:
        render_mock_viva_page()
    elif "Settings" in selected_page:
        render_settings_page()


if __name__ == "__main__":
    main()
