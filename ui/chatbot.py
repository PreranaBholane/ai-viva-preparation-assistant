"""
AI Viva Chatbot UI module.
Provides an interactive chat interface for students to ask viva questions,
grounded strictly in FAISS retrieved study material, with source expanders.
"""

import streamlit as st
import config
from rag.rag_chain import rag_chain
from rag.faiss_store import faiss_manager
from utils.helpers import export_chat_history_to_markdown


def render_chatbot_page():
    """
    Render the AI Viva Chatbot page.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">💬 AI Viva Chatbot</h1>
            <p class="hero-subtitle">Ask any viva question from your uploaded study material. Answers are strictly grounded in your documents with source attribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history in session_state if not present
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your **AI Viva Preparation Assistant**. Ask me any question from your uploaded study material, and I'll explain it clearly with viva tips, definitions, and examples.",
                "sources": [],
            }
        ]

    # Quick Question Chips / Starter Prompts
    st.markdown("##### 💡 Suggested Viva Questions:")
    quick_cols = st.columns(4)
    quick_questions = [
        "What is Random Forest?",
        "Explain CNN architecture.",
        "What is Supervised Learning?",
        "Compare BFS vs DFS.",
    ]

    selected_quick_prompt = None
    for idx, q_text in enumerate(quick_questions):
        with quick_cols[idx]:
            if st.button(q_text, key=f"quick_q_{idx}", use_container_width=True):
                selected_quick_prompt = q_text

    st.markdown("---")

    # Display Chat History
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Source Expanders if sources exist
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 View Retrieved Sources ({len(sources)} chunks used)"):
                    for s in sources:
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <strong>📄 {s.get('filename')}</strong> &nbsp;|&nbsp; 
                                <strong>Page/Slide:</strong> {s.get('page')} &nbsp;|&nbsp;
                                <strong>Section:</strong> {s.get('topic', 'General')}
                                <br/>
                                <small style="color: #94a3b8;">Similarity Distance: {s.get('similarity_score')}</small>
                                <div style="margin-top: 6px; font-style: italic; color: #cbd5e1;">
                                    "{s.get('content_snippet')}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input Handler
    user_query = st.chat_input("Ask your viva question...") or selected_quick_prompt

    if user_query:
        # Append and display user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant study material & generating viva answer..."):
                active_key = config.get_openai_api_key(st.session_state.get("custom_api_key"))
                active_model = config.get_active_model(st.session_state.get("custom_model"))

                rag_res = rag_chain.answer_question(
                    question=user_query,
                    top_k=config.TOP_K,
                    api_key=active_key,
                    model_name=active_model,
                )

                answer_text = rag_res.get("answer", "")
                sources = rag_res.get("sources", [])

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 View Retrieved Sources ({len(sources)} chunks used)"):
                        for s in sources:
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <strong>📄 {s.get('filename')}</strong> &nbsp;|&nbsp; 
                                    <strong>Page/Slide:</strong> {s.get('page')} &nbsp;|&nbsp;
                                    <strong>Section:</strong> {s.get('topic', 'General')}
                                    <br/>
                                    <small style="color: #94a3b8;">Similarity Distance: {s.get('similarity_score')}</small>
                                    <div style="margin-top: 6px; font-style: italic; color: #cbd5e1;">
                                        "{s.get('content_snippet')}"
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Append assistant message to history
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                })

    # Action buttons below chat
    st.markdown("---")
    act_col1, act_col2, _ = st.columns([1, 1, 2])
    with act_col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared! Feel free to ask another viva question.",
                    "sources": [],
                }
            ]
            st.rerun()

    with act_col2:
        chat_md = export_chat_history_to_markdown(st.session_state["chat_messages"])
        st.download_button(
            label="📥 Export Chat Transcript",
            data=chat_md,
            file_name="viva_chat_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )
