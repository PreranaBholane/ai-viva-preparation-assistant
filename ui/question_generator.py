"""
Generate Viva Questions UI module.
Allows students to generate tailored viva question banks from their study material
by Subject, Topic, Difficulty, and Question Count.
"""

import streamlit as st
import config
from viva.question_generator import question_generator
from utils.helpers import export_questions_to_markdown
from rag.faiss_store import faiss_manager


def render_question_generator_page():
    """
    Render the Generate Viva Questions page.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🎯 Generate Viva Questions</h1>
            <p class="hero-subtitle">Create targeted viva question banks grounded in your uploaded documents. Choose your subject, focus topic, and difficulty.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not faiss_manager.is_index_present():
        st.warning("⚠️ **Knowledge Base Empty**: Please upload and process study documents in the **📄 Upload Material** tab before generating viva questions.")
        return

    # Configuration Form
    with st.form(key="gen_questions_form"):
        col1, col2 = st.columns(2)

        with col1:
            subject_input = st.text_input(
                "Subject / Course",
                value="Machine Learning / Computer Science",
                help="e.g., Machine Learning, Operating Systems, Data Structures",
            )
            topic_input = st.text_input(
                "Focus Topic / Unit",
                value="Random Forest & Ensemble Methods",
                help="e.g., Ensemble Learning, CNN, Tree Traversals, Overfitting",
            )

        with col2:
            difficulty_choice = st.selectbox(
                "Difficulty Level",
                options=config.SUPPORTED_DIFFICULTIES,
                index=3,  # Mixed by default
                help="Easy (definitions), Medium (mechanisms/comparisons), Hard (deep architecture/math), Mixed (balanced)",
            )
            num_questions_choice = st.slider(
                "Number of Questions",
                min_value=3,
                max_value=15,
                value=5,
                step=1,
            )

        generate_btn = st.form_submit_button("⚡ Generate Viva Questions", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("Analyzing study material and generating viva questions..."):
            active_key = config.get_openai_api_key(st.session_state.get("custom_api_key"))
            active_model = config.get_active_model(st.session_state.get("custom_model"))

            res = question_generator.generate_questions(
                subject=subject_input,
                topic=topic_input,
                difficulty=difficulty_choice,
                num_questions=num_questions_choice,
                api_key=active_key,
                model_name=active_model,
            )

            if res.get("success"):
                st.session_state["generated_viva_questions"] = res["questions"]
                st.session_state["gen_subject"] = subject_input
                st.session_state["gen_topic"] = topic_input
                st.session_state["gen_sources"] = res.get("sources", [])
                st.success(f"Generated {len(res['questions'])} grounded viva questions!")
            else:
                st.error(f"Generation failed: {res.get('error')}")

    # Display Generated Questions
    if "generated_viva_questions" in st.session_state and st.session_state["generated_viva_questions"]:
        questions = st.session_state["generated_viva_questions"]
        subj = st.session_state.get("gen_subject", "Viva")
        top = st.session_state.get("gen_topic", "General")

        st.markdown("---")
        st.subheader(f"📋 Question Bank ({len(questions)} Questions)")

        for idx, q in enumerate(questions, start=1):
            diff = q.get("difficulty", "Medium")
            badge_class = "badge-medium"
            if diff.lower() == "easy":
                badge_class = "badge-easy"
            elif diff.lower() == "hard":
                badge_class = "badge-hard"

            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: #f8fafc;">Q{idx}. {q.get('question')}</span>
                        <span class="badge {badge_class}">{diff}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 10px;">
                        <strong>Topic:</strong> {q.get('topic', top)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"🔍 View Model Answer & Key Points for Q{idx}"):
                if q.get("ideal_answer"):
                    st.markdown(f"**Ideal Viva Answer:**\n{q.get('ideal_answer')}")
                if q.get("key_points"):
                    st.markdown("**Key Points Expected by Examiner:**")
                    for pt in q.get("key_points", []):
                        st.markdown(f"- {pt}")

        # Quick Actions: Export & Start Mock Viva
        st.markdown("---")
        act_col1, act_col2 = st.columns([1, 1])

        with act_col1:
            q_md = export_questions_to_markdown(questions, subject=subj, topic=top)
            st.download_button(
                label="📥 Download Question Bank (Markdown)",
                data=q_md,
                file_name=f"viva_questions_{top.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with act_col2:
            if st.button("🎤 Practice These in Mock Viva Room", type="primary", use_container_width=True):
                # Transfer questions to Mock Viva Session
                st.session_state["mock_custom_questions"] = questions
                st.session_state["active_nav"] = "Mock Viva"
                st.rerun()
