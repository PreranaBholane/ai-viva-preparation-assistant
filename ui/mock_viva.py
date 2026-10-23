"""
Interactive Mock Viva Examination Room UI module.
Simulates a live oral viva examination with step-by-step questions, real-time
evaluator feedback, follow-up probing, and a final comprehensive scorecard.
"""

import streamlit as st
import config
from rag.faiss_store import faiss_manager
from viva.question_generator import question_generator
from viva.mock_viva import MockVivaSession
from utils.helpers import create_radar_score_chart, create_questions_bar_chart


def render_mock_viva_page():
    """
    Render the Interactive Mock Viva page.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🎤 Interactive Mock Viva Examination</h1>
            <p class="hero-subtitle">Step into the exam room. The AI Examiner will ask viva questions, probe your understanding, evaluate your answers in real time, and compile your final scorecard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not faiss_manager.is_index_present():
        st.warning("⚠️ **Knowledge Base Empty**: Please upload course materials in the **📄 Upload Material** tab before launching a Mock Viva.")
        return

    # Check if a session is currently active
    if "mock_session" not in st.session_state or st.session_state["mock_session"] is None:
        render_mock_setup()
    else:
        session: MockVivaSession = st.session_state["mock_session"]
        if session.completed:
            render_mock_results(session)
        else:
            render_active_examination(session)


def render_mock_setup():
    """
    Render setup view to configure and initiate a mock viva session.
    """
    st.subheader("⚙️ Mock Viva Setup")

    custom_q = st.session_state.get("mock_custom_questions", None)

    if custom_q:
        st.info(f"✨ Using **{len(custom_q)} questions** transferred from the Question Generator.")
        use_transferred = True
    else:
        use_transferred = False

    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", value="Machine Learning / CS", key="mock_subj_in")
        topic = st.text_input("Topic", value="Supervised Learning & Algorithms", key="mock_topic_in")
    with col2:
        num_q = st.slider("Number of Questions", min_value=3, max_value=10, value=5, key="mock_num_q")
        difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"], key="mock_diff")

    if st.button("🚀 Enter Viva Examination Room", type="primary", use_container_width=True):
        with st.spinner("Preparing viva examination panel and questions..."):
            active_key = config.get_openai_api_key(st.session_state.get("custom_api_key"))
            active_model = config.get_active_model(st.session_state.get("custom_model"))

            if use_transferred:
                questions = custom_q
                st.session_state.pop("mock_custom_questions", None)
            else:
                res = question_generator.generate_questions(
                    subject=subject,
                    topic=topic,
                    difficulty=difficulty,
                    num_questions=num_q,
                    api_key=active_key,
                    model_name=active_model,
                )
                if not res.get("success"):
                    st.error(f"Failed to set up mock viva: {res.get('error')}")
                    return
                questions = res.get("questions", [])

            session = MockVivaSession(questions=questions, subject=subject, api_key=active_key)
            st.session_state["mock_session"] = session
            st.session_state["last_eval_record"] = None
            st.rerun()


def render_active_examination(session: MockVivaSession):
    """
    Render active question-and-answer exam interface.
    """
    current_q = session.get_current_question()
    q_idx = session.current_index + 1
    total_q = len(session.questions)

    # Progress Bar
    progress_val = session.current_index / total_q
    st.progress(progress_val, text=f"Question {q_idx} of {total_q}")

    # Display Current Question Card
    diff = current_q.get("difficulty", "Medium")
    badge_class = "badge-medium"
    if diff.lower() == "easy":
        badge_class = "badge-easy"
    elif diff.lower() == "hard":
        badge_class = "badge-hard"

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #6366f1;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.9rem; color: #a5b4fc; text-transform: uppercase; font-weight: 700;">Examiner Question #{q_idx}</span>
                <span class="badge {badge_class}">{diff}</span>
            </div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-bottom: 10px;">
                "{current_q.get('question')}"
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                <strong>Topic:</strong> {current_q.get('topic', 'General')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show Previous Answer Feedback if just evaluated
    last_record = st.session_state.get("last_eval_record")
    if last_record and last_record.get("question_index") == session.current_index:
        with st.expander(f"📝 Immediate Feedback on Question #{last_record['question_index']}", expanded=True):
            s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
            s_col1.metric("Score", f"{last_record['overall_score']}/10")
            s_col2.metric("Correctness", f"{last_record['correctness']}/10")
            s_col3.metric("Relevance", f"{last_record['relevance']}/10")
            s_col4.metric("Completeness", f"{last_record['completeness']}/10")
            s_col5.metric("Clarity", f"{last_record['clarity']}/10")

            st.markdown(f"**Examiner Feedback:** {last_record['feedback']}")

            if last_record.get("strengths"):
                st.markdown("**✓ Strengths:** " + ", ".join(last_record["strengths"]))
            if last_record.get("improvements"):
                st.markdown("**⚠ Missing Points:** " + ", ".join(last_record["improvements"]))

    # Student Input Area
    with st.form(key=f"answer_form_{q_idx}"):
        student_response = st.text_area(
            "Your Answer (type what you would say in viva):",
            placeholder="Explain the concept clearly, define terms, and provide examples...",
            height=130,
        )

        col_sub, col_skip = st.columns([3, 1])
        with col_sub:
            submit_ans_btn = st.form_submit_button("📤 Submit Answer to Examiner", type="primary", use_container_width=True)
        with col_skip:
            skip_btn = st.form_submit_button("⏭️ Pass Question", use_container_width=True)

    if submit_ans_btn or skip_btn:
        ans_text = student_response if submit_ans_btn else "I do not know the answer to this question."
        with st.spinner("Examiner is evaluating your response against study material..."):
            active_key = config.get_openai_api_key(st.session_state.get("custom_api_key"))
            active_model = config.get_active_model(st.session_state.get("custom_model"))

            record = session.submit_answer(
                student_answer=ans_text,
                api_key=active_key,
                model_name=active_model,
            )
            st.session_state["last_eval_record"] = record
            st.rerun()

    # Cancel session option
    if st.button("🛑 End Mock Viva Early"):
        session.completed = True
        st.rerun()


def render_mock_results(session: MockVivaSession):
    """
    Render the comprehensive final Mock Viva Result scorecard.
    """
    st.markdown("## 🎓 MOCK VIVA EXAMINATION RESULT")

    active_key = config.get_openai_api_key(st.session_state.get("custom_api_key"))
    active_model = config.get_active_model(st.session_state.get("custom_model"))

    with st.spinner("Compiling examiner assessment report and performance analytics..."):
        report = session.get_final_report(api_key=active_key, model_name=active_model)

    # Top Metric Banners
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Questions Evaluated</div>
                <div class="metric-value">{report['total_questions']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Score</div>
                <div class="metric-value">{report['overall_score']}/10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Percentage</div>
                <div class="metric-value">{report['percentage']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        readiness = "Ready" if report["percentage"] >= 75 else ("Revision Needed" if report["percentage"] >= 50 else "Critical Prep")
        color = "#10b981" if report["percentage"] >= 75 else ("#f59e0b" if report["percentage"] >= 50 else "#ef4444")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Viva Readiness</div>
                <div class="metric-value" style="color: {color}; font-size: 1.4rem;">{readiness}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Visual Charts
    col_rad, col_bar = st.columns(2)

    with col_rad:
        st.markdown("##### 🕸️ Multi-Criteria Competency Radar")
        radar_fig = create_radar_score_chart(
            correctness=report.get("avg_correctness", 0.0),
            relevance=report.get("avg_relevance", 0.0),
            completeness=report.get("avg_completeness", 0.0),
            clarity=report.get("avg_clarity", 0.0),
            overall=report.get("overall_score", 0.0),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with col_bar:
        st.markdown("##### 📊 Score Distribution by Question")
        bar_fig = create_questions_bar_chart(session.history)
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # Strong Areas vs Needs Improvement
    col_str, col_imp = st.columns(2)

    with col_str:
        st.markdown(
            """
            <div class="glass-card" style="border-top: 3px solid #10b981;">
                <h4 style="color: #34d399; margin-top: 0;">✓ Strong Areas</h4>
            """,
            unsafe_allow_html=True,
        )
        for area in report.get("strong_areas", []):
            st.markdown(f"- ✅ **{area}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_imp:
        st.markdown(
            """
            <div class="glass-card" style="border-top: 3px solid #f59e0b;">
                <h4 style="color: #fbbf24; margin-top: 0;">⚠ Needs Improvement</h4>
            """,
            unsafe_allow_html=True,
        )
        for area in report.get("needs_improvement", []):
            st.markdown(f"- ⚠️ **{area}**")
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Examiner Verdict
    st.markdown("### 📋 Examiner Summary & Actionable Viva Advice")
    st.markdown(f"> *{report.get('summary')}*")

    if report.get("actionable_tips"):
        st.markdown("##### 💡 Key Actionable Tips for Viva Day:")
        for tip in report["actionable_tips"]:
            st.markdown(f"- 📌 {tip}")

    st.markdown("---")

    # Detailed Breakdown Table
    with st.expander("🔍 View Complete Question-by-Question Transcript & Scores"):
        for item in session.history:
            st.markdown(f"#### Q{item['question_index']}. {item['question']}")
            st.markdown(f"- **Your Answer:** {item['student_answer']}")
            st.markdown(f"- **Score:** `{item['overall_score']}/10` (Correctness: {item['correctness']}, Relevance: {item['relevance']}, Completeness: {item['completeness']}, Clarity: {item['clarity']})")
            st.markdown(f"- **Examiner Feedback:** {item['feedback']}")
            if item.get("ideal_answer"):
                st.markdown(f"- **Model Viva Answer:** *{item['ideal_answer']}*")
            st.markdown("---")

    # Retake / Restart Button
    if st.button("🔄 Retake or Start New Mock Viva", type="primary", use_container_width=True):
        st.session_state["mock_session"] = None
        st.session_state["last_eval_record"] = None
        st.rerun()
