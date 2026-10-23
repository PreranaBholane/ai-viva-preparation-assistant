"""
Feedback module for aggregating session performance, calculating final scores,
identifying strong/weak topic areas, and generating actionable revision recommendations.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
import config
from llm.llm_service import invoke_llm
from llm.prompts import FINAL_REPORT_SYSTEM_PROMPT
from viva.question_generator import extract_json_from_response


class VivaFeedbackCompiler:
    """
    Compiles detailed final assessment report from individual mock viva evaluations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate_final_report(
        self,
        session_evaluations: List[Dict[str, Any]],
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compile statistical metrics and synthesize an overarching assessment report.
        """
        if not session_evaluations:
            return {
                "total_questions": 0,
                "overall_score": 0.0,
                "percentage": 0.0,
                "strong_areas": [],
                "needs_improvement": [],
                "summary": "No questions were completed during this viva session.",
                "actionable_tips": [],
            }

        total_questions = len(session_evaluations)
        scores = [e.get("overall_score", 0.0) for e in session_evaluations]
        avg_score = round(sum(scores) / total_questions, 2)
        percentage = round((avg_score / 10.0) * 100, 1)

        # Average metric breakdowns
        avg_correctness = round(
            sum(e.get("correctness", 0.0) for e in session_evaluations) / total_questions, 1
        )
        avg_relevance = round(
            sum(e.get("relevance", 0.0) for e in session_evaluations) / total_questions, 1
        )
        avg_completeness = round(
            sum(e.get("completeness", 0.0) for e in session_evaluations) / total_questions, 1
        )
        avg_clarity = round(
            sum(e.get("clarity", 0.0) for e in session_evaluations) / total_questions, 1
        )

        # Build context summary for LLM final assessment
        eval_summaries = []
        for idx, item in enumerate(session_evaluations):
            q_text = item.get("question", f"Question {idx+1}")
            topic = item.get("topic", "General")
            s_score = item.get("overall_score", 0.0)
            feedback = item.get("feedback", "")
            eval_summaries.append(
                f"- Q{idx+1} ({topic}): Score {s_score}/10 | Feedback: {feedback}"
            )

        context_text = "\n".join(eval_summaries)

        active_key = api_key or config.get_openai_api_key(self.api_key)

        strong_areas = []
        needs_improvement = []
        summary_text = f"You completed {total_questions} questions with an overall score of {avg_score}/10 ({percentage}%)."
        actionable_tips = []

        # Compute strong and weak areas directly from the actual questions attempted
        for e in session_evaluations:
            t = e.get("topic", "General")
            if e.get("overall_score", 0) >= 7.5:
                if t not in strong_areas:
                    strong_areas.append(t)
            else:
                if t not in needs_improvement:
                    needs_improvement.append(t)

        if active_key:
            try:
                user_msg = f"Student Mock Viva Summary (Total Score: {avg_score}/10):\n\n{context_text}"
                messages = [
                    SystemMessage(content=FINAL_REPORT_SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
                resp = invoke_llm(
                    messages=messages,
                    model_name=model_name or config.get_active_model(),
                    temperature=0.3,
                    api_key=active_key,
                )
                parsed = extract_json_from_response(resp)
                summary_text = parsed.get("summary", summary_text)
                actionable_tips = parsed.get("actionable_tips", [])
            except Exception:
                pass

        return {
            "total_questions": total_questions,
            "overall_score": avg_score,
            "percentage": percentage,
            "avg_correctness": avg_correctness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_clarity": avg_clarity,
            "strong_areas": strong_areas or ["Foundational Concepts"],
            "needs_improvement": needs_improvement or ["Deep Architectural Nuances"],
            "summary": summary_text,
            "actionable_tips": actionable_tips
            or ["Review highlighted topics before your final viva."],
        }


# Global instance
feedback_compiler = VivaFeedbackCompiler()
