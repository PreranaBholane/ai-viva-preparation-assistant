"""
Evaluator module for scoring student viva responses across Correctness, Relevance,
Completeness, Clarity, and Overall Score grounded in retrieved study material.
"""

import json
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
import config
from rag.retriever import VivaRetriever
from llm.llm_service import invoke_llm
from llm.prompts import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_TEMPLATE
from viva.question_generator import extract_json_from_response


class VivaAnswerEvaluator:
    """
    Evaluates student viva responses against reference study material.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.retriever = VivaRetriever(top_k=4, api_key=api_key)

    def evaluate_answer(
        self,
        question: str,
        student_answer: str,
        ideal_concept: str = "",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a student's answer by retrieving relevant material and executing
        the multi-criteria scoring prompt.
        """
        active_key = api_key or config.get_openai_api_key(self.api_key)
        if not active_key:
            return {
                "success": False,
                "error": "OpenAI API Key is missing. Please configure it in Settings.",
                "overall_score": 0.0,
            }

        if not student_answer or not student_answer.strip():
            return {
                "success": True,
                "correctness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "overall_score": 0.0,
                "feedback": "No answer provided.",
                "strengths": [],
                "improvements": ["Please attempt the question to receive feedback."],
                "ideal_answer": ideal_concept or "A complete explanation was expected.",
            }

        # Retrieve relevant context from FAISS
        try:
            try:
                retrieval = self.retriever.get_context_and_sources(
                    query=f"{question} {ideal_concept}".strip(),
                    k=4,
                    api_key=active_key,
                )
                context = retrieval.get("context", "")
            except Exception as r_err:
                err_lower = str(r_err).lower()
                if "credit_balance_exhausted" in err_lower or "insufficient_quota" in err_lower or "429" in err_lower:
                    self.retriever.api_key = None
                    retrieval = self.retriever.get_context_and_sources(
                        query=f"{question} {ideal_concept}".strip(), k=4
                    )
                    context = retrieval.get("context", "")
                else:
                    raise r_err

            user_prompt = EVALUATOR_USER_TEMPLATE.format(
                context=context or "Evaluate based on standard technical principles.",
                question=question,
                student_answer=student_answer,
                ideal_concept=ideal_concept or "N/A",
            )

            messages = [
                SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response_text = invoke_llm(
                messages=messages,
                model_name=model_name or config.get_active_model(),
                temperature=0.2,
                api_key=active_key,
            )

            result = extract_json_from_response(response_text)

            # Ensure numeric safety
            correctness = min(10.0, max(0.0, float(result.get("correctness", 0.0))))
            relevance = min(10.0, max(0.0, float(result.get("relevance", 0.0))))
            completeness = min(10.0, max(0.0, float(result.get("completeness", 0.0))))
            clarity = min(10.0, max(0.0, float(result.get("clarity", 0.0))))
            overall_score = min(10.0, max(0.0, float(result.get("overall_score", 0.0))))

            return {
                "success": True,
                "correctness": round(correctness, 1),
                "relevance": round(relevance, 1),
                "completeness": round(completeness, 1),
                "clarity": round(clarity, 1),
                "overall_score": round(overall_score, 1),
                "feedback": result.get("feedback", "Good effort."),
                "strengths": result.get("strengths", []),
                "improvements": result.get("improvements", []),
                "ideal_answer": result.get("ideal_answer", ""),
                "sources": retrieval.get("sources", []),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Evaluation error: {str(e)}",
                "overall_score": 0.0,
            }


# Global instance
evaluator = VivaAnswerEvaluator()
