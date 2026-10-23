"""
Mock Viva Engine managing interactive oral examination state machine,
probing follow-up question generation, and score aggregation.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
import config
from rag.retriever import VivaRetriever
from llm.llm_service import invoke_llm
from llm.prompts import MOCK_EXAMINER_SYSTEM_PROMPT
from viva.evaluator import evaluator
from viva.feedback import feedback_compiler


class MockVivaSession:
    """
    Stateful engine representing an ongoing mock viva examination session.
    """

    def __init__(
        self,
        questions: List[Dict[str, Any]],
        subject: str = "Computer Science",
        api_key: Optional[str] = None,
    ):
        self.questions = questions
        self.subject = subject
        self.api_key = api_key
        self.current_index = 0
        self.history: List[Dict[str, Any]] = []
        self.completed = False
        self.retriever = VivaRetriever(top_k=4, api_key=api_key)

    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """
        Return the current viva question object.
        """
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def submit_answer(
        self,
        student_answer: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit student answer for current question, evaluate it, record in history,
        and advance to the next question.
        """
        current_q = self.get_current_question()
        if not current_q:
            return {"error": "No active question found."}

        active_key = api_key or self.api_key

        # Evaluate answer
        eval_result = evaluator.evaluate_answer(
            question=current_q["question"],
            student_answer=student_answer,
            ideal_concept=current_q.get("ideal_answer", ""),
            api_key=active_key,
            model_name=model_name,
        )

        record = {
            "question_index": self.current_index + 1,
            "question": current_q["question"],
            "difficulty": current_q.get("difficulty", "Medium"),
            "topic": current_q.get("topic", "General"),
            "ideal_answer": current_q.get("ideal_answer", ""),
            "student_answer": student_answer,
            "correctness": eval_result.get("correctness", 0.0),
            "relevance": eval_result.get("relevance", 0.0),
            "completeness": eval_result.get("completeness", 0.0),
            "clarity": eval_result.get("clarity", 0.0),
            "overall_score": eval_result.get("overall_score", 0.0),
            "feedback": eval_result.get("feedback", ""),
            "strengths": eval_result.get("strengths", []),
            "improvements": eval_result.get("improvements", []),
            "sources": eval_result.get("sources", []),
        }

        self.history.append(record)
        self.current_index += 1

        if self.current_index >= len(self.questions):
            self.completed = True

        return record

    def generate_followup_question(
        self,
        previous_question: str,
        student_answer: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Generate a realistic follow-up probing question from the examiner based on the student's answer.
        """
        active_key = api_key or self.api_key
        if not active_key:
            return "Can you explain the core advantages and trade-offs of this approach?"

        try:
            retrieval = self.retriever.get_context_and_sources(
                query=f"{previous_question} {student_answer}", k=3, api_key=active_key
            )
            context = retrieval.get("context", "")

            prompt_content = f"""Study Material:
{context}

Examiner Question Asked: {previous_question}
Student's Response: {student_answer}

As the viva examiner, acknowledge the student's response in 1 brief phrase and ask 1 focused, probing follow-up question to test their deeper conceptual understanding. Keep your entire response under 2 sentences."""

            messages = [
                SystemMessage(content=MOCK_EXAMINER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]

            followup = invoke_llm(
                messages=messages,
                model_name=model_name or config.get_active_model(),
                temperature=0.4,
                api_key=active_key,
            )
            return followup.strip()

        except Exception:
            return "Good. Can you give one real-world application or example where this concept is applied?"

    def get_final_report(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compile final scorecard from history.
        """
        return feedback_compiler.generate_final_report(
            session_evaluations=self.history,
            api_key=api_key or self.api_key,
            model_name=model_name,
        )
