"""
Question Generator module for producing grounded viva questions across difficulties
(Easy, Medium, Hard, Mixed) using RAG context and GPT-5.6 Luna.
"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
import config
from rag.retriever import VivaRetriever
from rag.faiss_store import faiss_manager
from llm.llm_service import invoke_llm
from llm.prompts import (
    QUESTION_GENERATOR_SYSTEM_PROMPT,
    QUESTION_GENERATOR_USER_TEMPLATE,
)


def extract_json_from_response(text: str) -> Any:
    """
    Safely parse JSON from LLM response even if wrapped in markdown code blocks.
    """
    text = text.strip()
    # Remove markdown ```json ... ``` fences if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding the outermost JSON array or object
        array_match = re.search(r"(\[.*\])", text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(1))
            except json.JSONDecodeError:
                pass
        obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse valid JSON from LLM response: {text[:200]}...")


class VivaQuestionGenerator:
    """
    Generates college viva questions grounded in uploaded study material.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.retriever = VivaRetriever(top_k=8, api_key=api_key)

    def generate_questions(
        self,
        subject: str = "Computer Science / Machine Learning",
        topic: str = "General",
        difficulty: str = "Medium",
        num_questions: int = 5,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant study material context for the given topic/subject,
        and generate structured viva questions with model answers and key points.
        """
        active_key = api_key or config.get_openai_api_key(self.api_key)
        if not active_key:
            return {
                "success": False,
                "error": "OpenAI API Key is missing. Please configure it in Settings or .env.",
                "questions": [],
            }

        if not faiss_manager.is_index_present():
            return {
                "success": False,
                "error": "Knowledge base is empty. Please upload study materials in the Upload tab first.",
                "questions": [],
            }

        # Retrieve context for topic/subject
        query = f"{subject} {topic}".strip()
        try:
            try:
                retrieval = self.retriever.get_context_and_sources(
                    query=query, k=8, api_key=active_key
                )
                context = retrieval.get("context", "")
                sources = retrieval.get("sources", [])
            except Exception as ret_err:
                err_lower = str(ret_err).lower()
                if "credit_balance_exhausted" in err_lower or "insufficient_quota" in err_lower or "429" in err_lower:
                    from rag.embeddings import LocalOfflineEmbeddings
                    self.retriever.api_key = None
                    retrieval = self.retriever.get_context_and_sources(query=query, k=8)
                    context = retrieval.get("context", "")
                    sources = retrieval.get("sources", [])
                else:
                    raise ret_err

            if not context.strip():
                return {
                    "success": False,
                    "error": "No relevant study material found for the specified topic.",
                    "questions": [],
                }

            user_prompt = QUESTION_GENERATOR_USER_TEMPLATE.format(
                context=context,
                subject=subject,
                topic=topic,
                difficulty=difficulty,
                num_questions=num_questions,
            )

            messages = [
                SystemMessage(content=QUESTION_GENERATOR_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response_text = invoke_llm(
                messages=messages,
                model_name=model_name or config.get_active_model(),
                temperature=0.4,
                api_key=active_key,
            )

            parsed_questions = extract_json_from_response(response_text)
            if not isinstance(parsed_questions, list):
                if isinstance(parsed_questions, dict) and "questions" in parsed_questions:
                    parsed_questions = parsed_questions["questions"]
                else:
                    parsed_questions = [parsed_questions]

            # Normalize output
            cleaned_questions = []
            for idx, q in enumerate(parsed_questions):
                cleaned_questions.append({
                    "id": idx + 1,
                    "question": q.get("question", f"Question {idx+1}"),
                    "difficulty": q.get("difficulty", difficulty),
                    "topic": q.get("topic", topic),
                    "ideal_answer": q.get("ideal_answer", ""),
                    "key_points": q.get("key_points", []),
                })

            return {
                "success": True,
                "questions": cleaned_questions,
                "sources": sources,
                "subject": subject,
                "topic": topic,
                "difficulty": difficulty,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate viva questions: {str(e)}",
                "questions": [],
            }


# Global instance
question_generator = VivaQuestionGenerator()
