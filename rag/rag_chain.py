"""
RAG Chain module that executes the end-to-end question answering pipeline:
Question -> FAISS Retrieval -> Context Assembly -> LLM Grounded Answer -> Source Citations.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
import config
from rag.retriever import VivaRetriever
from rag.faiss_store import faiss_manager
from llm.llm_service import invoke_llm
from llm.prompts import RAG_CHAT_SYSTEM_PROMPT, RAG_CHAT_USER_TEMPLATE


class VivaRAGChain:
    """
    Orchestrates RAG answering for viva questions.
    Ensures answers are grounded in retrieved context and attributes sources.
    """

    def __init__(self, top_k: Optional[int] = None, api_key: Optional[str] = None):
        self.top_k = top_k or config.TOP_K
        self.api_key = api_key
        self.retriever = VivaRetriever(top_k=self.top_k, api_key=self.api_key)

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute RAG retrieval and generate grounded viva answer.
        """
        active_key = api_key or config.get_openai_api_key(self.api_key)
        if not active_key:
            return {
                "answer": "⚠️ **OpenAI API Key Missing**: Please set your `OPENAI_API_KEY` in the `.env` file or enter it in the **⚙ Settings** page to start chatting.",
                "sources": [],
                "has_context": False,
                "error": "API_KEY_MISSING",
            }

        if not faiss_manager.is_index_present():
            return {
                "answer": "📚 **No Study Material Found**: Please upload your lecture notes, textbooks, or slides in the **📄 Upload Material** tab first so I can build your RAG knowledge base.",
                "sources": [],
                "has_context": False,
                "error": "INDEX_NOT_FOUND",
            }

        # Step 1 & 2: Similarity search and context extraction
        try:
            try:
                retrieval_res = self.retriever.get_context_and_sources(
                    query=question, k=top_k, api_key=active_key
                )
            except Exception as r_err:
                err_lower = str(r_err).lower()
                if "credit_balance_exhausted" in err_lower or "insufficient_quota" in err_lower or "429" in err_lower:
                    self.retriever.api_key = None
                    retrieval_res = self.retriever.get_context_and_sources(query=question, k=top_k)
                else:
                    raise r_err
        except Exception as e:
            return {
                "answer": f"⚠️ **Retrieval Error**: Failed to search the FAISS vector index. Details: {str(e)}",
                "sources": [],
                "has_context": False,
                "error": str(e),
            }

        context = retrieval_res.get("context", "")
        sources = retrieval_res.get("sources", [])

        if not context.strip():
            return {
                "answer": "The provided study material does not contain relevant information to answer this question. Please upload relevant notes or refine your query.",
                "sources": [],
                "has_context": False,
                "error": None,
            }

        # Step 3: Format Prompt
        user_message_content = RAG_CHAT_USER_TEMPLATE.format(
            context=context,
            question=question,
        )

        messages = [
            SystemMessage(content=RAG_CHAT_SYSTEM_PROMPT),
            HumanMessage(content=user_message_content),
        ]

        # Step 4: Invoke LLM (GPT-5.6 Luna / Fallback)
        try:
            llm_response = invoke_llm(
                messages=messages,
                model_name=model_name or config.get_active_model(),
                temperature=0.2,
                api_key=active_key,
            )

            return {
                "answer": llm_response,
                "sources": sources,
                "has_context": True,
                "error": None,
            }

        except Exception as e:
            return {
                "answer": f"⚠️ **LLM Generation Error**: {str(e)}",
                "sources": sources,
                "has_context": True,
                "error": str(e),
            }


# Global default instance
rag_chain = VivaRAGChain()
