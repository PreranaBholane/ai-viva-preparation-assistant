"""
Retriever module for semantic similarity retrieval from FAISS.
Formats context blocks for LLM prompt injection and parses source metadata.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from rag.faiss_store import faiss_manager
import config


class VivaRetriever:
    """
    RAG Retriever that queries the FAISS index and returns both raw documents
    and formatted context + structured source references.
    """

    def __init__(self, top_k: int = None, api_key: Optional[str] = None):
        self.top_k = top_k or config.TOP_K
        self.api_key = api_key

    def retrieve_relevant_chunks(
        self,
        query: str,
        k: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Query FAISS vector store and return list of (Document, score) tuples.
        """
        active_k = k or self.top_k
        active_key = api_key or self.api_key
        return faiss_manager.search_index(query=query, k=active_k, api_key=active_key)

    def get_context_and_sources(
        self,
        query: str,
        k: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves top relevant chunks, builds a unified context string for the prompt,
        and constructs structured source citations for Streamlit UI display.
        """
        results = self.retrieve_relevant_chunks(query=query, k=k, api_key=api_key)

        if not results:
            return {
                "context": "",
                "sources": [],
                "has_context": False,
            }

        context_blocks = []
        sources = []

        for idx, (doc, score) in enumerate(results):
            meta = doc.metadata or {}
            source_file = meta.get("source", "Unknown Document")
            page_num = meta.get("page", 1)
            topic = meta.get("topic", "")
            file_type = meta.get("file_type", "DOC")

            # Context block for LLM prompt
            header = f"[Source #{idx+1}: {source_file} | Page/Slide: {page_num}"
            if topic:
                header += f" | Section: {topic}"
            header += "]"

            context_blocks.append(f"{header}\n{doc.page_content.strip()}")

            # Structured citation for UI
            sources.append({
                "index": idx + 1,
                "filename": source_file,
                "page": page_num,
                "file_type": file_type,
                "topic": topic,
                "similarity_score": round(float(score), 4),
                "content_snippet": doc.page_content.strip(),
            })

        unified_context = "\n\n---\n\n".join(context_blocks)

        return {
            "context": unified_context,
            "sources": sources,
            "has_context": bool(unified_context.strip()),
            "total_retrieved": len(results),
        }
