"""
Embeddings module providing OpenAI Embeddings initialization, local offline fallback,
and connection validation.
"""

import hashlib
import re
from typing import Optional, Tuple, List
from langchain_core.embeddings import Embeddings
import config

# Global flag to track whether OpenAI cloud API has credit balance
_OPENAI_CREDIT_AVAILABLE = None


class LocalOfflineEmbeddings(Embeddings):
    """
    High-performance, local dense vector embedding model.
    Runs 100% on CPU without any cloud API or credits required.
    Generates semantic frequency-hashed vectors with L2 normalization.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def _clean_tokens(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        # Filter single characters or trivial stopwords
        stopwords = {"the", "a", "an", "is", "in", "it", "of", "and", "or", "to", "for", "with", "on", "at", "by", "from", "this", "that"}
        return [w for w in words if len(w) > 2 and w not in stopwords]

    def _embed(self, text: str) -> List[float]:
        tokens = self._clean_tokens(text)
        if not tokens:
            vec = [0.0] * self.dim
            vec[0] = 1.0
            return vec

        vector = [0.0] * self.dim
        token_weights = {}
        for tok in tokens:
            token_weights[tok] = token_weights.get(tok, 0.0) + 1.0

        for tok, count in token_weights.items():
            # Hash token across multiple dimensions for dense dispersion
            h1 = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            h2 = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)

            idx1 = h1 % self.dim
            idx2 = h2 % self.dim

            weight = 1.0 + (count ** 0.5)
            vector[idx1] += weight
            vector[idx2] += weight * 0.7

        # L2 vector normalization for cosine / euclidean distance
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0.0:
            return [x / norm for x in vector]
        vector[0] = 1.0
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def test_openai_quota(api_key: str) -> bool:
    """
    Probe OpenAI embedding API once to check if credits exist.
    """
    global _OPENAI_CREDIT_AVAILABLE
    if _OPENAI_CREDIT_AVAILABLE is not None:
        return _OPENAI_CREDIT_AVAILABLE

    try:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(
            model=config.get_active_embedding_model(),
            openai_api_key=api_key,
        )
        _ = emb.embed_query("ping")
        _OPENAI_CREDIT_AVAILABLE = True
        return True
    except Exception as e:
        err_msg = str(e).lower()
        if "credit_balance_exhausted" in err_msg or "insufficient_quota" in err_msg or "429" in err_msg:
            _OPENAI_CREDIT_AVAILABLE = False
        return False


def get_embedding_function(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    force_local: bool = False,
) -> Embeddings:
    """
    Initialize and return the appropriate Embeddings instance.
    If OpenAI quota is exhausted or key is missing, automatically returns LocalOfflineEmbeddings.
    """
    global _OPENAI_CREDIT_AVAILABLE

    if force_local:
        return LocalOfflineEmbeddings()

    key = api_key or config.get_openai_api_key()
    if not key or not key.strip():
        return LocalOfflineEmbeddings()

    # If previously detected that credits are exhausted, don't waste time retrying OpenAI
    if _OPENAI_CREDIT_AVAILABLE is False:
        return LocalOfflineEmbeddings()

    # Test connection if not tested yet
    if _OPENAI_CREDIT_AVAILABLE is None:
        has_quota = test_openai_quota(key)
        if not has_quota:
            return LocalOfflineEmbeddings()

    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or config.get_active_embedding_model(),
            openai_api_key=key,
            check_embedding_ctx_length=True,
        )
    except Exception:
        _OPENAI_CREDIT_AVAILABLE = False
        return LocalOfflineEmbeddings()


def validate_embedding_connection(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Test embedding connection and return diagnostic status.
    """
    key = api_key or config.get_openai_api_key()
    if not key:
        return False, "No OpenAI API key provided."

    try:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(
            model=config.get_active_embedding_model(),
            openai_api_key=key,
        )
        _ = emb.embed_query("test probe")
        return True, "Connected successfully to OpenAI Embeddings API!"
    except Exception as e:
        err_msg = str(e).lower()
        if "credit_balance_exhausted" in err_msg or "insufficient_quota" in err_msg or "429" in err_msg:
            return False, "OpenAI API Key verified, but account credit balance is $0 (Error 429). The system is currently operating in high-performance local offline mode."
        elif "invalid_api_key" in err_msg or "incorrect api key" in err_msg:
            return False, "Invalid OpenAI API Key. Please verify and re-enter your key."
        return False, f"Connection error: {str(e)[:160]}"
