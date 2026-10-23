"""
Configuration module for the AI Viva Preparation Assistant.
Handles environment variables, default model settings, chunking parameters,
and directory paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent

# Load .env from project root if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ---------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_documents"
SAMPLE_DIR = DATA_DIR / "sample_materials"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# FAISS Storage File Paths
FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
FAISS_PKL_PATH = VECTORSTORE_DIR / "index.pkl"

# ---------------------------------------------------------
# Default OpenAI & RAG Settings
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
FALLBACK_OPENAI_MODEL = os.getenv("FALLBACK_OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# RAG Chunking Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 5))

# Viva Exam Configuration Defaults
DEFAULT_MOCK_QUESTIONS_COUNT = 5
SUPPORTED_DIFFICULTIES = ["Easy", "Medium", "Hard", "Mixed"]
SUPPORTED_FILE_EXTENSIONS = [".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt"]


def get_openai_api_key(session_key: str = None) -> str:
    """
    Retrieve the active OpenAI API key, prioritizing session state / runtime override
    over the .env variable.
    """
    if session_key and session_key.strip():
        return session_key.strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_active_model(session_model: str = None) -> str:
    """
    Retrieve active LLM model name.
    """
    if session_model and session_model.strip():
        return session_model.strip()
    return os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()


def get_active_embedding_model(session_embedding: str = None) -> str:
    """
    Retrieve active embedding model name.
    """
    if session_embedding and session_embedding.strip():
        return session_embedding.strip()
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
