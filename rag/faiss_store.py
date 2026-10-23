"""
FAISS Vector Store Manager.
Handles local FAISS index creation, persistence, incremental updates,
similarity search, index clearing, and status diagnostics.
No ChromaDB, SQLite, or external databases are used.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import config
from rag.embeddings import get_embedding_function, LocalOfflineEmbeddings


class FAISSStoreManager:
    """
    Manager for the local FAISS vector store.
    Saves index to `vectorstore/index.faiss` and `vectorstore/index.pkl`.
    """

    def __init__(self, vectorstore_dir: Optional[Path] = None):
        self.vectorstore_dir = vectorstore_dir or config.VECTORSTORE_DIR
        self.index_name = "index"
        self._vectorstore: Optional[FAISS] = None

    def is_index_present(self) -> bool:
        """
        Check if FAISS index files already exist on disk.
        """
        faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
        pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
        return faiss_file.exists() and pkl_file.exists()

    def create_index(
        self,
        documents: List[Document],
        embeddings=None,
        api_key: Optional[str] = None,
    ) -> FAISS:
        """
        Create a new FAISS index from documents and persist it to disk.
        Automatically uses local embeddings if OpenAI quota is unavailable.
        """
        if not documents:
            raise ValueError("No documents provided to create FAISS index.")

        embed_fn = embeddings or get_embedding_function(api_key=api_key)
        try:
            self._vectorstore = FAISS.from_documents(documents, embed_fn)
        except Exception as e:
            # If any failure occurs with remote embeddings, fallback to local offline
            print(f"Embedding generation with primary model failed ({e}). Switching to local offline model...")
            self._vectorstore = FAISS.from_documents(documents, LocalOfflineEmbeddings())

        self.save_index()
        return self._vectorstore

    def save_index(self, folder_path: Optional[Path] = None) -> None:
        """
        Save the current in-memory FAISS index to disk.
        """
        if self._vectorstore is None:
            raise ValueError("No active FAISS index to save.")

        save_dir = folder_path or self.vectorstore_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(save_dir), index_name=self.index_name)

    def load_index(
        self,
        folder_path: Optional[Path] = None,
        embeddings=None,
        api_key: Optional[str] = None,
    ) -> Optional[FAISS]:
        """
        Load FAISS index from disk. Returns None if files do not exist.
        """
        load_dir = folder_path or self.vectorstore_dir
        faiss_file = load_dir / f"{self.index_name}.faiss"
        pkl_file = load_dir / f"{self.index_name}.pkl"

        if not (faiss_file.exists() and pkl_file.exists()):
            return None

        embed_fn = embeddings or get_embedding_function(api_key=api_key)
        try:
            self._vectorstore = FAISS.load_local(
                folder_path=str(load_dir),
                embeddings=embed_fn,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            # Fallback to local offline embedding
            self._vectorstore = FAISS.load_local(
                folder_path=str(load_dir),
                embeddings=LocalOfflineEmbeddings(),
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
        return self._vectorstore

    def get_or_load_index(
        self,
        embeddings=None,
        api_key: Optional[str] = None,
    ) -> Optional[FAISS]:
        """
        Get the in-memory index or load from disk if available.
        """
        if self._vectorstore is not None:
            return self._vectorstore
        return self.load_index(embeddings=embeddings, api_key=api_key)

    def add_documents(
        self,
        documents: List[Document],
        embeddings=None,
        api_key: Optional[str] = None,
    ) -> FAISS:
        """
        Add documents to an existing FAISS index (or create one if none exists).
        """
        if not documents:
            return self._vectorstore

        if self._vectorstore is None:
            if self.is_index_present():
                self.load_index(embeddings=embeddings, api_key=api_key)
            else:
                return self.create_index(documents, embeddings=embeddings, api_key=api_key)

        try:
            self._vectorstore.add_documents(documents)
        except Exception:
            # If add fails with remote embedder, re-index with local offline embeddings
            all_existing = self.get_all_documents() + documents
            return self.create_index(all_existing, embeddings=LocalOfflineEmbeddings())

        self.save_index()
        return self._vectorstore

    def get_all_documents(self) -> List[Document]:
        """
        Extract all documents currently stored in the FAISS index docstore.
        """
        if self._vectorstore is None or not hasattr(self._vectorstore, "docstore"):
            return []
        try:
            return list(self._vectorstore.docstore._dict.values())
        except Exception:
            return []

    def update_index(
        self,
        new_documents: List[Document],
        embeddings=None,
        api_key: Optional[str] = None,
    ) -> FAISS:
        """
        Alias for adding new documents and saving updated index.
        """
        return self.add_documents(new_documents, embeddings=embeddings, api_key=api_key)

    def search_index(
        self,
        query: str,
        k: int = None,
        embeddings=None,
        api_key: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with distance scores.
        Returns list of (Document, score) tuples.
        """
        top_k = k or config.TOP_K
        store = self.get_or_load_index(embeddings=embeddings, api_key=api_key)

        if store is None:
            raise ValueError("Knowledge base is empty. Please upload study materials first.")

        try:
            results = store.similarity_search_with_score(query, k=top_k)
        except Exception:
            # Fallback search using local offline embeddings
            self._vectorstore = self.load_index(embeddings=LocalOfflineEmbeddings())
            results = self._vectorstore.similarity_search_with_score(query, k=top_k)

        return results

    def clear_index(self, folder_path: Optional[Path] = None) -> bool:
        """
        Remove local FAISS index files and reset in-memory vectorstore.
        """
        target_dir = folder_path or self.vectorstore_dir
        self._vectorstore = None

        cleared = False
        for fname in [f"{self.index_name}.faiss", f"{self.index_name}.pkl"]:
            fpath = target_dir / fname
            if fpath.exists():
                try:
                    fpath.unlink()
                    cleared = True
                except Exception:
                    pass

        return cleared

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic stats about the local FAISS index.
        """
        is_ready = self.is_index_present()
        faiss_size_kb = 0.0
        pkl_size_kb = 0.0

        if is_ready:
            f_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            p_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            if f_file.exists():
                faiss_size_kb = round(f_file.stat().st_size / 1024, 2)
            if p_file.exists():
                pkl_size_kb = round(p_file.stat().st_size / 1024, 2)

        doc_count = 0
        if self._vectorstore is not None and hasattr(self._vectorstore, "index"):
            doc_count = self._vectorstore.index.ntotal

        return {
            "index_exists": is_ready,
            "total_vectors": doc_count,
            "faiss_file_size_kb": faiss_size_kb,
            "pkl_file_size_kb": pkl_size_kb,
            "storage_path": str(self.vectorstore_dir),
        }


# Global instance for shared access across app
faiss_manager = FAISSStoreManager()
