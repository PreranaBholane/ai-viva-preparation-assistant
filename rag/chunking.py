"""
Chunking module for splitting processed documents into semantically coherent chunks
using RecursiveCharacterTextSplitter.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split processed documents into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves and enriches metadata for each chunk.
    """
    c_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)

    # Enrich metadata for each chunk
    for i, chunk in enumerate(split_docs):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}_p{chunk.metadata.get('page', 1)}_c{i}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_chars"] = len(chunk.page_content)

    return split_docs


def get_chunk_statistics(chunks: List[Document]) -> Dict[str, Any]:
    """
    Compute statistical distribution of generated chunks.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "sources_count": 0,
        }

    sizes = [len(c.page_content) for c in chunks]
    sources = set(c.metadata.get("source", "unknown") for c in chunks)

    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": round(sum(sizes) / len(sizes), 1),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "sources_count": len(sources),
        "unique_sources": list(sources),
    }
