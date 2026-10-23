"""
Document Processor module for cleaning, normalizing, and enriching text documents
prior to chunking and vector embedding.
"""

import re
from typing import List
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean and normalize raw extracted document text.
    - Removes non-printable control characters (except newline, tab).
    - Normalizes multiple whitespace characters.
    - Fixes hyphenated words broken across line breaks.
    - Trims leading/trailing whitespace.
    """
    if not text:
        return ""

    # Replace carriage returns with standard newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin words split by hyphen at end of line (e.g., "algo-\nrithm" -> "algorithm")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove non-printable control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Collapse multiple consecutive blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple horizontal spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def process_documents(raw_documents: List[Document]) -> List[Document]:
    """
    Process a list of raw extracted Documents:
    - Cleans document content
    - Filters out empty or trivial chunks
    - Calculates word count and character count metadata
    """
    processed_docs = []

    for idx, doc in enumerate(raw_documents):
        cleaned = clean_text(doc.page_content)
        if len(cleaned) < 20:  # Skip trivial empty or very short artifacts
            continue

        meta = dict(doc.metadata)
        meta["char_count"] = len(cleaned)
        meta["word_count"] = len(cleaned.split())
        meta["doc_index"] = idx

        processed_docs.append(Document(page_content=cleaned, metadata=meta))

    return processed_docs
