"""
Document Loader module for extracting text and metadata from PDF, PPT/PPTX, DOC/DOCX, and TXT files.
Uses:
- PyMuPDF (fitz) for PDF
- python-pptx for PPT/PPTX
- python-docx for DOC/DOCX
- Built-in file handling for TXT
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document


def load_pdf(file_path: Path) -> List[Document]:
    """
    Extract text from PDF file page by page using PyMuPDF.
    """
    import pymupdf as fitz

    documents = []
    doc = fitz.open(str(file_path))
    filename = file_path.name

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_idx + 1,
                        "file_type": "PDF",
                        "total_pages": len(doc),
                        "topic": filename.rsplit(".", 1)[0].replace("_", " "),
                    },
                )
            )
    doc.close()
    return documents


def load_pptx(file_path: Path) -> List[Document]:
    """
    Extract text from PPT/PPTX file slide by slide using python-pptx.
    """
    from pptx import Presentation

    documents = []
    prs = Presentation(str(file_path))
    filename = file_path.name
    total_slides = len(prs.slides)

    for slide_idx, slide in enumerate(prs.slides):
        slide_text_parts = []
        slide_title = ""

        # Extract title if present
        if slide.shapes.title and slide.shapes.title.text:
            slide_title = slide.shapes.title.text.strip()
            slide_text_parts.append(f"Slide Title: {slide_title}")

        # Extract all shape texts and tables
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    para_text = paragraph.text.strip()
                    if para_text and para_text != slide_title:
                        slide_text_parts.append(para_text)
            elif shape.has_table:
                table_rows = []
                for row in shape.table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        table_rows.append(row_text)
                if table_rows:
                    slide_text_parts.append("\n".join(table_rows))

        full_slide_text = "\n".join(slide_text_parts).strip()
        if full_slide_text:
            documents.append(
                Document(
                    page_content=full_slide_text,
                    metadata={
                        "source": filename,
                        "page": slide_idx + 1,  # Slide number as page
                        "file_type": "PPTX",
                        "total_pages": total_slides,
                        "topic": slide_title or filename.rsplit(".", 1)[0].replace("_", " "),
                    },
                )
            )
    return documents


def load_docx(file_path: Path) -> List[Document]:
    """
    Extract text from DOC/DOCX file using python-docx.
    Groups content logically with paragraph and table extraction.
    """
    import docx

    documents = []
    doc = docx.Document(str(file_path))
    filename = file_path.name

    current_section = filename.rsplit(".", 1)[0].replace("_", " ")
    buffer_text = []
    page_approx = 1
    char_count = 0

    for para in doc.paragraphs:
        p_text = para.text.strip()
        if not p_text:
            continue

        # Detect heading as topic transition
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            current_section = p_text

        buffer_text.append(p_text)
        char_count += len(p_text)

        # Approximate a page boundary every ~2000 characters
        if char_count > 2000:
            documents.append(
                Document(
                    page_content="\n".join(buffer_text),
                    metadata={
                        "source": filename,
                        "page": page_approx,
                        "file_type": "DOCX",
                        "topic": current_section,
                    },
                )
            )
            buffer_text = []
            char_count = 0
            page_approx += 1

    # Extract tables
    for table in doc.tables:
        table_lines = []
        for row in table.rows:
            row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_vals:
                table_lines.append(" | ".join(row_vals))
        if table_lines:
            buffer_text.append("Table Data:\n" + "\n".join(table_lines))

    if buffer_text:
        documents.append(
            Document(
                page_content="\n".join(buffer_text),
                metadata={
                    "source": filename,
                    "page": page_approx,
                    "file_type": "DOCX",
                    "topic": current_section,
                },
            )
        )

    return documents


def load_txt(file_path: Path) -> List[Document]:
    """
    Extract text from TXT files with multi-encoding fallback.
    """
    filename = file_path.name
    text = ""
    encodings = ["utf-8", "latin-1", "cp1252", "utf-16"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if not text.strip():
        return []

    # If text has clear markdown headers (e.g., # UNIT or ## Topic), split by sections
    sections = text.split("\n# ")
    documents = []

    if len(sections) > 1:
        for idx, sec in enumerate(sections):
            sec_text = sec.strip()
            if not sec_text:
                continue
            if idx > 0:
                sec_text = "# " + sec_text

            first_line = sec_text.split("\n", 1)[0].replace("#", "").strip()
            documents.append(
                Document(
                    page_content=sec_text,
                    metadata={
                        "source": filename,
                        "page": idx + 1,
                        "file_type": "TXT",
                        "topic": first_line or filename.rsplit(".", 1)[0].replace("_", " "),
                    },
                )
            )
    else:
        documents.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "source": filename,
                    "page": 1,
                    "file_type": "TXT",
                    "topic": filename.rsplit(".", 1)[0].replace("_", " "),
                },
            )
        )

    return documents


def load_single_document(file_path: Path) -> List[Document]:
    """
    Route document file to appropriate loader based on file extension.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix in [".pptx", ".ppt"]:
        return load_pptx(file_path)
    elif suffix in [".docx", ".doc"]:
        return load_docx(file_path)
    elif suffix == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: PDF, PPTX, DOCX, TXT")


def load_multiple_documents(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Load multiple documents and collect extraction summary stats.
    """
    all_documents = []
    file_summaries = []
    errors = []

    for path in file_paths:
        try:
            docs = load_single_document(path)
            all_documents.extend(docs)
            file_summaries.append({
                "filename": path.name,
                "file_type": path.suffix.upper().replace(".", ""),
                "pages_detected": len(docs),
                "total_chars": sum(len(d.page_content) for d in docs),
                "status": "Success",
            })
        except Exception as e:
            errors.append({"filename": path.name, "error": str(e)})
            file_summaries.append({
                "filename": path.name,
                "file_type": path.suffix.upper().replace(".", ""),
                "pages_detected": 0,
                "total_chars": 0,
                "status": f"Error: {str(e)}",
            })

    return {
        "documents": all_documents,
        "file_summaries": file_summaries,
        "errors": errors,
    }
