"""
File utilities for saving, reading, clearing, and validating study material documents.
"""

import shutil
from pathlib import Path
from typing import List, Optional
import config


def validate_file_extension(filename: str) -> bool:
    """
    Check if the file extension is supported (PDF, PPT, PPTX, DOC, DOCX, TXT).
    """
    ext = Path(filename).suffix.lower()
    return ext in config.SUPPORTED_FILE_EXTENSIONS


def save_single_uploaded_file(
    uploaded_file, target_dir: Optional[Path] = None, replace_existing: bool = True
) -> Optional[Path]:
    """
    Save a single uploaded file. If replace_existing is True (default),
    clears any previously uploaded documents so only the new document is processed.
    """
    save_path = target_dir or config.UPLOAD_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    if not validate_file_extension(uploaded_file.name):
        return None

    if replace_existing:
        clear_uploaded_documents(save_path)

    file_dest = save_path / uploaded_file.name
    with open(file_dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_dest


def save_uploaded_files(
    uploaded_files, target_dir: Optional[Path] = None, replace_existing: bool = False
) -> List[Path]:
    """
    Save Streamlit uploaded files to the target storage folder.
    """
    save_path = target_dir or config.UPLOAD_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    if replace_existing:
        clear_uploaded_documents(save_path)

    saved_paths = []
    for uploaded_file in uploaded_files:
        if not validate_file_extension(uploaded_file.name):
            continue

        file_dest = save_path / uploaded_file.name
        with open(file_dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_dest)

    return saved_paths


def get_stored_documents(target_dir: Optional[Path] = None) -> List[Path]:
    """
    List all supported documents currently in the upload directory.
    """
    scan_dir = target_dir or config.UPLOAD_DIR
    if not scan_dir.exists():
        return []

    files = []
    for item in scan_dir.iterdir():
        if item.is_file() and validate_file_extension(item.name):
            files.append(item)
    return sorted(files, key=lambda p: p.name)


def clear_uploaded_documents(target_dir: Optional[Path] = None) -> int:
    """
    Remove all uploaded files from storage directory.
    """
    scan_dir = target_dir or config.UPLOAD_DIR
    if not scan_dir.exists():
        return 0

    count = 0
    for item in scan_dir.iterdir():
        if item.is_file() and item.name != ".gitkeep":
            try:
                item.unlink()
                count += 1
            except Exception:
                pass
    return count


def load_sample_material_to_uploads(
    sample_dir: Optional[Path] = None,
    upload_dir: Optional[Path] = None,
    single_file: bool = True,
) -> List[Path]:
    """
    Copy pre-packaged sample study notes into the upload directory for one-click testing.
    If single_file is True (default), clears any existing files and copies just one sample document.
    """
    s_dir = sample_dir or config.SAMPLE_DIR
    u_dir = upload_dir or config.UPLOAD_DIR
    u_dir.mkdir(parents=True, exist_ok=True)

    if single_file:
        clear_uploaded_documents(u_dir)

    copied = []
    if s_dir.exists():
        # Prefer PDF if available, otherwise first supported sample file
        preferred_order = [
            "Machine_Learning_Viva_Notes.pdf",
            "Machine_Learning_Viva_Notes.docx",
            "Machine_Learning_Viva_Notes.txt",
            "Machine_Learning_Viva_Slides.pptx",
        ]
        sample_map = {item.name: item for item in s_dir.iterdir() if item.is_file() and validate_file_extension(item.name)}

        if single_file:
            for name in preferred_order:
                if name in sample_map:
                    dest = u_dir / name
                    shutil.copy2(sample_map[name], dest)
                    copied.append(dest)
                    return copied

        for item in s_dir.iterdir():
            if item.is_file() and validate_file_extension(item.name):
                dest = u_dir / item.name
                shutil.copy2(item, dest)
                copied.append(dest)
    return copied


def format_file_size(size_bytes: int) -> str:
    """
    Format bytes to human readable string (KB, MB).
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{round(size_bytes / 1024, 1)} KB"
    else:
        return f"{round(size_bytes / (1024 * 1024), 2)} MB"
