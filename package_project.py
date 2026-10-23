"""
Project Packaging Script.
Creates clean 'AI-Viva-Preparation-Assistant.zip' excluding caches, virtual environments,
.env secrets, and temporary files.
"""

import os
import zipfile
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent
ZIP_NAME = "AI-Viva-Preparation-Assistant.zip"
ZIP_PATH = ROOT_DIR / ZIP_NAME

# Exclusion rules
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "venv",
    "env",
    ".venv",
    ".idea",
    ".vscode",
    "vectorstore_test",
}

EXCLUDE_FILES = {
    ".env",
    ".env.local",
    "AI-Viva-Preparation-Assistant.zip",
    "package_project.py",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
}


def package_project():
    print(f"Creating package: {ZIP_PATH}")
    added_count = 0

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(ROOT_DIR):
            # Modify dirs in-place to avoid recursing into excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and (not d.startswith(".") or d == ".streamlit")]

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(ROOT_DIR)

                # Check exclusions
                if file in EXCLUDE_FILES:
                    continue
                if file_path.suffix in EXCLUDE_EXTENSIONS:
                    continue
                if any(part in EXCLUDE_DIRS for part in rel_path.parts):
                    continue
                # Do not package uploaded documents except .gitkeep
                if "uploaded_documents" in rel_path.parts and file != ".gitkeep":
                    continue

                # Add file inside the archive under top-level folder 'AI-Viva-Preparation-Assistant'
                archive_name = Path("AI-Viva-Preparation-Assistant") / rel_path
                zipf.write(file_path, str(archive_name))
                print(f"  + Added: {archive_name}")
                added_count += 1

    zip_size_kb = round(ZIP_PATH.stat().st_size / 1024, 2)
    print(f"\nSuccessfully created {ZIP_NAME} ({zip_size_kb} KB) with {added_count} files.")


if __name__ == "__main__":
    package_project()
