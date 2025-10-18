"""Utilities for extracting text content from files."""

from __future__ import annotations

import os

from .constants import TEXT_EXTENSIONS

__all__ = ["TEXT_EXTENSIONS", "read_text_from_file"]


def read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        if ext == ".pdf":
            try:
                import pdfplumber  # type: ignore
            except Exception:
                return ""
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        if ext == ".docx":
            try:
                import docx  # type: ignore
            except Exception:
                return ""
            try:
                document = docx.Document(path)
                return "\n".join(paragraph.text for paragraph in document.paragraphs)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""
