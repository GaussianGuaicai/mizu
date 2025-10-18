"""Backwards-compatibility layer for the refactored code layout."""

from __future__ import annotations

from rag_core.chroma import ChromaRAG
from rag_core.file_utils import is_image, is_video, load_image, walk_media
from rag_core.hashing import fast_file_hash
from rag_core.text_utils import TEXT_EXTENSIONS, read_text_from_file

_read_text_from_file = read_text_from_file

__all__ = [
    "ChromaRAG",
    "fast_file_hash",
    "is_image",
    "is_video",
    "load_image",
    "walk_media",
    "TEXT_EXTENSIONS",
    "read_text_from_file",
    "_read_text_from_file",
]
