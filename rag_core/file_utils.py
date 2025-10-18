"""File system helpers for media discovery."""

from __future__ import annotations

import os
from typing import Iterable

from .constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

__all__ = ["is_image", "is_video", "walk_media", "load_image"]

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow must be installed to load images") from exc


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def walk_media(folder: str) -> Iterable[str]:
    for root, _, files in os.walk(folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                yield os.path.join(root, filename)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
