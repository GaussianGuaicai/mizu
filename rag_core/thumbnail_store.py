"""Thumbnail disk caching utilities."""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
from PIL import Image

try:  # Optional OpenCV import for smoother resizing
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

__all__ = ["ThumbnailStore"]


class ThumbnailStore:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        try:
            os.makedirs(self.root_dir, exist_ok=True)
        except Exception:
            pass

    def path_for(self, source_path: str, *, extra: Optional[str] = None) -> str:
        base = os.path.abspath(source_path)
        if extra is not None:
            base = f"{base}::{extra}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
        subdir = os.path.join(self.root_dir, digest[:2])
        try:
            os.makedirs(subdir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(subdir, f"{digest}.jpg")

    def save(self, image: Image.Image, dest_path: str) -> str:
        if not dest_path:
            return dest_path
        if os.path.exists(dest_path):
            return dest_path
        try:
            image.save(dest_path, format="JPEG", quality=85, optimize=True)
            return dest_path
        except Exception:
            return ""

    def resize_for_storage(self, image: Image.Image, max_size: int = 256) -> np.ndarray:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            arr = np.asarray(image)
        except Exception:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        height, width = arr.shape[:2]
        if height == 0 or width == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if max(height, width) <= max_size:
            return arr

        scale = max_size / float(max(height, width))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))

        if cv2 is not None:
            try:
                resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if resized.dtype != np.uint8:
                    resized = resized.astype(np.uint8)
                return resized
            except Exception:
                pass

        try:
            pil_resized = Image.fromarray(arr).resize((new_w, new_h))
            return np.asarray(pil_resized)
        except Exception:
            return arr
