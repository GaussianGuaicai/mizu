"""Hashing helpers."""

from __future__ import annotations

import hashlib
import os

__all__ = ["fast_file_hash"]


def fast_file_hash(path: str, sample_size: int = 8192) -> str:
    """Return a fast Blake2b fingerprint for *path*.

    The function reads file size, mtime, and small head/tail byte samples.
    It is intentionally lightweight to support quick change detection for
    large media files where full-file hashing would be costly.
    """
    hasher = hashlib.blake2b(digest_size=20)
    try:
        stat = os.stat(path)
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
        with open(path, "rb") as fh:
            head = fh.read(sample_size)
            hasher.update(head)
            if stat.st_size > sample_size:
                try:
                    fh.seek(max(0, stat.st_size - sample_size))
                    tail = fh.read(sample_size)
                    hasher.update(tail)
                except Exception:
                    pass
    except Exception:
        return ""
    return hasher.hexdigest()
