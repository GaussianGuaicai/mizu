from __future__ import annotations

import os
import io
import hashlib
import functools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence, Callable, Set

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .embeddings import EmbeddingModel, sample_video_frames
except ImportError:  # pragma: no cover - support running as a script module
    from embeddings import EmbeddingModel, sample_video_frames

# Optional OpenCV import for robust resizing
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from pi_heif import register_heif_opener
register_heif_opener()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".avif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def fast_file_hash(path: str, sample_size: int = 8192) -> str:
    """Compute a fast, lightweight hash for a file.

    Strategy: use Blake2b on a few small samples to keep it fast for large files.
    We include: file size + first and last `sample_size` bytes (if available).
    This is a pragmatic, performant fingerprint (not a cryptographic full-file
    integrity hash) suitable for change-detection and deduplication hints.
    Returns a hex string.
    """
    # Note: this is a pragmatic, fast fingerprint used for change-detection
    # and light-weight deduplication. It is NOT a full-file cryptographic
    # checksum like sha256 over the entire file. Using a small sample from the
    # head/tail + size and mtime keeps the operation fast for large media files.
    h = hashlib.blake2b(digest_size=20)
    try:
        st = os.stat(path)
        # incorporate size and mtime to further differentiate
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        with open(path, "rb") as f:
            # head
            head = f.read(sample_size)
            h.update(head)
            # tail
            if st.st_size > sample_size:
                try:
                    f.seek(max(0, st.st_size - sample_size))
                    tail = f.read(sample_size)
                    h.update(tail)
                except Exception:
                    pass
    except Exception:
        # fallback to empty string hash
        return ""
    return h.hexdigest()


@dataclass
class MediaDoc:
    id: str
    path: str
    kind: str  # 'image' | 'video'
    metadata: Dict


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def walk_media(folder: str) -> Iterable[str]:
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                yield p


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


TEXT_EXTS = {".txt", ".md", ".pdf", ".docx"}

def _read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == ".pdf":
            try:
                import pdfplumber  # type: ignore
            except Exception:
                return ""
            text_parts: List[str] = []
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
                doc = docx.Document(path)
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


class ChromaRAG:
    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "media",
        model_name: str = "jinaai/jina-embeddings-v4",
        quantize = True,
        enable_flash_atten: bool = False,
    ) -> None:
        self.persist_dir = os.path.abspath(persist_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # Directory for on-disk thumbnails (images & video frame previews)
        self.thumb_dir = os.path.join(self.persist_dir, "thumbnails")
        try:
            os.makedirs(self.thumb_dir, exist_ok=True)
        except Exception:
            pass  # If creation fails, later thumbnail saves will attempt again
        
        # Create separate collections
        self.col_text = self.client.get_or_create_collection(
            name=f"{collection_name}_text",
            metadata={"hnsw:space": "cosine"},
        )
        self.col_image = self.client.get_or_create_collection(
            name=f"{collection_name}_image",
            metadata={"hnsw:space": "cosine"},
        )
        self.col_video = self.client.get_or_create_collection(
            name=f"{collection_name}_video",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = EmbeddingModel(model_name,quantize=quantize,enable_flash_atten=enable_flash_atten)
        
        # Thumbnail storage strategy:
        # - Thumbnails (max 256px) are saved to disk under <persist_dir>/thumbnails/<sha1>.jpg
        # - Metadata stores both 'path' (original file) and 'thumb_path' (cached thumbnail)
        # - UI prefers thumb_path to avoid repeated loading/conversion of large originals (esp. HEIF/AVIF)
        # - For video frames, each frame gets its own thumbnail keyed by video_path + frame_index
    
    # Memory optimization notes:
    # - Image and text indexing are now streamed in batches (no full list retention).
    # - Video frame embeddings are chunked to avoid large GPU/CPU peaks.
    # - An async interface `upsert_folder_async` allows concurrent file I/O and batched embedding.
    #   Use: `await rag.upsert_folder_async(path, batch_size=8)`.

    # ---------------- Thumbnail disk storage helpers -----------------
    def _thumb_path_for(self, source_path: str, extra: str | None = None) -> str:
        """Deterministic thumbnail filepath for a given source path (and optional suffix)."""
        # Use absolute path to avoid collisions + optional extra identifier (e.g., frame index)
        base_key = f"{os.path.abspath(source_path)}"
        if extra is not None:
            base_key += f"::{extra}"
        h = hashlib.sha1(base_key.encode("utf-8")).hexdigest()
        # Bucket by first 2 chars to avoid huge single directory
        subdir = os.path.join(self.thumb_dir, h[:2])
        try:
            os.makedirs(subdir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(subdir, f"{h}.jpg")

    def _save_thumbnail(self, img: Image.Image, dest_path: str) -> str:
        """Save a PIL Image thumbnail to dest_path if missing. Returns path (existing or newly written)."""
        if not dest_path:
            return dest_path
        if os.path.exists(dest_path):
            return dest_path
        try:
            img.save(dest_path, format="JPEG", quality=85, optimize=True)
        except Exception:
            # Ignore save errors — caller can still fallback to original media path
            return ""
        return dest_path

    # ---------------- Thumbnail utilities -----------------
    def _resize_for_storage(self, img: Image.Image, max_size: int = 256) -> np.ndarray:
        """Return a resized copy (thumbnail) as a numpy ndarray (H,W,3 uint8) using cv2.

        Uses OpenCV for consistent resizing. Falls back to PIL or identity if cv2 isn't available.
        Always returns a uint8 ndarray; never None.
        """
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img)
        except Exception:
            # If conversion fails, return a 1x1 black pixel to avoid None
            return np.zeros((1, 1, 3), dtype=np.uint8)

        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # If already within bounds, return as-is
        if max(h, w) <= max_size:
            return arr

        # Compute new size preserving aspect ratio
        scale = max_size / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if cv2 is not None:
            try:
                resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if resized.dtype != np.uint8:
                    resized = resized.astype(np.uint8)
                return resized
            except Exception:
                pass  # fall back to PIL below

        # Fallback: PIL resize
        try:
            pil_resized = Image.fromarray(arr).resize((new_w, new_h))
            return np.asarray(pil_resized)
        except Exception:
            # Last resort: return original array
            return arr

    # ---------------- Internal utilities -----------------
    def _filter_new_ids(self, collection: Any, candidate_ids: List[str], chunk_size: int = 256) -> Tuple[List[str], set[str]]:
        """Return (new_ids, existing_ids_set) for the provided candidate ids.

        Uses collection.get with id chunks to discover which ids already exist. Silent on failures.
        """
        if not candidate_ids:
            return [], set()
        existing: set[str] = set()
        for i in range(0, len(candidate_ids), chunk_size):
            chunk = candidate_ids[i:i+chunk_size]
            try:
                res = collection.get(ids=chunk, include=[])
                # Chroma get() returns a dict with 'ids' for existing ones only
                returned = res.get("ids", []) if isinstance(res, dict) else []
                for eid in returned:
                    if eid:
                        existing.add(eid)
            except Exception:
                # Ignore errors (e.g., backend hiccups) and proceed treating them as new
                continue
        new_ids = [cid for cid in candidate_ids if cid not in existing]
        return new_ids, existing

    # ---------------- Async variant -----------------
    async def upsert_folder_async(
        self,
        folder: str,
        max_video_frames: int | None = 16,
        frame_interval_sec: float = 3.0,
        batch_size: int = 4,
        add_only: bool = False,
        image_workers: int = 4,
        text_workers: int = 4,
        video_workers: int = 2,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[int, int]:
        """Asynchronously index media in a folder.

        Strategy:
        - Discover paths synchronously (fast os.walk) then schedule IO-bound reads in threads.
        - Images & texts processed in streaming batches using limited concurrency.
        - Video processing kept sequential (can be parallelized later) to avoid GPU overload.
        - Embedding calls offloaded via asyncio.to_thread keeping API unchanged.
        - add_only=True: pre-filters existing IDs with collection.get before any embedding work.
        - progress_cb(optional): callable(processed_count, total_count, stage) invoked after each batch / file.
        """
        import asyncio

        loop = asyncio.get_running_loop()

        paths = list(walk_media(folder))
        image_paths: List[str] = [p for p in paths if is_image(p)]
        video_paths: List[str] = [p for p in paths if is_video(p)]
        text_paths: List[str] = [p for p in paths if os.path.splitext(p)[1].lower() in TEXT_EXTS]

        effective_embed_batch = max(1, batch_size)

        img_sema = asyncio.Semaphore(image_workers)
        txt_sema = asyncio.Semaphore(text_workers)
        # Global semaphore to ensure only one embedding call runs at a time
        embed_sema = asyncio.Semaphore(1)

        # Total number of source PATH files (not counting per-frame) for progress purposes
        total_files = len(image_paths) + len(text_paths) + len(video_paths)
        processed_files = 0

        def _notify(stage: str):
            if progress_cb and total_files > 0:
                try:
                    progress_cb(processed_files, total_files, stage)
                except Exception:
                    pass

        _notify("start")

        async def load_image_async(p: str):
            async with img_sema:
                try:
                    return p, await loop.run_in_executor(None, load_image, p)
                except Exception:
                    return p, None

        async def read_text_async(p: str):
            async with txt_sema:
                return p, await loop.run_in_executor(None, _read_text_from_file, p)

        # -------- Images --------
        async def process_images():
            if not image_paths:
                return

            queue: "asyncio.Queue[Tuple[str, Optional[Image.Image]] | object]" = asyncio.Queue(maxsize=max(1, batch_size) * 2)
            sentinel = object()
            target_batch = max(1, batch_size)

            async def embed_batch(items: List[Tuple[str, Image.Image]]):
                if not items:
                    return
                paths = [p for p, _ in items]
                images = [im for _, im in items]
                try:
                    target_embed_batch = min(len(images), effective_embed_batch)
                    embed_callable = functools.partial(
                        self.embedder.embed_images,
                        images,
                        batch_size=target_embed_batch,
                    )
                    async with embed_sema:
                        img_embs = await loop.run_in_executor(None, embed_callable)
                except Exception:
                    for im in images:
                        try:
                            im.close()
                        except Exception:
                            pass
                    return

                metas: List[Dict[str, Any]] = []
                ids: List[str] = []
                for p, emb, im in zip(paths, img_embs, images):
                    thumb_arr = self._resize_for_storage(im)
                    try:
                        pil_thumb = Image.fromarray(thumb_arr)
                    except Exception:
                        pil_thumb = im
                    thumb_path = self._thumb_path_for(p)
                    saved_thumb = self._save_thumbnail(pil_thumb, thumb_path)
                    metas.append({
                        "path": p,
                        "kind": "image",
                        "thumb_path": saved_thumb or p,
                        "file_hash": fast_file_hash(p),
                    })
                    ids.append(f"img::{p}")

                for im in images:
                    try:
                        im.close()
                    except Exception:
                        pass

                if ids:
                    if add_only:
                        self.col_image.add(ids=ids, embeddings=img_embs, metadatas=metas, documents=None)  # type: ignore[arg-type]
                    else:
                        self.col_image.upsert(ids=ids, embeddings=img_embs, metadatas=metas, documents=None)  # type: ignore[arg-type]

            async def producer():
                nonlocal processed_files
                chunk_size = max(1, batch_size)
                for start in range(0, len(image_paths), chunk_size):
                    batch = image_paths[start:start+chunk_size]
                    batch_original_len = len(batch)
                    candidate_ids = [f"img::{p}" for p in batch]
                    if add_only:
                        new_ids, _ = self._filter_new_ids(self.col_image, candidate_ids)
                        if not new_ids:
                            processed_files += batch_original_len
                            _notify("images")
                            continue
                        batch = [p for p in batch if f"img::{p}" in new_ids]
                    load_tasks = [asyncio.create_task(load_image_async(p)) for p in batch]
                    for fut in asyncio.as_completed(load_tasks):
                        try:
                            p, im = await fut
                        except Exception:
                            continue
                        if im is None:
                            continue
                        await queue.put((p, im))
                    processed_files += batch_original_len
                    _notify("images")
                await queue.put(sentinel)

            async def consumer():
                buffer: List[Tuple[str, Image.Image]] = []
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    p, im = item  # type: ignore[misc]
                    if im is None:
                        continue
                    buffer.append((p, im))
                    if len(buffer) >= target_batch:
                        await embed_batch(buffer)
                        buffer = []
                if buffer:
                    await embed_batch(buffer)

            await asyncio.gather(producer(), consumer())

        # -------- Texts --------
        async def process_texts():
            if not text_paths:
                return

            queue: "asyncio.Queue[Tuple[str, str] | object]" = asyncio.Queue(maxsize=max(1, batch_size) * 2)
            sentinel = object()
            target_batch = max(1, batch_size)

            async def embed_batch(ids: List[str], texts: List[str], metas: List[Dict[str, Any]], docs: List[str]):
                if not texts:
                    return
                target_embed_batch = min(len(texts), effective_embed_batch)
                embed_callable = functools.partial(
                    self.embedder.embed_text,
                    texts,
                    batch_size=target_embed_batch,
                )
                try:
                    async with embed_sema:
                        text_embs = await loop.run_in_executor(None, embed_callable)
                except Exception:
                    return
                if add_only:
                    self.col_text.add(ids=ids, embeddings=text_embs, metadatas=metas, documents=docs)  # type: ignore[arg-type]
                else:
                    self.col_text.upsert(ids=ids, embeddings=text_embs, metadatas=metas, documents=docs)  # type: ignore[arg-type]

            async def producer():
                nonlocal processed_files
                chunk_size = max(1, batch_size)
                for start in range(0, len(text_paths), chunk_size):
                    chunk = text_paths[start:start+chunk_size]
                    chunk_original_len = len(chunk)
                    if add_only:
                        candidate_ids = [f"txt::{p}" for p in chunk]
                        new_ids, _ = self._filter_new_ids(self.col_text, candidate_ids)
                        if not new_ids:
                            processed_files += chunk_original_len
                            _notify("texts")
                            continue
                        chunk = [p for p in chunk if f"txt::{p}" in new_ids]
                    load_tasks = [asyncio.create_task(read_text_async(p)) for p in chunk]
                    for fut in asyncio.as_completed(load_tasks):
                        try:
                            p, content = await fut
                        except Exception:
                            continue
                        if not content:
                            continue
                        await queue.put((p, content))
                    processed_files += chunk_original_len
                    _notify("texts")
                await queue.put(sentinel)

            async def consumer():
                batch_ids: List[str] = []
                batch_texts: List[str] = []
                batch_metas: List[Dict[str, Any]] = []
                batch_docs: List[str] = []
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    p, content = item  # type: ignore[misc]
                    if not content:
                        continue
                    truncated = content[:20000]
                    if not truncated:
                        continue
                    cid = f"txt::{p}"
                    batch_ids.append(cid)
                    batch_texts.append(truncated)
                    batch_docs.append(truncated)
                    batch_metas.append({"path": p, "kind": "text", "file_hash": fast_file_hash(p)})
                    if len(batch_texts) >= target_batch:
                        await embed_batch(batch_ids, batch_texts, batch_metas, batch_docs)
                        batch_ids = []
                        batch_texts = []
                        batch_metas = []
                        batch_docs = []
                if batch_texts:
                    await embed_batch(batch_ids, batch_texts, batch_metas, batch_docs)

            await asyncio.gather(producer(), consumer())

        # -------- Videos (sequential) --------
        async def process_videos():
            if not video_paths:
                return

            queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=max(1, batch_size) * 2)
            sentinel = object()
            video_sema = asyncio.Semaphore(max(1, video_workers))

            async def handle_video(p: str):
                nonlocal processed_files
                try:
                    # video_hash = fast_file_hash(p)
                    existing_indices: Set[int] = set()
                    if add_only:
                        try:
                            already_indexed = self.col_video.get(
                                where={"video_path": p},
                                include=["metadatas"],
                                limit=1,
                            )
                            if isinstance(already_indexed, dict) and already_indexed.get("ids"):
                                return
                        except Exception as e:
                            print(f"Error checking existing video {p}: {e}")
                        try:
                            existing_meta_resp = self.col_video.get(
                                where={"video_path": p},
                                include=["metadatas"],
                            )
                            metas_existing_raw = existing_meta_resp.get("metadatas", []) if isinstance(existing_meta_resp, dict) else []
                            metas_existing = metas_existing_raw or []
                            hash_mismatch = False
                            for meta in metas_existing:
                                if not isinstance(meta, dict):
                                    continue
                                idx_val = meta.get("frame_index")
                                if isinstance(idx_val, (int, float)):
                                    existing_indices.add(int(idx_val))
                                # if meta.get("file_hash") != video_hash:
                                #     hash_mismatch = True
                            # if hash_mismatch:
                            #     try:
                            #         self.col_video.delete(where={"video_path": p})
                            #         existing_indices.clear()
                            #     except Exception:
                            #         pass
                        except Exception:
                            existing_indices = set()

                    async with video_sema:
                        print(f"Sample frames from video: {p}",end="")
                        frames, duration, total, indices, fps = await loop.run_in_executor(
                            None,
                            sample_video_frames,
                            p,
                            max_video_frames if max_video_frames is not None else 0,
                            float(frame_interval_sec),
                        )
                        print(f" - extracted {len(frames)} frames")

                    if not frames:
                        return

                    frame_indices = [int(indices[i]) if i < len(indices) else i for i in range(len(frames))]
                    if add_only and existing_indices:
                        filtered_frames: List[np.ndarray] = []
                        filtered_indices: List[int] = []
                        for rgb, idx in zip(frames, frame_indices):
                            if idx in existing_indices:
                                continue
                            filtered_frames.append(rgb)
                            filtered_indices.append(idx)
                        frames = filtered_frames
                        frame_indices = filtered_indices
                        if not frames:
                            return

                    sampled_count = len(frame_indices)
                    video_hash = fast_file_hash(p)
                    await queue.put((p, frames, frame_indices, duration, total, fps, video_hash, sampled_count))
                except Exception as e:
                    print(f"Failed to index video {p}: {e}")
                finally:
                    processed_files += 1
                    _notify("videos")

            async def producer():
                tasks = [asyncio.create_task(handle_video(p)) for p in video_paths]
                if tasks:
                    await asyncio.gather(*tasks)
                await queue.put(sentinel)

            async def consumer():
                processing_tasks: List[asyncio.Future[Any]] = []

                def process_frame_batch(
                    path: str,
                    frames: List[np.ndarray],
                    frame_embs: List[np.ndarray],
                    frame_indices: List[int],
                    duration: float,
                    total: int,
                    fps: float,
                    video_hash: str,
                    sampled_count: int,
                ) -> None:
                    try:
                        if not frame_embs:
                            return

                        ids: List[str] = []
                        embs: List[List[float]] = []
                        metas: List[Dict[str, Any]] = []
                        for i, (rgb, emb, frame_idx) in enumerate(zip(frames, frame_embs, frame_indices)):
                            ts = float(frame_idx / fps) if fps > 0 else float(i)
                            try:
                                pil_frame = Image.fromarray(rgb)
                            except Exception:
                                continue
                            thumb_arr = self._resize_for_storage(pil_frame)
                            try:
                                pil_thumb = Image.fromarray(thumb_arr)
                            except Exception:
                                pil_thumb = pil_frame
                            thumb_path = self._thumb_path_for(path, extra=str(frame_idx))
                            saved_thumb = self._save_thumbnail(pil_thumb, thumb_path)
                            ids.append(f"vframe::{path}::{frame_idx}")
                            embs.append(emb.tolist())
                            metas.append({
                                "path": path,
                                "kind": "video_frame",
                                "video_path": path,
                                "frame_index": frame_idx,
                                "timestamp": ts,
                                "duration": float(duration),
                                "total_frames": int(total),
                                "sampled_frames": int(sampled_count),
                                "thumb_path": saved_thumb or path,
                                "file_hash": video_hash,
                            })
                            try:
                                pil_frame.close()
                            except Exception:
                                pass

                        if not ids:
                            return

                        embs_arr = np.asarray(embs, dtype=np.float32)
                        try:
                            if add_only:
                                self.col_video.add(ids=ids, embeddings=embs_arr, metadatas=metas, documents=None)  # type: ignore[arg-type]
                            else:
                                self.col_video.upsert(ids=ids, embeddings=embs_arr, metadatas=metas, documents=None)  # type: ignore[arg-type]
                        finally:
                            for emb in frame_embs:
                                try:
                                    del emb
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"Failed to store video frames for {path}: {e}")
                    finally:
                        try:
                            del frames
                        except Exception:
                            pass

                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    (
                        p,
                        frames,
                        frame_indices,
                        duration,
                        total,
                        fps,
                        video_hash,
                        sampled_count,
                    ) = item  # type: ignore[misc]
                    if not frames:
                        continue

                    frame_embs: Optional[List[np.ndarray]] = None
                    try:
                        target_embed_batch = min(len(frames), effective_embed_batch)
                        try:
                            async with embed_sema:
                                frame_embs = await loop.run_in_executor(
                                    None,
                                    self.embedder.embed_video_frames,
                                    frames,
                                    target_embed_batch,
                                )
                        except Exception:
                            continue

                        if not frame_embs:
                            continue

                        task = loop.run_in_executor(
                            None,
                            functools.partial(
                                process_frame_batch,
                                p,
                                frames,
                                frame_embs,
                                frame_indices,
                                float(duration),
                                int(total),
                                float(fps),
                                video_hash,
                                int(sampled_count),
                            ),
                        )
                        processing_tasks.append(asyncio.wrap_future(task))
                        frame_embs = None
                    finally:
                        try:
                            del frames
                        except Exception:
                            pass
                        if frame_embs is not None:
                            del frame_embs

                if processing_tasks:
                    await asyncio.gather(*processing_tasks)

            await asyncio.gather(producer(), consumer())


        await asyncio.gather(process_images(), process_texts(), process_videos())
        _notify("done")
        return len(image_paths), len(video_paths)

    def search_by_text(self, query: str, k: int = 10):
                """Search by text without merging collection results.

                Returns a dict with per-collection Chroma-style query outputs so that
                the UI can keep indices stable per modality and avoid ambiguity caused
                by cross-type merges & resorting.
                {
                    'image': <image_result>,
                    'video': <video_frame_result>,
                    'text': <text_result>
                }
                Each result contains (ids, metadatas, distances) for a *single* query.
                """
                q_emb = self.embedder.embed_text([query])[0]
                res_img = self.col_image.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["metadatas", "distances"])
                res_vid = self.col_video.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["metadatas", "distances"])
                res_txt = self.col_text.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["metadatas", "distances"])
                return {"image": res_img, "video": res_vid, "text": res_txt}

    def search_by_image(self, image: Image.Image, k: int = 10):
                """Search by image without merging results.

                Returns a dict with keys 'image' & 'video'.
                """
                q_emb = self.embedder.embed_images([image])[0]
                res_img = self.col_image.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["metadatas", "distances"])
                res_vid = self.col_video.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["metadatas", "distances"])
                return {"image": res_img, "video": res_vid}

    @staticmethod
    def _merge_results(results: Sequence[Any], k: int) -> Dict[str, Any]:
        """Merge multiple Chroma query results into a single result sorted by distance.

        Since thumbnails are stored on disk, we only propagate metadata & distances.
        """
        raise NotImplementedError("_merge_results no longer used; results returned per collection.")
