"""Chroma indexing utilities."""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

import chromadb
import numpy as np
from PIL import Image
from pi_heif import register_heif_opener

from .constants import TEXT_EXTENSIONS
from .file_utils import is_image, is_video, load_image, walk_media
from .hashing import fast_file_hash
from .thumbnail_store import ThumbnailStore
from .text_utils import read_text_from_file

try:
    from .embeddings import EmbeddingModel, sample_video_frames
except ImportError:  # pragma: no cover - fallback for script execution
    from embeddings import EmbeddingModel, sample_video_frames

register_heif_opener()

__all__ = ["ChromaRAG"]


class ChromaRAG:
    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "media",
        model_name: str = "jinaai/jina-embeddings-v4",
        quantize: bool = True,
        enable_flash_atten: bool = False,
    ) -> None:
        self.persist_dir = os.path.abspath(persist_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.thumb_dir = os.path.join(self.persist_dir, "thumbnails")
        self._thumb_store = ThumbnailStore(self.thumb_dir)

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
        self.embedder = EmbeddingModel(
            model_name,
            quantize=quantize,
            enable_flash_atten=enable_flash_atten,
        )

    # Thin wrappers over thumbnail helpers so legacy code remains unchanged.
    def _thumb_path_for(self, source_path: str, extra: Optional[str] = None) -> str:
        return self._thumb_store.path_for(source_path, extra=extra)

    def _save_thumbnail(self, img: Image.Image, dest_path: str) -> str:
        return self._thumb_store.save(img, dest_path)

    def _resize_for_storage(self, img: Image.Image, max_size: int = 256) -> np.ndarray:
        return self._thumb_store.resize_for_storage(img, max_size=max_size)

    def _filter_new_ids(
        self,
        collection: Any,
        candidate_ids: List[str],
        chunk_size: int = 256,
    ) -> Tuple[List[str], Set[str]]:
        if not candidate_ids:
            return [], set()
        existing: Set[str] = set()
        for idx in range(0, len(candidate_ids), chunk_size):
            chunk = candidate_ids[idx:idx + chunk_size]
            try:
                res = collection.get(ids=chunk, include=[])
                returned = res.get("ids", []) if isinstance(res, dict) else []
                for eid in returned:
                    if eid:
                        existing.add(eid)
            except Exception:
                continue
        new_ids = [cid for cid in candidate_ids if cid not in existing]
        return new_ids, existing

    async def upsert_folder_async(
        self,
        folder: str,
        max_video_frames: Optional[int] = 16,
        frame_interval_sec: float = 3.0,
        batch_size: int = 4,
        add_only: bool = False,
        image_workers: int = 4,
        text_workers: int = 4,
        video_workers: int = 2,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[int, int]:
        loop = asyncio.get_running_loop()

        paths = list(walk_media(folder))
        image_paths = [p for p in paths if is_image(p)]
        video_paths = [p for p in paths if is_video(p)]
        text_paths = [p for p in paths if os.path.splitext(p)[1].lower() in TEXT_EXTENSIONS]

        effective_embed_batch = max(1, batch_size)

        img_sema = asyncio.Semaphore(image_workers)
        txt_sema = asyncio.Semaphore(text_workers)
        embed_sema = asyncio.Semaphore(1)

        total_files = len(image_paths) + len(text_paths) + len(video_paths)
        processed_files = 0

        def _notify(stage: str) -> None:
            if progress_cb and total_files > 0:
                try:
                    progress_cb(processed_files, total_files, stage)
                except Exception:
                    pass

        _notify("start")

        async def load_image_async(path: str):
            async with img_sema:
                try:
                    return path, await loop.run_in_executor(None, load_image, path)
                except Exception:
                    return path, None

        async def read_text_async(path: str):
            async with txt_sema:
                return path, await loop.run_in_executor(None, read_text_from_file, path)

        async def process_images():
            if not image_paths:
                return

            queue: "asyncio.Queue[Tuple[str, Optional[Image.Image]] | object]" = asyncio.Queue(
                maxsize=max(1, batch_size) * 2
            )
            sentinel = object()
            target_batch = max(1, batch_size)

            async def embed_batch(items: List[Tuple[str, Image.Image]]):
                if not items:
                    return
                paths_local = [p for p, _ in items]
                images_local = [im for _, im in items]
                try:
                    target_embed_batch = min(len(images_local), effective_embed_batch)
                    embed_callable = functools.partial(
                        self.embedder.embed_images,
                        images_local,
                        batch_size=target_embed_batch,
                    )
                    async with embed_sema:
                        img_embs = await loop.run_in_executor(None, embed_callable)
                except Exception:
                    for im in images_local:
                        try:
                            im.close()
                        except Exception:
                            pass
                    return

                metas: List[Dict[str, Any]] = []
                ids: List[str] = []
                for path, emb, image_obj in zip(paths_local, img_embs, images_local):
                    thumb_arr = self._resize_for_storage(image_obj)
                    try:
                        pil_thumb = Image.fromarray(thumb_arr)
                    except Exception:
                        pil_thumb = image_obj
                    thumb_path = self._thumb_path_for(path)
                    saved_thumb = self._save_thumbnail(pil_thumb, thumb_path)
                    metas.append(
                        {
                            "path": path,
                            "kind": "image",
                            "thumb_path": saved_thumb or path,
                            "file_hash": fast_file_hash(path),
                        }
                    )
                    ids.append(f"img::{path}")

                for image_obj in images_local:
                    try:
                        image_obj.close()
                    except Exception:
                        pass

                if ids:
                    if add_only:
                        self.col_image.add(
                            ids=ids,
                            embeddings=img_embs,
                            metadatas=cast(Any, metas),
                            documents=None,
                        )
                    else:
                        self.col_image.upsert(
                            ids=ids,
                            embeddings=img_embs,
                            metadatas=cast(Any, metas),
                            documents=None,
                        )

            async def producer():
                nonlocal processed_files
                chunk = max(1, batch_size)
                for start in range(0, len(image_paths), chunk):
                    batch = image_paths[start:start + chunk]
                    batch_original_len = len(batch)
                    candidate_ids = [f"img::{path}" for path in batch]
                    if add_only:
                        new_ids, _ = self._filter_new_ids(self.col_image, candidate_ids)
                        if not new_ids:
                            processed_files += batch_original_len
                            _notify("images")
                            continue
                        batch = [path for path in batch if f"img::{path}" in new_ids]
                    load_tasks = [asyncio.create_task(load_image_async(path)) for path in batch]
                    for future in asyncio.as_completed(load_tasks):
                        try:
                            path, image_obj = await future
                        except Exception:
                            continue
                        if image_obj is None:
                            continue
                        await queue.put((path, image_obj))
                    processed_files += batch_original_len
                    _notify("images")
                await queue.put(sentinel)

            async def consumer():
                buffer: List[Tuple[str, Image.Image]] = []
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    path, image_obj = item  # type: ignore[misc]
                    if image_obj is None:
                        continue
                    buffer.append((path, image_obj))
                    if len(buffer) >= target_batch:
                        await embed_batch(buffer)
                        buffer = []
                if buffer:
                    await embed_batch(buffer)

            await asyncio.gather(producer(), consumer())

        async def process_texts():
            if not text_paths:
                return

            queue: "asyncio.Queue[Tuple[str, str] | object]" = asyncio.Queue(
                maxsize=max(1, batch_size) * 2
            )
            sentinel = object()
            target_batch = max(1, batch_size)

            async def embed_batch(
                batch_ids: List[str],
                batch_texts: List[str],
                batch_metas: List[Dict[str, Any]],
                batch_docs: List[str],
            ):
                if not batch_texts:
                    return
                target_embed_batch = min(len(batch_texts), effective_embed_batch)
                embed_callable = functools.partial(
                    self.embedder.embed_text,
                    batch_texts,
                    batch_size=target_embed_batch,
                )
                try:
                    async with embed_sema:
                        text_embs = await loop.run_in_executor(None, embed_callable)
                except Exception:
                    return
                if add_only:
                    self.col_text.add(
                        ids=batch_ids,
                        embeddings=text_embs,
                        metadatas=cast(Any, batch_metas),
                        documents=batch_docs,
                    )
                else:
                    self.col_text.upsert(
                        ids=batch_ids,
                        embeddings=text_embs,
                        metadatas=cast(Any, batch_metas),
                        documents=batch_docs,
                    )

            async def producer():
                nonlocal processed_files
                chunk = max(1, batch_size)
                for start in range(0, len(text_paths), chunk):
                    segment = text_paths[start:start + chunk]
                    original_len = len(segment)
                    if add_only:
                        candidate_ids = [f"txt::{path}" for path in segment]
                        new_ids, _ = self._filter_new_ids(self.col_text, candidate_ids)
                        if not new_ids:
                            processed_files += original_len
                            _notify("texts")
                            continue
                        segment = [path for path in segment if f"txt::{path}" in new_ids]
                    load_tasks = [asyncio.create_task(read_text_async(path)) for path in segment]
                    for future in asyncio.as_completed(load_tasks):
                        try:
                            path, content = await future
                        except Exception:
                            continue
                        if not content:
                            continue
                        await queue.put((path, content))
                    processed_files += original_len
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
                    path, content = item  # type: ignore[misc]
                    if not content:
                        continue
                    truncated = content[:20000]
                    if not truncated:
                        continue
                    cid = f"txt::{path}"
                    batch_ids.append(cid)
                    batch_texts.append(truncated)
                    batch_docs.append(truncated)
                    batch_metas.append(
                        {
                            "path": path,
                            "kind": "text",
                            "file_hash": fast_file_hash(path),
                        }
                    )
                    if len(batch_texts) >= target_batch:
                        await embed_batch(batch_ids, batch_texts, batch_metas, batch_docs)
                        batch_ids = []
                        batch_texts = []
                        batch_metas = []
                        batch_docs = []
                if batch_texts:
                    await embed_batch(batch_ids, batch_texts, batch_metas, batch_docs)

            await asyncio.gather(producer(), consumer())

        async def process_videos():
            if not video_paths:
                return

            queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=max(1, batch_size) * 2)
            sentinel = object()
            video_sema = asyncio.Semaphore(max(1, video_workers))

            async def handle_video(path: str):
                nonlocal processed_files
                try:
                    existing_indices: Set[int] = set()
                    if add_only:
                        try:
                            already_indexed = self.col_video.get(
                                where={"video_path": path},
                                include=["metadatas"],
                                limit=1,
                            )
                            if isinstance(already_indexed, dict) and already_indexed.get("ids"):
                                return
                        except Exception:
                            pass
                        try:
                            existing_meta_resp = self.col_video.get(
                                where={"video_path": path},
                                include=["metadatas"],
                            )
                            metas_existing_raw = (
                                existing_meta_resp.get("metadatas", []) if isinstance(existing_meta_resp, dict) else []
                            )
                            metas_existing = metas_existing_raw or []
                            for meta in metas_existing:
                                if not isinstance(meta, dict):
                                    continue
                                idx_val = meta.get("frame_index")
                                if isinstance(idx_val, (int, float)):
                                    existing_indices.add(int(idx_val))
                        except Exception:
                            existing_indices = set()

                    async with video_sema:
                        print(f"Sample frames from video: {path}", end="")
                        frames, duration, total, indices, fps = await loop.run_in_executor(
                            None,
                            sample_video_frames,
                            path,
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
                    video_hash = fast_file_hash(path)
                    await queue.put((path, frames, frame_indices, duration, total, fps, video_hash, sampled_count))
                except Exception as exc:
                    print(f"Failed to index video {path}: {exc}")
                finally:
                    processed_files += 1
                    _notify("videos")

            async def producer():
                tasks = [asyncio.create_task(handle_video(path)) for path in video_paths]
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
                            timestamp = float(frame_idx / fps) if fps > 0 else float(i)
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
                            metas.append(
                                {
                                    "path": path,
                                    "kind": "video_frame",
                                    "video_path": path,
                                    "frame_index": frame_idx,
                                    "timestamp": timestamp,
                                    "duration": float(duration),
                                    "total_frames": int(total),
                                    "sampled_frames": int(sampled_count),
                                    "thumb_path": saved_thumb or path,
                                    "file_hash": video_hash,
                                }
                            )
                            try:
                                pil_frame.close()
                            except Exception:
                                pass

                        if not ids:
                            return

                        embs_arr = np.asarray(embs, dtype=np.float32)
                        try:
                            if add_only:
                                self.col_video.add(
                                    ids=ids,
                                    embeddings=embs_arr,
                                    metadatas=cast(Any, metas),
                                    documents=None,
                                )
                            else:
                                self.col_video.upsert(
                                    ids=ids,
                                    embeddings=embs_arr,
                                    metadatas=cast(Any, metas),
                                    documents=None,
                                )
                        finally:
                            for emb in frame_embs:
                                try:
                                    del emb
                                except Exception:
                                    pass
                    except Exception as exc:
                        print(f"Failed to store video frames for {path}: {exc}")
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
                        path,
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
                                path,
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
        q_emb = self.embedder.embed_text([query])[0]
        res_img = self.col_image.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        res_vid = self.col_video.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        res_txt = self.col_text.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        return {"image": res_img, "video": res_vid, "text": res_txt}

    def search_by_image(self, image: Image.Image, k: int = 10):
        q_emb = self.embedder.embed_images([image])[0]
        res_img = self.col_image.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        res_vid = self.col_video.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        return {"image": res_img, "video": res_vid}

    @staticmethod
    def _merge_results(results: Sequence[Any], k: int) -> Dict[str, Any]:
        raise NotImplementedError(
            "_merge_results is deprecated; results are returned per collection."
        )