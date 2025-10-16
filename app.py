from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Tuple

import gradio as gr
from PIL import Image
from pi_heif import register_heif_opener
import atexit
import hashlib
import shutil
import tempfile
import subprocess

# Allow running this file directly without installing the package
try:  # try absolute import first
    from indexer import ChromaRAG
except Exception:  # pragma: no cover - fallback for direct script execution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from indexer import ChromaRAG


def _extract_video_frame_image(video_path: str, frame_index: int) -> Image.Image | None:
    """Best-effort extract a single frame (as PIL Image) from a video at the given frame index.
    Returns None if OpenCV is unavailable or extraction fails.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None


def _metadata_modified_time(md: Dict[str, Any]) -> float:
    """Best-effort stat call to determine the underlying file's modified time."""
    path = md.get("video_path") or md.get("path")
    if not path:
        return 0.0
    try:
        return float(os.path.getmtime(path))
    except OSError:
        return 0.0


def _sorted_metadata_with_distance(res: dict | None, sort_mode: str) -> List[Tuple[Dict[str, Any], float]]:
    """Return metadatas paired with distances in the order requested."""
    if not res or not res.get("metadatas"):
        return []
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    records: List[Tuple[Dict[str, Any], float, float]] = []
    for md, d in zip(metadatas, distances):
        try:
            dist = float(d)
        except Exception:
            dist = 0.0
        records.append((md, dist, _metadata_modified_time(md)))
    if sort_mode == "mtime_desc":
        records.sort(key=lambda r: r[2], reverse=True)
    return [(md, dist) for md, dist, _ in records]


def format_results(res: dict, sort_mode: str = "score") -> List[Tuple[Any, str]]:
    """Convert Chroma query result into gallery items using on-disk thumbnails.

    Each item becomes (image_path_or_media_path, caption). We prefer metadata['thumb_path'] if present.
    For text items (no image), the original path string is used so Gradio shows a placeholder.
    """
    items: List[Tuple[Any, str]] = []
    if not res or not res.get("metadatas"):
        return items
    for md, d in _sorted_metadata_with_distance(res, sort_mode):
        path = md.get("path", "")
        kind = md.get("kind", "?")
        # Prefer explicit thumb_path in metadata when present
        thumb_path = md.get("thumb_path") or path

        # Build caption text
        if kind == "video_frame":
            base = os.path.basename(md.get("video_path", path))
            fidx = md.get("frame_index", None)
            ts = md.get("timestamp", None)
            pos = f"frame {fidx}" if fidx is not None else "frame"
            if isinstance(ts, (int, float)):
                pos += f" @ {ts:.2f}s"
            caption = f"video-frame | {base} | {pos} | score={1 - d:.3f}"
        else:
            caption = f"{kind} | {os.path.basename(path)} | score={1 - d:.3f}"

        # For image-like results, always try to return a PIL Image (RGB); never return the raw thumb_path string.
        img_obj: Any = None
        try:
            if kind == "video_frame":
                # Prefer an existing thumbnail if provided in metadata (thumb_path).
                if thumb_path and os.path.isfile(thumb_path):
                    try:
                        with Image.open(thumb_path) as im:
                            img_obj = im.convert("RGB")
                    except Exception:
                        img_obj = None
                # If no thumb or open failed, extract the frame image from the video as a fallback.
                if img_obj is None:
                    vf_path = _get_or_create_video_frame_cached_path(md.get("video_path", path), int(md.get("frame_index", 0)))
                    if vf_path and os.path.isfile(vf_path):
                        with Image.open(vf_path) as im:
                            img_obj = im.convert("RGB")
            elif kind == "image":
                candidate = thumb_path if os.path.isfile(thumb_path) else path
                if candidate and os.path.isfile(candidate):
                    with Image.open(candidate) as im:
                        img_obj = im.convert("RGB")
        except Exception:
            img_obj = None

        if img_obj is not None:
            display_value: Any = img_obj
        else:
            # For text/other kinds we fall back to original path (not thumbnail path) so gallery shows placeholder.
            display_value = path
        items.append((display_value, caption))
    return items


# --- HEIF/HEIC temporary JPG cache utilities ---
_HEIF_EXTS = {".heif", ".heic", ".avif"}
_TMP_CACHE_DIR: str = tempfile.mkdtemp(prefix="rag_works_heif_cache_")
_heif_cache: Dict[str, str] = {}
_vidframe_cache: Dict[str, str] = {}


# Register HEIF/HEIC opener for PIL (requires pi_heif installed)
try:
    register_heif_opener()
except Exception:
    # If registration fails, continue; non-HEIF images and videos still work
    pass


def _cache_key_for_path(src_path: str) -> str:
    try:
        mtime = os.path.getmtime(src_path)
    except OSError:
        mtime = 0
    key = f"{os.path.abspath(src_path)}::{mtime}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def _cache_key_for_video_frame(video_path: str, frame_index: int) -> str:
    # Use stable key derived from absolute path and frame index only
    key = f"{os.path.abspath(video_path)}::frame::{int(frame_index)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def _get_or_create_video_frame_cached_path(video_path: str, frame_index: int) -> str:
    key = _cache_key_for_video_frame(video_path, frame_index)
    cached = _vidframe_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    img = _extract_video_frame_image(video_path, frame_index)
    if img is None:
        return video_path
    out_path = os.path.join(_TMP_CACHE_DIR, f"vf_{key}.jpg")
    try:
        img.save(out_path, format="JPEG", quality=90, optimize=True)
        _vidframe_cache[key] = out_path
        return out_path
    except Exception:
        return video_path

def _convert_for_gallery_if_needed(src_path: str) -> str:
    """
    If src_path is a HEIF/HEIC/AVIF image, convert to JPG in temp cache and return the JPG path.
    Otherwise return src_path unchanged. Reuse cached conversion between calls.
    """
    if not src_path:
        return src_path
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in _HEIF_EXTS:
        gr.set_static_paths(src_path)
        return src_path

    key = _cache_key_for_path(src_path)
    if key in _heif_cache and os.path.exists(_heif_cache[key]):
        return _heif_cache[key]

    # Convert using PIL (if HEIF opener registered); if conversion fails, fall back to original path
    out_path = os.path.join(_TMP_CACHE_DIR, f"{key}.jpg")
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.save(out_path, format="JPEG", quality=90, optimize=True)
        _heif_cache[key] = out_path
        return out_path
    except Exception:
        return src_path


def _cleanup_tmp_cache_dir() -> None:
    try:
        shutil.rmtree(_TMP_CACHE_DIR, ignore_errors=True)
    except Exception:
        pass


atexit.register(_cleanup_tmp_cache_dir)


_RAG_CACHE: Dict[str, ChromaRAG] = {}
QUANTIZE = True
ENABLE_FLASH_ATTENTION = False


def get_rag(persist_dir: str) -> ChromaRAG:
    """Lazily create or return cached ChromaRAG for the given persist_dir."""
    pd = os.path.abspath(persist_dir or ".chroma")
    if pd in _RAG_CACHE:
        return _RAG_CACHE[pd]
    rag = ChromaRAG(persist_dir=pd,quantize=QUANTIZE)
    _RAG_CACHE[pd] = rag
    return rag


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="RAG Image/Video Search (ChromaDB)") as demo:
        gr.Markdown("# RAG Image/Video Search\nIndex a folder of images and videos, then search by text or image.")
        gr.Markdown("## Settings")
        persist_dir_box = gr.Textbox(label="ChromaDB persist_dir", value=".chroma", placeholder=".chroma")
        enable_quantize = gr.Checkbox(label="Enable model quantization", value=QUANTIZE, info="When enabled, use a smaller quantized embedding model (quantize to int8, slightly less accurate).")
        enable_flash_attention = gr.Checkbox(label="Enable Flash Attention (if available)", value=ENABLE_FLASH_ATTENTION, info="When enabled, use flash attention in the embedding model if installed (great for indexing).")

        # Update global QUANTIZE variable when checkbox changes, so new RAG instances use the updated setting
        def _on_quantize_change(q: bool):
            global QUANTIZE
            QUANTIZE = q
        enable_quantize.change(fn=_on_quantize_change, inputs=[enable_quantize])

        # Update global ENABLE_FLASH_ATTENTION variable when checkbox changes, so new RAG instances use the updated setting
        def _on_flash_atten_change(e: bool):
            global ENABLE_FLASH_ATTENTION
            ENABLE_FLASH_ATTENTION = e
        enable_flash_attention.change(fn=_on_flash_atten_change, inputs=[enable_flash_attention])

        with gr.Tab("Index"):
            folder = gr.Textbox(label="Folder to index", placeholder="C:/path/to/media or /path/to/media")
            add_only = gr.Checkbox(label="Add only (skip updates)", value=True, info="When enabled, only add new items; skip updating existing items in the index.")
            batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch size", info="Number of items to embed in one batch. Larger values use more memory but may be faster.")
            limit_max_frames = gr.Checkbox(label="Limit max frames", value=False, info="When enabled, cap sampled frames by Max frames per video")
            max_frames = gr.Slider(4, 64, value=16, step=1, label="Max frames per video", visible=limit_max_frames.value)
            frame_interval = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Frame interval (seconds)")
            index_btn = gr.Button("Index Folder")
            index_status = gr.Markdown()

            # Use Gradio's built-in Progress tracker instead of a Slider.
            # The progress UI is provided by gr.Progress and is injected automatically
            # into the event handler when you declare a default arg: progress=gr.Progress()
            stage_box = gr.Textbox(label="Stage", value="", interactive=False)

            def do_index(persist_dir: str, p: str, mf: int, interval: float, limit_max: bool, batch_size: int, progress=gr.Progress()):
                """Index folder and report progress via gr.Progress.

                The gr.Progress object is injected by Gradio and calling it with a
                fractional value and optional description updates the UI.
                """
                gr.set_static_paths(persist_dir)
                if not p or not os.path.isdir(p):
                    return "", "Provide a valid folder path."
                rag = get_rag(persist_dir)
                mf_arg = int(mf) if limit_max else None

                def _cb(done: int, total: int, stage: str):
                    # Compute fractional progress and send it to gr.Progress
                    try:
                        if total <= 0:
                            frac = 0.0
                        else:
                            frac = min(1.0, max(0.0, done / total))
                        # progress accepts (fraction, desc=...) or (fraction, None)
                        progress(frac, desc=stage)
                    except Exception:
                        # Fail silently to avoid breaking indexing when progress UI isn't available
                        pass

                task = rag.upsert_folder_async(
                    p,
                    max_video_frames=mf_arg,
                    frame_interval_sec=float(interval),
                    batch_size=batch_size,
                    add_only=bool(add_only),
                    progress_cb=_cb,
                )
                imgs, vids = asyncio.run(task)
                final_stage = f"done: indexed {imgs} images and {vids} videos"
                # send a final 100% progress update
                try:
                    progress(1.0, desc=final_stage)
                except Exception:
                    pass
                return final_stage, f"Indexed {imgs} images and {vids} videos from {p}."

            # Note: gr.Progress is injected automatically into the function when declared
            # as a default parameter (progress=gr.Progress()). We therefore only wire
            # the explicit outputs we care about here (stage and status message).
            index_btn.click(
                fn=do_index,
                inputs=[persist_dir_box, folder, max_frames, frame_interval, limit_max_frames, batch_size],
                outputs=[stage_box, index_status]
            )

            # Hide or show the max_frames slider depending on the limit_max_frames checkbox
            def _on_limit_max_change(limit: bool):
                # When limit is False, hide the slider. When True, show it.
                return gr.update(visible=bool(limit))

            limit_max_frames.change(fn=_on_limit_max_change, inputs=[limit_max_frames], outputs=[max_frames])

        with gr.Tab("Search"):
            with gr.Row():
                query = gr.Textbox(label="Text query", placeholder="e.g., 'a dog playing in the park'")
                with gr.Column():
                    # Use File to allow HEIF/HEIC/AVIF uploads; Image component restricts to image/* which may reject HEIF
                    image_file = gr.File(label="Image query (optional)",height=30, type="filepath", file_types=["image", ".heic", ".heif", ".avif"])
                    preview = gr.Image(label="Uploaded image preview", type="filepath")
                topk = gr.Slider(1, 50, value=10, step=1, label="Top-K")
            type_filter = gr.CheckboxGroup(
                choices=["image", "video", "text"],
                value=["image", "video"],
                label="Result types (filter)",
                info="Select which types to include in search results. 'video' shows matched frames."
            )
            sort_mode = gr.Radio(
                choices=["Best score", "Modified time (newest first)"],
                value="Best score",
                label="Sort order",
                info="Choose how to order gallery results."
            )
            search_btn = gr.Button("Search")
            # Two top-level galleries: images (as before) and video groups (one per source video)
            images_gallery = gr.Gallery(label="Image Results", show_label=True, columns=10, height=300, visible=False)
            videos_gallery = gr.Gallery(label="Video Results (click to expand)", show_label=True, columns=15, rows=1, interactive=False, height=200, visible=False)
            # Gallery to show expanded frames for a selected video group (hidden until used)
            video_frames_gallery = gr.Gallery(label="Video Frames", show_label=True, columns=15, rows=1, visible=False, height=200)
            # Cache last unfiltered search result to avoid re-querying on filter changes
            cached_res = gr.State(value=None)
            # Track currently selected path for "open in file manager"
            selected_path = gr.State(value=None)
            with gr.Row():
                selected_info = gr.Textbox(label="Selected file", interactive=False)
                open_btn = gr.Button("Open in file manager")
            open_status = gr.Markdown(visible=False)

            SORT_LABEL_TO_MODE = {
                "Best score": "score",
                "Modified time (newest first)": "mtime_desc",
            }

            def _resolve_sort_mode(label: str | None) -> str:
                return SORT_LABEL_TO_MODE.get(label or "", "score")

            def _filter_collection(res: dict | None, allowed_kinds: set[str]) -> dict:
                if not res or not res.get("metadatas"):
                    return {"metadatas": [[]], "distances": [[]]}
                mds = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                keep_mds: List[Dict[str, Any]] = []
                keep_d: List[float] = []
                for md, d in zip(mds, dists):
                    if md.get("kind") in allowed_kinds:
                        keep_mds.append(md)
                        keep_d.append(float(d))
                return {"metadatas": [keep_mds], "distances": [keep_d]}

            def _build_allowed_sets(types: List[str]) -> set[str]:
                allowed: set[str] = set()
                if "image" in types:
                    allowed.add("image")
                if "video" in types:
                    allowed.add("video_frame")
                if "text" in types:
                    allowed.add("text")
                return allowed

            def _build_video_gallery_items(filtered_vid_res: dict | None, sort_mode_key: str) -> Tuple[List[tuple[Any, str]], List[str]]:
                video_data: Dict[str, Dict[str, Any]] = {}
                if filtered_vid_res and filtered_vid_res.get("metadatas"):
                    vmetas = filtered_vid_res.get("metadatas", [[]])[0]
                    vdists = filtered_vid_res.get("distances", [[]])[0]
                    for md, d in zip(vmetas, vdists):
                        vp = md.get("video_path") or md.get("path") or ""
                        if not vp:
                            continue
                        try:
                            dist = float(d)
                        except Exception:
                            dist = 0.0
                        data = video_data.setdefault(
                            vp,
                            {
                                "video_path": vp,
                                "frames": [],
                                "best_md": None,
                                "best_dist": float("inf"),
                                "mtime": _metadata_modified_time(md),
                            },
                        )
                        data["frames"].append((md, dist))
                        if dist < data["best_dist"]:
                            data["best_dist"] = dist
                            data["best_md"] = md

                video_entries = list(video_data.values())
                if sort_mode_key == "mtime_desc":
                    video_entries.sort(key=lambda item: item["mtime"], reverse=True)
                else:
                    video_entries.sort(key=lambda item: item["best_dist"])

                video_group_items: List[tuple[Any, str]] = []
                video_paths_order: List[str] = []
                for entry in video_entries:
                    best_md = entry.get("best_md")
                    if not best_md:
                        continue
                    thumb = best_md.get("thumb_path") or best_md.get("path")
                    img_obj = None
                    try:
                        if thumb and os.path.isfile(thumb):
                            with Image.open(thumb) as im:
                                img_obj = im.convert("RGB")
                    except Exception:
                        img_obj = None
                    if img_obj is None:
                        try:
                            img_path = _get_or_create_video_frame_cached_path(
                                entry["video_path"], int(best_md.get("frame_index", 0))
                            )
                            if img_path and os.path.isfile(img_path):
                                with Image.open(img_path) as im:
                                    img_obj = im.convert("RGB")
                        except Exception:
                            img_obj = None
                    score = 1.0 - float(entry.get("best_dist", 0.0))
                    caption = (
                        f"video | {os.path.basename(entry['video_path'])} | "
                        f"frames={len(entry['frames'])} | score={score:.3f}"
                    )
                    display = img_obj if img_obj is not None else entry["video_path"]
                    video_group_items.append((display, caption))
                    video_paths_order.append(entry["video_path"])
                return video_group_items, video_paths_order

            def _render_results(raw_res: Dict[str, Any], types: List[str], sort_mode_key: str):
                allowed = _build_allowed_sets(types)
                filtered_img_res = _filter_collection(raw_res.get("image"), allowed)
                filtered_vid_res = _filter_collection(raw_res.get("video"), allowed)
                imgs_items = format_results(filtered_img_res, sort_mode_key)
                video_group_items, order = _build_video_gallery_items(filtered_vid_res, sort_mode_key)
                return imgs_items, video_group_items, order

            def do_search(persist_dir: str, q: str, img_path: str | None, k: int, types: List[str], sort_choice: str):
                gr.set_static_paths(persist_dir)
                res: Dict[str, Any] | None = None
                rag = get_rag(persist_dir)
                if img_path and os.path.isfile(img_path):
                    try:
                        with Image.open(img_path) as im:
                            img = im.convert("RGB")
                        res = rag.search_by_image(img, k=int(k))
                    except Exception:
                        res = None
                if res is None and q:
                    res = rag.search_by_text(q, k=int(k))
                if res is None:
                    return (
                        gr.update(value=[], visible=False),
                        gr.update(value=[], visible=False),
                        gr.update(value=None, visible=False),
                        None,
                    )

                sort_mode_key = _resolve_sort_mode(sort_choice)
                imgs_items, video_group_items, video_paths_order = _render_results(res, types, sort_mode_key)

                # cached_res now stores tuple of raw multi-collection res plus ordered video paths
                show_images = "image" in types and len(imgs_items) > 0
                show_videos = "video" in types and len(video_group_items) > 0
                img_update = gr.update(value=imgs_items, visible=show_images)
                vid_update = gr.update(value=video_group_items, visible=show_videos)
                frames_update = gr.update(value=None, visible=False)
                return img_update, vid_update, frames_update, (res, video_paths_order)
            search_btn.click(
                fn=do_search,
                inputs=[persist_dir_box, query, image_file, topk, type_filter, sort_mode],
                outputs=[images_gallery, videos_gallery, video_frames_gallery, cached_res]
            )

            # Re-render gallery when filter changes, without re-running the search
            def _refresh_from_cache(types: List[str], sort_choice: str, res_state: Dict[str, Any] | None):
                if not res_state:
                    return (
                        gr.update(value=[], visible=False),
                        gr.update(value=[], visible=False),
                        gr.update(value=None, visible=False),
                        None,
                    )
                raw_res, _ = res_state if isinstance(res_state, tuple) else (res_state, [])
                sort_mode_key = _resolve_sort_mode(sort_choice)
                imgs_items, video_group_items, order = _render_results(raw_res, types, sort_mode_key)
                show_images = "image" in types and len(imgs_items) > 0
                show_videos = "video" in types and len(video_group_items) > 0
                img_update = gr.update(value=imgs_items, visible=show_images)
                vid_update = gr.update(value=video_group_items, visible=show_videos)
                frames_update = gr.update(value=None, visible=False)
                return img_update, vid_update, frames_update, (raw_res, order)

            type_filter.change(
                fn=_refresh_from_cache,
                inputs=[type_filter, sort_mode, cached_res],
                outputs=[images_gallery, videos_gallery, video_frames_gallery, cached_res]
            )

            sort_mode.change(
                fn=_refresh_from_cache,
                inputs=[type_filter, sort_mode, cached_res],
                outputs=[images_gallery, videos_gallery, video_frames_gallery, cached_res]
            )

            # When a file is uploaded via gr.File, create/update a preview image.
            def _on_file_upload(fp: str | None):
                if not fp or not os.path.isfile(fp):
                    return None
                # If HEIF/HEIC/AVIF, convert to cached JPG for preview; otherwise return original path
                ext = os.path.splitext(fp)[1].lower()
                if ext in _HEIF_EXTS:
                    return _convert_for_gallery_if_needed(fp)
                # For other images, just return the file path; non-image files will fail to display
                return fp

            image_file.change(fn=_on_file_upload, inputs=[image_file], outputs=[preview])

            # Handle selection in gallery to capture the underlying file path
            def _on_image_select(types: List[str], sort_choice: str, res_state: Dict[str, Any] | None, evt: gr.SelectData):
                if not res_state or evt is None:
                    return "", None
                raw_res, _ = res_state if isinstance(res_state, tuple) else (res_state, [])
                allowed = _build_allowed_sets(types)
                filtered_img_res = _filter_collection(raw_res.get("image"), allowed)
                sort_mode_key = _resolve_sort_mode(sort_choice)
                md_pairs = _sorted_metadata_with_distance(filtered_img_res, sort_mode_key)
                try:
                    idx = int(evt.index)
                except Exception:
                    idx = -1
                if idx < 0 or idx >= len(md_pairs):
                    return "", None
                md, _ = md_pairs[idx]
                path = md.get("path")
                if not path:
                    return "", None
                return f"image -> {path}", path

            def _on_video_group_select(types: List[str], res_state: Dict[str, Any] | None, evt: gr.SelectData):
                if not res_state or evt is None:
                    return "", None, gr.update(visible=False)
                raw_res, video_paths = res_state if isinstance(res_state, tuple) else (res_state, [])
                try:
                    idx = int(evt.index)
                except Exception:
                    idx = -1
                if idx < 0 or idx >= len(video_paths):
                    return "", None, gr.update(visible=False)
                vp = video_paths[idx]
                vid_res = raw_res.get("video") if isinstance(raw_res, dict) else None
                if not vid_res:
                    return "", None, gr.update(visible=False)
                all_mds = vid_res.get("metadatas", [[]])[0]
                all_dists = vid_res.get("distances", [[]])[0]
                frame_items: List[Tuple[int | None, float | None, Any, str]] = []
                for md, d in zip(all_mds, all_dists):
                    if md.get("kind") != "video_frame":
                        continue
                    if (md.get("video_path") or md.get("path")) != vp:
                        continue
                    thumb = md.get("thumb_path") or md.get("path")
                    img_obj = None
                    try:
                        if thumb and os.path.isfile(thumb):
                            with Image.open(thumb) as im:
                                img_obj = im.convert("RGB")
                    except Exception:
                        img_obj = None
                    if img_obj is None:
                        try:
                            img_path = _get_or_create_video_frame_cached_path(vp, int(md.get("frame_index", 0)))
                            if img_path and os.path.isfile(img_path):
                                with Image.open(img_path) as im:
                                    img_obj = im.convert("RGB")
                        except Exception:
                            img_obj = None
                    score = 1.0 - float(d)
                    base = os.path.basename(vp)
                    fidx = md.get("frame_index", None)
                    ts = md.get("timestamp", None)
                    pos = f"frame {fidx}" if fidx is not None else "frame"
                    if isinstance(ts, (int, float)):
                        pos += f" @ {ts:.2f}s"
                    caption = f"{base} | {pos} | score={score:.3f}"
                    display = img_obj if img_obj is not None else vp
                    frame_items.append((md.get("frame_index"), md.get("timestamp"), display, caption))
                frame_items.sort(
                    key=lambda item: (
                        item[0] if isinstance(item[0], int) else float("inf"),
                        item[1] if isinstance(item[1], (int, float)) else float("inf"),
                    )
                )
                frames = [(display, caption) for _, _, display, caption in frame_items]
                return f"video -> {vp}", vp, gr.update(visible=True, value=frames)

            images_gallery.select(
                fn=_on_image_select,
                inputs=[type_filter, sort_mode, cached_res],
                outputs=[selected_info, selected_path]
            )

            videos_gallery.select(
                fn=_on_video_group_select,
                inputs=[type_filter, cached_res],
                outputs=[selected_info, selected_path, video_frames_gallery]
            )

            # Cross-platform open in file manager
            def _open_in_file_manager(path: str | None):
                if not path:
                    return "No selection."
                p = os.path.abspath(path)
                if not os.path.exists(p):
                    return f"Path does not exist: {p}"
                try:
                    if sys.platform.startswith("win"):
                        # Open Explorer with the file selected if possible
                        # Use explorer /select,"path"
                        subprocess.run(["explorer", "/select,", p], check=False)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", "-R", p], check=False)
                    else:
                        # Fallback: open containing directory on Linux
                        dirp = p if os.path.isdir(p) else os.path.dirname(p)
                        subprocess.run(["xdg-open", dirp], check=False)
                    return f"Opened in file manager: {p}"
                except Exception as e:
                    return f"Failed to open: {p} (error: {e})"

            open_btn.click(fn=_open_in_file_manager, inputs=[selected_path], outputs=[open_status])

    return demo


def main() -> None:
    demo = build_interface()
    # Allow serving files from our HEIF/HEIC JPG cache directory as well
    demo.launch(allowed_paths=[os.path.abspath(os.curdir)])


if __name__ == "__main__":
    main()
