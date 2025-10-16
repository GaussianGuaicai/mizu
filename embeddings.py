from __future__ import annotations

from typing import List, Tuple, Optional, Union, Any, cast
"""
Embeddings using Hugging Face transformers only.

Model: jinaai/jina-embeddings-v4 (loaded via transformers with trust_remote_code=True)

Dependencies:
- transformers
- torch (CUDA optional)

This module intentionally avoids sentence-transformers and calls the model's
custom encode helpers (provided by the model implementation when using
`trust_remote_code=True`) such as `encode_text` / `encode_image`.

Quantization (experimental):
- You can enable lightweight quantized loading (via bitsandbytes/BitsAndBytes
    integration in Transformers) by setting `quantize=True` when constructing
    `EmbeddingModel`. The code configures a `BitsAndBytesConfig` to request a
    lower-precision/efficient load path. Make sure the `bitsandbytes` package
    (and Transformers support) is installed in your environment if you enable
    this option.

Notes:
- This file updates model loading and docstrings only; runtime behavior is
    unchanged. The model's `encode_*` helpers are invoked directly and the
    wrapper returns numpy float32 arrays.
"""
import numpy as np
from PIL import Image

import cv2
import torch
from transformers import AutoModel, BitsAndBytesConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model


class EmbeddingModel:
    """
    Thin wrapper around Hugging Face transformers for multi-modal embeddings.

    Uses `jinaai/jina-embeddings-v4` via transformers with `trust_remote_code=True`,
    which provides a convenient `.encode(...)` API for text and images.

    Notes:
    - Loads to CUDA automatically if available, otherwise CPU.
    - Returns L2-normalized float32 numpy arrays by default.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: Optional[str] = None,
        max_length: int = 8192,
        task: str = "retrieval",
        *,
        quantize: bool = False,
        enable_flash_atten: bool = False,
    ) -> None:
        # Device & dtype selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        quantization_config = None
        if quantize:
            try:
                # Configure quantized/efficient loading via BitsAndBytesConfig.
                # This requests an 8-bit/optimized load path from Transformers
                # (which in turn relies on the `bitsandbytes` package at runtime).
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                # Provide a clearer hint when optional dependency is missing
                raise RuntimeError(
                    "Failed to create BitsAndBytesConfig. Ensure 'bitsandbytes' is installed (e.g., 'pip install bitsandbytes')."
                ) from e

        if quantization_config is not None:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if enable_flash_atten else None
            )  # type: JinaEmbeddingsV4Model
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if enable_flash_atten else None
            )  # type: JinaEmbeddingsV4Model
        self.model.to(self.device) # pyright: ignore[reportArgumentType]
        self.model.eval()

        self.max_length = max_length
        self.task = task

        torch.cuda.empty_cache()

    @torch.inference_mode()
    def embed_text(self, texts: List[str], batch_size = 4):
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        embs = self.model.encode_text(
            texts,
            max_length=self.max_length,
            task=self.task,
            batch_size=min(batch_size, len(texts))
        )
        if isinstance(embs, torch.Tensor):
            embs = [embs]

        torch.cuda.empty_cache()
        return [emb.detach().cpu().numpy() for emb in embs]

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image], batch_size = 4):
        embs = self.model.encode_image(
            images, # pyright: ignore[reportArgumentType]
            task=self.task,
            batch_size=min(batch_size, len(images))
        )
        if isinstance(embs, torch.Tensor):
            embs = [embs]

        torch.cuda.empty_cache()
        return [emb.detach().cpu().numpy() for emb in embs]

    @torch.inference_mode()
    def embed_video_frames(self, frames: List[np.ndarray], batch_size = 4) -> List[np.ndarray]:
        """
        Compute per-frame embeddings for a video.

        frames: list of HxWxC arrays in RGB order.
        returns: list of (D,) vectors, one per input frame.
        """
        if not frames:
            raise ValueError("No frames provided for video embedding")
        images = [Image.fromarray(f) for f in frames]
        frame_embs = self.embed_images(images, batch_size=batch_size)
        return frame_embs


def sample_video_frames(
    path: str,
    max_frames: int | None = 16,
    min_interval_sec: float = 3.0,
) -> Tuple[List[np.ndarray], float, int, List[int], float]:
    """
    Sample frames from a video using a default 3-second interval (configurable via
    min_interval_sec). The number of samples is derived from the estimated video
    duration and then capped by max_frames. Sampling prefers exact frame indices
    when total frame count and fps are known; otherwise, it performs timestamp-based
    seeking. Falls back to sequential sampling when metadata is unavailable.

    Returns (frames_rgb, duration_sec_est, total_frames_est, sampled_indices, fps)
    where sampled_indices align exactly with the returned frames (only successful
    reads are included). Indices represent frame numbers when available; otherwise,
    they may be approximations.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    # Gather metadata
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(total_frames_raw) if total_frames_raw and total_frames_raw > 0 else 0

    # Estimate duration: prefer total_frames/fps, otherwise try seeking to end and reading POS_MSEC
    duration = float(total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0
    if duration <= 0.0:
        try:
            # Save current position
            curr_ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
            # Seek to end (ratio 1.0), then read POS_MSEC for duration
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            # Some backends require a read after seek to update POS_MSEC; try a non-fatal read
            _ = cap.read()
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            duration = pos_msec / 1000.0 if pos_msec > 0 else 0.0
            # Reset to start
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, curr_ratio if curr_ratio is not None else 0.0)
        except Exception:
            duration = 0.0

    # Normalize interval
    interval = float(min_interval_sec) if min_interval_sec and min_interval_sec > 0 else 3.0

    # Compute desired number of samples from duration
    if duration > 0:
        # e.g., 0s -> 1 sample; 3.1s with 3s interval -> floor(3.1/3)+1 = 2 samples
        samples_from_duration = int(np.floor(duration / interval)) + 1
        sample_count = max(1, samples_from_duration)
    else:
        # No duration: choose up to max_frames later via fallback strategy
        sample_count = max(1, int(max_frames) if (max_frames and max_frames > 0) else 1)

    # Cap by max_frames if provided (>0). If max_frames is None or 0 -> no cap (use duration-derived)
    if max_frames is not None and int(max_frames) > 0:
        sample_count = min(sample_count, int(max_frames))
    else:
        sample_count = max(1, sample_count)

    frames: List[np.ndarray] = []
    used_indices: List[int] = []

    # Helper: append frame if read OK and convert to RGB
    def _append_frame(idx_hint: Optional[int] = None) -> None:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        if idx_hint is not None:
            used_indices.append(int(idx_hint))
        else:
            # Try to infer current frame index from capture; may be float
            pos_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos_f is not None and pos_f > 0:
                used_indices.append(int(max(0, int(round(pos_f - 1)))))
            else:
                used_indices.append(len(frames) - 1)

    # Strategy A: Have total_frames and fps -> sample by frame indices computed from timestamps
    if total_frames > 0 and fps > 0 and duration > 0:
        # Build timestamps [0, duration) with given interval
        timestamps: List[float] = []
        t = 0.0
        # Avoid seeking exactly at duration to prevent EOF read issues
        max_t = max(0.0, duration - (1.0 / max(fps, 1.0)))
        while len(timestamps) < sample_count and t <= max_t + 1e-6:
            timestamps.append(t)
            t += interval
        # If we still have fewer than requested due to very short duration, pad last near end
        if len(timestamps) < sample_count and duration > 0:
            timestamps.append(min(max_t, duration))

        # Map timestamps to frame indices and deduplicate while preserving order
        target_indices: List[int] = []
        seen = set()
        for ts in timestamps:
            idx = int(round(ts * fps))
            idx = max(0, min(total_frames - 1, idx))
            if idx not in seen:
                target_indices.append(idx)
                seen.add(idx)
            if len(target_indices) >= sample_count:
                break

        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            _append_frame(idx_hint=int(idx))

    # Strategy B: Have duration but not reliable frame count -> sample by timestamps
    elif duration > 0:
        timestamps = []
        t = 0.0
        # When fps unknown, just avoid exact end by small epsilon
        max_t = max(0.0, duration - 1e-3)
        while len(timestamps) < sample_count and t <= max_t + 1e-9:
            timestamps.append(t)
            t += interval
        if len(timestamps) < sample_count and duration > 0:
            timestamps.append(min(max_t, duration))

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts * 1000.0))
            # idx_hint attempt: derive from POS_FRAMES if available after read
            before_read_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            pos_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos_f is not None and pos_f > 0:
                used_indices.append(int(max(0, int(round(pos_f - 1)))))
            elif before_read_pos is not None and before_read_pos > 0:
                used_indices.append(int(max(0, int(round(before_read_pos)))))
            else:
                # Fallback to rough estimate using fps (if available) or sequence index
                if fps > 0:
                    approx_idx = int(max(0, min(int(round(ts * fps)), (total_frames - 1) if total_frames > 0 else int(round(ts * fps)))))
                    used_indices.append(approx_idx)
                else:
                    used_indices.append(len(frames) - 1)

    # Strategy C: No duration metadata -> fallback sequentially from start
    else:
        # Reset to start just in case
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(sample_count):
            before_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            if before_pos is not None and before_pos >= 0:
                used_indices.append(int(before_pos))
            else:
                used_indices.append(len(frames) - 1)

    cap.release()
    return frames, float(duration), int(total_frames), used_indices, float(fps)
