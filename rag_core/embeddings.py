from __future__ import annotations

"""Helpers around multimodal embedding transformers.

Supported backends:
- jinaai/jina-embeddings-v4 (default)
- Qwen3-VL-Embedding models (Transformers + AutoProcessor flow)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

__all__ = ["EmbeddingModel", "sample_video_frames"]


class EmbeddingModel:
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.task = task

        quantization_config = None
        if quantize:
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to configure BitsAndBytes quantization. Install 'bitsandbytes'."
                ) from exc

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2" if enable_flash_atten else None,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)  # type: ignore[assignment]
        self.model.to(self.device)
        self.model.eval()

        self.backend = "jina"
        self.processor = None
        if "qwen3-vl-embedding" in model_name.lower():
            self.backend = "qwen3_vl_embedding"
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        torch.cuda.empty_cache()

    def _batch_slices(self, total: int, batch_size: int) -> List[slice]:
        if total <= 0:
            return []
        step = max(1, min(batch_size, total))
        return [slice(i, min(i + step, total)) for i in range(0, total, step)]

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _pool_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if embeddings.ndim == 2:
            return embeddings
        if embeddings.ndim != 3:
            raise RuntimeError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")
        if attention_mask is not None and attention_mask.ndim == 2:
            mask = attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            denom = torch.clamp(mask.sum(dim=1), min=1e-6)
            pooled = (embeddings * mask).sum(dim=1) / denom
            return pooled
        return embeddings.mean(dim=1)

    def _extract_qwen_embedding_tensor(self, outputs: Any, inputs: Dict[str, Any]) -> torch.Tensor:
        candidate = None
        for attr in (
            "embeddings",
            "text_embeds",
            "image_embeds",
            "video_embeds",
            "pooler_output",
            "last_hidden_state",
        ):
            candidate = getattr(outputs, attr, None)
            if isinstance(candidate, torch.Tensor):
                break
        if not isinstance(candidate, torch.Tensor):
            if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
                candidate = outputs[0]
            elif isinstance(outputs, dict):
                for value in outputs.values():
                    if isinstance(value, torch.Tensor):
                        candidate = value
                        break
        if not isinstance(candidate, torch.Tensor):
            raise RuntimeError("Failed to extract embeddings from Qwen3-VL-Embedding output")

        candidate = self._pool_embeddings(candidate, attention_mask=inputs.get("attention_mask"))
        return torch.nn.functional.normalize(candidate, p=2, dim=-1)

    @torch.inference_mode()
    def embed_text(self, texts: List[str], batch_size: int = 4):
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if not texts:
            return []

        if self.backend == "jina":
            embeddings = self.model.encode_text(
                texts,
                max_length=self.max_length,
                task=self.task,
                batch_size=min(batch_size, len(texts)),
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = [embeddings]
            torch.cuda.empty_cache()
            return [emb.detach().cpu().numpy() for emb in embeddings]

        if self.processor is None:
            raise RuntimeError("Qwen3-VL-Embedding processor was not initialized")

        rows: List[np.ndarray] = []
        for s in self._batch_slices(len(texts), batch_size):
            chunk = texts[s]
            inputs = self.processor(
                text=chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            moved = self._move_inputs_to_device(inputs)
            outputs = self.model(**moved)
            emb = self._extract_qwen_embedding_tensor(outputs, moved)
            rows.extend([r.detach().cpu().numpy() for r in emb])

        torch.cuda.empty_cache()
        return rows

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image], batch_size: int = 4):
        if not images:
            return []

        if self.backend == "jina":
            embeddings = self.model.encode_image(
                images,
                task=self.task,
                batch_size=min(batch_size, len(images)),
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = [embeddings]
            torch.cuda.empty_cache()
            return [emb.detach().cpu().numpy() for emb in embeddings]

        if self.processor is None:
            raise RuntimeError("Qwen3-VL-Embedding processor was not initialized")

        rows: List[np.ndarray] = []
        for s in self._batch_slices(len(images), batch_size):
            chunk = images[s]
            inputs = self.processor(
                images=chunk,
                return_tensors="pt",
                padding=True,
            )
            moved = self._move_inputs_to_device(inputs)
            outputs = self.model(**moved)
            emb = self._extract_qwen_embedding_tensor(outputs, moved)
            rows.extend([r.detach().cpu().numpy() for r in emb])

        torch.cuda.empty_cache()
        return rows

    @torch.inference_mode()
    def embed_video_frames(self, frames: List[np.ndarray], batch_size: int = 4):
        if not frames:
            raise ValueError("No frames provided for video embedding")

        if self.backend == "jina":
            images = [Image.fromarray(frame) for frame in frames]
            return self.embed_images(images, batch_size=batch_size)

        if self.processor is None:
            raise RuntimeError("Qwen3-VL-Embedding processor was not initialized")

        inputs = self.processor(
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )
        moved = self._move_inputs_to_device(inputs)
        outputs = self.model(**moved)
        emb = self._extract_qwen_embedding_tensor(outputs, moved)
        torch.cuda.empty_cache()
        return [r.detach().cpu().numpy() for r in emb]


def sample_video_frames(
    path: str,
    max_frames: Optional[int] = 16,
    min_interval_sec: float = 3.0,
) -> Tuple[List[np.ndarray], float, int, List[int], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(total_frames_raw) if total_frames_raw and total_frames_raw > 0 else 0

    duration = float(total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0
    if duration <= 0.0:
        try:
            current_ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            _ = cap.read()
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            duration = pos_msec / 1000.0 if pos_msec > 0 else 0.0
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, current_ratio if current_ratio is not None else 0.0)
        except Exception:
            duration = 0.0

    interval = float(min_interval_sec) if min_interval_sec and min_interval_sec > 0 else 3.0

    if duration > 0:
        samples_from_duration = int(np.floor(duration / interval)) + 1
        sample_count = max(1, samples_from_duration)
    else:
        sample_count = max(1, int(max_frames) if (max_frames and max_frames > 0) else 1)

    if max_frames is not None and int(max_frames) > 0:
        sample_count = min(sample_count, int(max_frames))
    else:
        sample_count = max(1, sample_count)

    frames: List[np.ndarray] = []
    used_indices: List[int] = []

    def _append_frame(idx_hint: Optional[int] = None) -> None:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        if idx_hint is not None:
            used_indices.append(int(idx_hint))
        else:
            pos_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos_f is not None and pos_f > 0:
                used_indices.append(int(max(0, int(round(pos_f - 1)))))
            else:
                used_indices.append(len(frames) - 1)

    if total_frames > 0 and fps > 0 and duration > 0:
        timestamps: List[float] = []
        t = 0.0
        max_t = max(0.0, duration - (1.0 / max(fps, 1.0)))
        while len(timestamps) < sample_count and t <= max_t + 1e-6:
            timestamps.append(t)
            t += interval
        if len(timestamps) < sample_count and duration > 0:
            timestamps.append(min(max_t, duration))

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

    elif duration > 0:
        timestamps = []
        t = 0.0
        max_t = max(0.0, duration - 1e-3)
        while len(timestamps) < sample_count and t <= max_t + 1e-9:
            timestamps.append(t)
            t += interval
        if len(timestamps) < sample_count and duration > 0:
            timestamps.append(min(max_t, duration))

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts * 1000.0))
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
                if fps > 0:
                    approx_idx = int(
                        max(
                            0,
                            min(
                                int(round(ts * fps)),
                                (total_frames - 1) if total_frames > 0 else int(round(ts * fps)),
                            ),
                        )
                    )
                    used_indices.append(approx_idx)
                else:
                    used_indices.append(len(frames) - 1)

    else:
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
