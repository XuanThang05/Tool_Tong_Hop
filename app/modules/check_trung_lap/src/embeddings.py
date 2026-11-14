from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from src.config import (
    EMBED_BATCH_CPU,
    EMBED_BATCH_GPU,
    EMBED_DIM,
    HF_HOME,
    MODEL_DIR,
    MODEL_NAME,
    ensure_dirs,
)

ProgressCb = Callable[[str, float], None]


class Embedder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self.device = "cpu"
        self.model_name = MODEL_NAME
        self.dim = EMBED_DIM
        self.error: str | None = None
        self._use_e5_prefix = False

    @property
    def ready(self) -> bool:
        return self._model is not None

    def device_label(self) -> str:
        if self.device == "cuda":
            try:
                import torch

                name = torch.cuda.get_device_name(0)
                return f"GPU · {name}"
            except Exception:
                return "GPU"
        return "CPU"

    def ensure_loaded(self, progress: ProgressCb | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            ensure_dirs()
            os.environ.setdefault("HF_HOME", str(HF_HOME))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_DIR))
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            if progress:
                progress("Đang chuẩn bị model tiếng Việt…", 0.05)
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                self.error = str(exc)
                raise RuntimeError(
                    "Thiếu thư viện model. Hãy chạy setup.bat rồi mở lại app."
                ) from exc

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if progress:
                progress(f"Đang tải {MODEL_NAME} ({self.device_label()})…", 0.15)
            model = SentenceTransformer(
                MODEL_NAME,
                device=self.device,
                cache_folder=str(MODEL_DIR),
            )
            dim_fn = getattr(model, "get_embedding_dimension", None)
            if dim_fn is None:
                dim_fn = model.get_sentence_embedding_dimension
            self.dim = int(dim_fn() or EMBED_DIM)
            self._use_e5_prefix = "e5-" in MODEL_NAME.lower() or "multilingual-e5" in MODEL_NAME
            self._model = model
            if progress:
                progress("Model sẵn sàng.", 1.0)

    def encode_passages(self, texts: list[str], progress: ProgressCb | None = None) -> np.ndarray:
        self.ensure_loaded(progress)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if getattr(self, "_use_e5_prefix", False):
            payloads = [f"passage: {t}" for t in texts]
        else:
            payloads = list(texts)
        batch = EMBED_BATCH_GPU if self.device == "cuda" else EMBED_BATCH_CPU
        total = len(payloads)
        chunks: list[np.ndarray] = []
        model = self._model
        for i in range(0, total, batch):
            part = payloads[i : i + batch]
            vec = model.encode(
                part,
                batch_size=len(part),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(vec, dtype=np.float32))
            if progress:
                done = min(i + len(part), total)
                progress(f"Đang nhúng {done}/{total} đoạn…", 0.2 + 0.8 * (done / total))
        return np.vstack(chunks)
