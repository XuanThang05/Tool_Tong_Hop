from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from src.config import EMBED_DIM, INDEX_DIR, ensure_dirs

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None  # type: ignore[assignment]


class VectorStore:
    GROW = 8000

    def __init__(self, folder: Path = INDEX_DIR, dim: int = EMBED_DIM) -> None:
        ensure_dirs()
        self.folder = folder
        self.dim = dim
        self._lock = threading.RLock()
        self.n = 0
        self.cap = self.GROW
        self.ids = np.zeros(self.cap, dtype=np.int64)
        self.nb = np.zeros(self.cap, dtype=np.int64)
        self.doc = np.zeros(self.cap, dtype=np.int64)
        self.vecs = np.zeros((self.cap, dim), dtype=np.float32)
        self._index = None
        self._load()

    @property
    def meta_path(self) -> Path:
        return self.folder / "meta.json"

    @property
    def ids_path(self) -> Path:
        return self.folder / "ids.npy"

    @property
    def nb_path(self) -> Path:
        return self.folder / "notebooks.npy"

    @property
    def doc_path(self) -> Path:
        return self.folder / "docs.npy"

    @property
    def vec_path(self) -> Path:
        return self.folder / "vectors.npy"

    def _load(self) -> None:
        if not self.ids_path.exists() or not self.vec_path.exists():
            self._rebuild_faiss()
            return
        ids = np.load(self.ids_path)
        nb = np.load(self.nb_path)
        doc = np.load(self.doc_path)
        vecs = np.load(self.vec_path)
        if vecs.ndim != 2 or int(vecs.shape[1]) != self.dim:
            self.n = 0
            self._rebuild_faiss()
            return
        n = int(len(ids))
        self.cap = max(self.GROW, n + self.GROW)
        self.ids = np.zeros(self.cap, dtype=np.int64)
        self.nb = np.zeros(self.cap, dtype=np.int64)
        self.doc = np.zeros(self.cap, dtype=np.int64)
        self.vecs = np.zeros((self.cap, self.dim), dtype=np.float32)
        if n:
            self.ids[:n] = ids
            self.nb[:n] = nb
            self.doc[:n] = doc
            self.vecs[:n] = vecs
        self.n = n
        self._rebuild_faiss()

    def _grow(self, need: int) -> None:
        if need <= self.cap:
            return
        new_cap = max(need + self.GROW, self.cap * 2)
        def _copy(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            arr = np.zeros(shape, dtype=old.dtype)
            arr[: self.n] = old[: self.n]
            return arr

        self.ids = _copy(self.ids, (new_cap,))
        self.nb = _copy(self.nb, (new_cap,))
        self.doc = _copy(self.doc, (new_cap,))
        self.vecs = _copy(self.vecs, (new_cap, self.dim))
        self.cap = new_cap

    def _rebuild_faiss(self) -> None:
        if faiss is None:
            self._index = None
            return
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        if self.n > 0:
            index.add_with_ids(
                np.ascontiguousarray(self.vecs[: self.n]),
                np.ascontiguousarray(self.ids[: self.n]),
            )
        self._index = index

    def save(self) -> None:
        with self._lock:
            ensure_dirs()
            np.save(self.ids_path, self.ids[: self.n])
            np.save(self.nb_path, self.nb[: self.n])
            np.save(self.doc_path, self.doc[: self.n])
            np.save(self.vec_path, self.vecs[: self.n])
            self.meta_path.write_text(
                json.dumps({"n": self.n, "dim": self.dim}, ensure_ascii=False),
                encoding="utf-8",
            )

    def add(
        self,
        chunk_ids: list[int],
        notebook_id: int,
        document_id: int,
        vectors: np.ndarray,
    ) -> None:
        if not chunk_ids:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError("Số vector không khớp số đoạn.")
        with self._lock:
            k = len(chunk_ids)
            self._grow(self.n + k)
            sl = slice(self.n, self.n + k)
            self.ids[sl] = np.asarray(chunk_ids, dtype=np.int64)
            self.nb[sl] = notebook_id
            self.doc[sl] = document_id
            self.vecs[sl] = vectors
            self.n += k
            if self._index is not None:
                self._index.add_with_ids(
                    vectors, np.asarray(chunk_ids, dtype=np.int64)
                )
            self.save()

    def clear(self) -> None:
        with self._lock:
            self.n = 0
            self.cap = self.GROW
            self.ids = np.zeros(self.cap, dtype=np.int64)
            self.nb = np.zeros(self.cap, dtype=np.int64)
            self.doc = np.zeros(self.cap, dtype=np.int64)
            self.vecs = np.zeros((self.cap, self.dim), dtype=np.float32)
            self._rebuild_faiss()
            self.save()

    def remove_document(self, document_id: int) -> None:
        with self._lock:
            if self.n == 0:
                return
            keep = self.doc[: self.n] != document_id
            if keep.all():
                return
            m = int(keep.sum())
            self.ids[:m] = self.ids[: self.n][keep]
            self.nb[:m] = self.nb[: self.n][keep]
            self.doc[:m] = self.doc[: self.n][keep]
            self.vecs[:m] = self.vecs[: self.n][keep]
            self.n = m
            self._rebuild_faiss()
            self.save()

    def remove_notebook(self, notebook_id: int) -> None:
        with self._lock:
            if self.n == 0:
                return
            keep = self.nb[: self.n] != notebook_id
            if keep.all():
                return
            m = int(keep.sum())
            self.ids[:m] = self.ids[: self.n][keep]
            self.nb[:m] = self.nb[: self.n][keep]
            self.doc[:m] = self.doc[: self.n][keep]
            self.vecs[:m] = self.vecs[: self.n][keep]
            self.n = m
            self._rebuild_faiss()
            self.save()

    def search(
        self,
        queries: np.ndarray,
        notebook_id: int | None = None,
        k: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, chunk_ids) each shape (n_query, k). Missing slots are -1."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        empty_s = np.full((nq, k), -1.0, dtype=np.float32)
        empty_i = np.full((nq, k), -1, dtype=np.int64)
        with self._lock:
            if self.n == 0 or nq == 0:
                return empty_s, empty_i
            if notebook_id is None:
                if self._index is None:
                    return self._numpy_search(queries, np.arange(self.n), k)
                kk = min(k, self.n)
                scores, ids = self._index.search(queries, kk)
                if kk < k:
                    pad_s = np.full((nq, k), -1.0, dtype=np.float32)
                    pad_i = np.full((nq, k), -1, dtype=np.int64)
                    pad_s[:, :kk] = scores
                    pad_i[:, :kk] = ids
                    return pad_s, pad_i
                return scores.astype(np.float32), ids.astype(np.int64)

            idx = np.flatnonzero(self.nb[: self.n] == notebook_id)
            if idx.size == 0:
                return empty_s, empty_i
            return self._numpy_search(queries, idx, k)

    def _numpy_search(
        self, queries: np.ndarray, row_idx: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sub = self.vecs[row_idx]
        sims = sub @ queries.T
        n_sub, nq = sims.shape
        kk = min(k, n_sub)
        scores = np.full((nq, k), -1.0, dtype=np.float32)
        ids = np.full((nq, k), -1, dtype=np.int64)
        if kk <= 0:
            return scores, ids
        if n_sub <= kk:
            order = np.argsort(-sims, axis=0)
            for q in range(nq):
                pick = order[:kk, q]
                scores[q, :kk] = sims[pick, q]
                ids[q, :kk] = self.ids[row_idx[pick]]
            return scores, ids
        part = np.argpartition(-sims, kth=kk - 1, axis=0)[:kk]
        for q in range(nq):
            pick = part[:, q]
            ranked = pick[np.argsort(-sims[pick, q])]
            scores[q, :kk] = sims[ranked, q]
            ids[q, :kk] = self.ids[row_idx[ranked]]
        return scores, ids

    def meta_for_ids(self, chunk_ids: np.ndarray) -> dict[int, tuple[int, int]]:
        """chunk_id -> (document_id, notebook_id)"""
        with self._lock:
            mapping = {
                int(cid): (int(d), int(n))
                for cid, d, n in zip(
                    self.ids[: self.n], self.doc[: self.n], self.nb[: self.n]
                )
            }
        out: dict[int, tuple[int, int]] = {}
        for cid in chunk_ids.reshape(-1):
            icid = int(cid)
            if icid in mapping:
                out[icid] = mapping[icid]
        return out
