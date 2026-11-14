from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.checker import CheckResult, Checker
from src.chunking import chunk_text
from src.config import DEFAULT_BLOCK, DEFAULT_WARN, MODEL_NAME, ensure_dirs
from src.database import Database
from src.embeddings import Embedder, ProgressCb
from src.plot_fp import PROBE_VERSION, PlotEncoder, dumps as dump_plot_fp
from src.fingerprints import content_hash, minhash_signature
from src.text_extract import ExtractError, collect_files, extract_file
from src.vector_store import VectorStore

Progress = Callable[[str, float], None]


class AppService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AppService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        ensure_dirs()
        self.db = Database()
        self.embedder = Embedder()
        self.store = VectorStore()
        self.plot = PlotEncoder(self.embedder)
        self.checker = Checker(self.db, self.embedder, self.store, self.plot)
        self._initialized = True

    def close(self) -> None:
        self.db.close()

    def warmup(self, progress: Progress | None = None) -> None:
        self.embedder.ensure_loaded(progress)
        self._ensure_index(progress)
        self._refresh_plot_fps(progress)

    def _ensure_index(self, progress: Progress | None = None) -> None:
        stored = (self.db.settings_json() or {}).get("embed_model")
        n_chunks = self.db.stats()["chunks"]
        need = (
            stored != MODEL_NAME
            or self.store.dim != self.embedder.dim
            or self.store.n != n_chunks
        )
        if not need:
            return
        chunks = self.db.list_chunks()
        if progress:
            progress("Đang dựng lại index nghĩa (model mới)…", 0.05)
        self.store.clear()
        if not chunks:
            data = self.db.settings_json()
            data["embed_model"] = MODEL_NAME
            self.db.save_settings_json(data)
            return
        batch = 64
        total = len(chunks)
        for i in range(0, total, batch):
            part = chunks[i : i + batch]
            vecs = self.embedder.encode_passages([str(r["text"]) for r in part])
            by_doc: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for j, row in enumerate(part):
                key = (int(row["notebook_id"]), int(row["document_id"]))
                by_doc.setdefault(key, []).append((j, int(row["id"])))
            for (nb_id, doc_id), items in by_doc.items():
                idxs = [x[0] for x in items]
                ids = [x[1] for x in items]
                self.store.add(ids, nb_id, doc_id, vecs[idxs])
            if progress:
                done = min(i + len(part), total)
                progress(f"Index nghĩa {done}/{total} đoạn…", done / total)
        data = self.db.settings_json()
        data["embed_model"] = MODEL_NAME
        self.db.save_settings_json(data)

    def _refresh_plot_fps(self, progress: Progress | None = None) -> None:
        data = self.db.settings_json()
        if data.get("plot_probes") == PROBE_VERSION:
            missing = any(blob is None for _, blob in self.db.iter_plot_fps())
            if not missing:
                return
        if progress:
            progress("Đang lấy dấu vân cốt truyện…", 0.05)
        docs = self.db.iter_plot_fps()
        total = max(len(docs), 1)
        for i, (doc_id, _) in enumerate(docs, start=1):
            with self.store._lock:
                idx = np.flatnonzero(self.store.doc[: self.store.n] == doc_id)
                vecs = np.array(self.store.vecs[idx], dtype=np.float32) if idx.size else None
            if vecs is None or vecs.size == 0:
                continue
            fp = self.plot.encode(vecs)
            self.db.update_plot_fp(doc_id, dump_plot_fp(fp))
            if progress:
                progress(f"Cốt truyện {i}/{total}…", i / total)
        data = self.db.settings_json()
        data["plot_probes"] = PROBE_VERSION
        self.db.save_settings_json(data)

    def ready(self) -> bool:
        return self.embedder.ready

    def device_label(self) -> str:
        return self.embedder.device_label()

    def thresholds(self) -> tuple[int, int]:
        data = self.db.settings_json()
        warn = int(data.get("warn", DEFAULT_WARN))
        block = int(data.get("block", DEFAULT_BLOCK))
        warn = min(max(warn, 40), 99)
        block = min(max(block, warn + 1), 100)
        return warn, block

    def save_thresholds(self, warn: int, block: int) -> None:
        data = self.db.settings_json()
        data["warn"] = int(warn)
        data["block"] = int(block)
        self.db.save_settings_json(data)

    def stats(self) -> dict[str, int]:
        return self.db.stats()

    def list_notebooks(self):
        return self.db.list_notebooks()

    def create_notebook(self, name: str) -> int:
        return self.db.create_notebook(name)

    def rename_notebook(self, notebook_id: int, name: str) -> None:
        self.db.rename_notebook(notebook_id, name)

    def delete_notebook(self, notebook_id: int) -> None:
        self.db.delete_notebook(notebook_id)
        self.store.remove_notebook(notebook_id)

    def list_documents(self, notebook_id: int):
        return self.db.list_documents(notebook_id)

    def read_document(self, document_id: int) -> tuple[str, str]:
        doc = self.db.get_document(document_id)
        if not doc:
            raise ValueError("Không tìm thấy truyện.")
        return str(doc["title"]), self.db.read_document_text(document_id)

    def delete_document(self, document_id: int) -> None:
        self.db.delete_document(document_id)
        self.store.remove_document(document_id)

    def add_text(
        self,
        notebook_id: int,
        title: str,
        text: str,
        source_path: str | None = None,
        progress: Progress | None = None,
    ) -> int:
        text = text.strip()
        if not text:
            raise ValueError("Nội dung trống.")
        title = (title or "Không tên").strip()
        digest = content_hash(text)
        existing = self.db.find_by_hash(digest, notebook_id)
        if existing:
            raise ValueError(
                f"Sổ này đã có nội dung trùng nguyên bản: «{existing[0]['title']}»."
            )
        if progress:
            progress("Đang tách đoạn…", 0.08)
        chunks = chunk_text(text)
        embs = self.embedder.encode_passages(chunks, progress=progress)
        sig = minhash_signature(text)
        plot_blob = dump_plot_fp(self.plot.encode(embs))
        doc_id, chunk_ids = self.db.insert_document(
            notebook_id,
            title,
            text,
            source_path,
            digest,
            sig,
            chunks,
            plot_fp=plot_blob,
        )
        try:
            self.store.add(chunk_ids, notebook_id, doc_id, embs)
        except Exception:
            self.db.delete_document(doc_id)
            raise
        if progress:
            progress("Đã thêm vào kho.", 1.0)
        return doc_id

    def import_paths(
        self,
        notebook_id: int,
        paths: list[Path],
        progress: Progress | None = None,
    ) -> tuple[int, list[str]]:
        files = collect_files(paths)
        ok = 0
        errors: list[str] = []
        total = max(len(files), 1)
        for i, path in enumerate(files, start=1):
            if progress:
                progress(f"Thêm {i}/{len(files)}: {path.name}", i / total * 0.95)
            try:
                text = extract_file(path)
                self.add_text(notebook_id, path.stem, text, str(path))
                ok += 1
            except ExtractError as exc:
                errors.append(str(exc))
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")
        if progress:
            progress(f"Đã thêm {ok}/{len(files)} file.", 1.0)
        return ok, errors

    def check_text(
        self,
        text: str,
        title: str,
        *,
        notebook_id: int | None,
        warn: float,
        block: float,
        source_path: str | None = None,
        progress: Progress | None = None,
    ) -> CheckResult:
        return self.checker.check(
            text,
            title,
            notebook_id=notebook_id,
            warn=warn,
            block=block,
            source_path=source_path,
            progress=progress,
        )

    def check_paths(
        self,
        paths: list[Path],
        *,
        notebook_id: int | None,
        warn: float,
        block: float,
        progress: Progress | None = None,
    ) -> list[CheckResult]:
        files = collect_files(paths)
        results: list[CheckResult] = []
        total = max(len(files), 1)
        for i, path in enumerate(files, start=1):
            if progress:
                progress(f"Check {i}/{len(files)}: {path.name}", i / total * 0.15)
            try:
                text = extract_file(path)
            except ExtractError as exc:
                results.append(
                    CheckResult(
                        title=path.stem,
                        text="",
                        source_path=str(path),
                        char_count=0,
                        elapsed=0.0,
                        overall_score=0.0,
                        verdict="ok",
                        note=f"Lỗi đọc file: {exc}",
                    )
                )
                continue

            def _inner(msg: str, frac: float, _i=i, _n=len(files)) -> None:
                if progress:
                    base = (_i - 1) / _n
                    progress(f"{path.name}: {msg}", base + frac / _n)

            results.append(
                self.check_text(
                    text,
                    path.stem,
                    notebook_id=notebook_id,
                    warn=warn,
                    block=block,
                    source_path=str(path),
                    progress=_inner,
                )
            )
        return results
