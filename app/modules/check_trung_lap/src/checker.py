from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from src.chunking import chunk_text
from src.config import COS_CEIL, COS_COVER, COS_FLOOR, COS_HIT, MATCH_KEEP_MIN, SEARCH_K
from src.database import Database
from src.embeddings import Embedder, ProgressCb
from src.fingerprints import content_hash, minhash_jaccard, minhash_signature
from src.plot_fp import PlotEncoder, fingerprint_sim, loads as load_plot_fp
from src.vector_store import VectorStore


@dataclass
class SpanHit:
    score: float
    query_excerpt: str
    source_excerpt: str


@dataclass
class Match:
    document_id: int
    notebook_id: int
    notebook_name: str
    title: str
    score: float
    exact: bool
    lexical: float
    semantic_max: float
    coverage: float
    kind: str


@dataclass
class CheckResult:
    title: str
    text: str
    source_path: str | None
    char_count: int
    elapsed: float
    overall_score: float
    verdict: str
    matches: list[Match] = field(default_factory=list)
    top_spans: list[SpanHit] = field(default_factory=list)
    note: str = ""


def calibrate_cos(sim: float) -> float:
    if sim <= COS_FLOOR:
        return 0.0
    return float(min(1.0, (sim - COS_FLOOR) / (COS_CEIL - COS_FLOOR)))


def combine_score(
    *,
    exact: bool,
    jaccard: float,
    max_cos: float,
    mean_cos: float,
    coverage: float,
    plot_sim: float = 0.0,
) -> float:
    if exact:
        return 100.0
        
    max_cal = calibrate_cos(max_cos)
    mean_cal = calibrate_cos(mean_cos)

    # Đưa về thang 0-1
    L = max(0.0, min(1.0, jaccard))
    M = max(0.0, min(1.0, max_cal))
    A = max(0.0, min(1.0, mean_cal))
    C = max(0.0, min(1.0, coverage))
    # Peak cũ được tính bằng max_cal * 0.90, nên ta chia 0.90 ở dưới.
    P = max(0.0, min(1.0, max_cal * 0.90))
    G = max(0.0, min(1.0, plot_sim))

    # Gộp Peak và max_cos
    H = max(M, min(1.0, P / 0.90))

    # Điểm tương đồng ngữ nghĩa cục bộ
    S = 0.05 * H + 0.35 * A + 0.60 * C

    # Điểm trùng lặp trực tiếp
    B = 0.40 * L + 0.60 * S

    # Bằng chứng hỗ trợ cho Plot Similarity
    E = 0.60 * A + 0.25 * C + 0.15 * L

    # Cổng kích hoạt Plot
    T = max(0.0, min(1.0, (E - 0.30) / 0.40))

    # Plot sau xác nhận
    Q = G * T

    # Plot bổ sung tối đa 35% phần điểm còn thiếu
    F = B + 0.35 * (1.0 - B) * Q

    return round(
        100.0 * max(0.0, min(1.0, F)),
        2
    )


def classify_kind(
    *,
    exact: bool,
    score: float,
    jaccard: float,
    max_cos: float,
    coverage: float,
    plot_sim: float = 0.0,
) -> str:
    if exact or (jaccard >= 0.92 and max_cos >= 0.985):
        return "Nguyên bản"
    if jaccard >= 0.50 or max_cos >= 0.97:
        return "Gần giống"
    if plot_sim >= 0.68 and jaccard < 0.40:
        return "Cùng cốt"
    if coverage >= 0.22 and max_cos >= COS_COVER:
        return "Viết lại"
    return "Hơi giống"


def _clip(text: str, n: int = 220) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def verdict_for(score: float, warn: float, block: float) -> str:
    if score >= block:
        return "block"
    if score >= warn:
        return "warn"
    return "ok"


class Checker:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        store: VectorStore,
        plot: PlotEncoder | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.store = store
        self.plot = plot

    def check(
        self,
        text: str,
        title: str,
        *,
        notebook_id: int | None,
        warn: float,
        block: float,
        source_path: str | None = None,
        progress: ProgressCb | None = None,
    ) -> CheckResult:
        started = perf_counter()
        text = text.strip()
        if not text:
            raise ValueError("Nội dung trống.")

        scoped_empty = self.store.n == 0
        if notebook_id is not None and not scoped_empty:
            with self.store._lock:
                scoped_empty = not bool(
                    (self.store.nb[: self.store.n] == notebook_id).any()
                )
        if scoped_empty:
            if progress:
                progress("Sổ/kho trống, bỏ qua so khớp.", 1.0)
            return CheckResult(
                title=title,
                text=text,
                source_path=source_path,
                char_count=len(text),
                elapsed=perf_counter() - started,
                overall_score=0.0,
                verdict="ok",
                matches=[],
                note="Sổ/kho đang trống — chưa có truyện để so. Có thể thêm luôn.",
            )

        if progress:
            progress("Đang lấy dấu vân nội dung…", 0.05)

        digest = content_hash(text)
        sig = minhash_signature(text)
        exact_rows = self.db.find_by_hash(digest, notebook_id)
        chunks = chunk_text(text)
        if progress:
            progress(f"Tách {len(chunks)} đoạn, đang nhúng…", 0.12)
        embs = self.embedder.encode_passages(chunks, progress=progress)
        q_plot = self.plot.encode(embs) if self.plot is not None else None

        scores_mat, ids_mat = self.store.search(embs, notebook_id=notebook_id, k=SEARCH_K)
        meta = self.store.meta_for_ids(ids_mat)

        span_raw: list[tuple[float, int, int]] = []
        for q_idx in range(len(chunks)):
            for j in range(ids_mat.shape[1]):
                cid = int(ids_mat[q_idx, j])
                sc = float(scores_mat[q_idx, j])
                if cid < 0 or sc < 0:
                    continue
                span_raw.append((sc, q_idx, cid))
        span_raw.sort(key=lambda x: x[0], reverse=True)
        chunk_texts = self.db.get_chunk_texts([cid for _, _, cid in span_raw[:48]])
        top_spans: list[SpanHit] = []
        seen_span: set[tuple[int, int]] = set()
        for sc, q_idx, cid in span_raw:
            key = (q_idx, cid)
            if key in seen_span:
                continue
            seen_span.add(key)
            src = chunk_texts.get(cid)
            if not src:
                continue
            top_spans.append(
                SpanHit(
                    score=round(calibrate_cos(sc) * 100.0, 1),
                    query_excerpt=_clip(chunks[q_idx]),
                    source_excerpt=_clip(src),
                )
            )
            if len(top_spans) >= 5:
                break

        per_doc_chunk: dict[int, list[float]] = defaultdict(lambda: [0.0] * len(chunks))
        for q_idx in range(len(chunks)):
            best_for_q: dict[int, float] = {}
            for j in range(scores_mat.shape[1]):
                cid = int(ids_mat[q_idx, j])
                sc = float(scores_mat[q_idx, j])
                if cid < 0 or sc < COS_HIT:
                    continue
                info = meta.get(cid)
                if not info:
                    continue
                doc_id, nb_id = info
                if notebook_id is not None and nb_id != notebook_id:
                    continue
                if sc > best_for_q.get(doc_id, 0.0):
                    best_for_q[doc_id] = sc
            for doc_id, sc in best_for_q.items():
                per_doc_chunk[doc_id][q_idx] = sc

        lexical: dict[int, tuple[float, int, str, str]] = {}
        for doc_id, nb_id, doc_title, nb_name, blob in self.db.iter_minhashes(notebook_id):
            jac = minhash_jaccard(sig, blob)
            if jac >= 0.18 or doc_id in per_doc_chunk:
                lexical[doc_id] = (jac, nb_id, doc_title, nb_name)

        plot_map: dict[int, float] = {}
        if q_plot is not None:
            for doc_id, blob in self.db.iter_plot_fps(notebook_id):
                fp = load_plot_fp(blob)
                if fp is None:
                    continue
                plot_map[int(doc_id)] = fingerprint_sim(q_plot, fp)

        cover_thr = COS_COVER if len(chunks) >= 5 else min(COS_COVER, 0.60)

        matches: list[Match] = []
        seen: set[int] = set()
        candidates = set(per_doc_chunk) | set(lexical) | {r["id"] for r in exact_rows}
        candidates |= {did for did, ps in plot_map.items() if ps >= 0.62}
        for doc_id in candidates:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            exact = any(r["id"] == doc_id for r in exact_rows)
            jac = lexical.get(doc_id, (0.0, -1, "", ""))[0]
            row = self.db.get_document(int(doc_id))
            if not row:
                continue
            if notebook_id is not None and int(row["notebook_id"]) != notebook_id:
                continue
            arr = np.asarray(per_doc_chunk.get(int(doc_id), [0.0] * max(len(chunks), 1)), dtype=np.float32)
            max_cos = float(arr.max()) if arr.size else 0.0
            mean_cos = float(arr.mean()) if arr.size else 0.0
            coverage = float(np.mean(arr >= cover_thr)) if arr.size else 0.0
            plot_sim = plot_map.get(int(doc_id), 0.0)
            score = combine_score(
                exact=exact,
                jaccard=jac,
                max_cos=max_cos,
                mean_cos=mean_cos,
                coverage=coverage,
                plot_sim=plot_sim,
            )
            if score < MATCH_KEEP_MIN and not exact:
                continue
            matches.append(
                Match(
                    document_id=int(row["id"]),
                    notebook_id=int(row["notebook_id"]),
                    notebook_name=str(row["notebook_name"]),
                    title=str(row["title"]),
                    score=score,
                    exact=exact,
                    lexical=round(jac * 100.0, 1),
                    semantic_max=round(max_cos * 100.0, 1),
                    coverage=round(coverage * 100.0, 1),
                    kind=classify_kind(
                        exact=exact,
                        score=score,
                        jaccard=jac,
                        max_cos=max_cos,
                        coverage=coverage,
                        plot_sim=plot_sim,
                    ),
                )
            )

        matches.sort(key=lambda m: m.score, reverse=True)
        overall = matches[0].score if matches else 0.0
        verdict = verdict_for(overall, warn, block)
        elapsed = perf_counter() - started
        if progress:
            progress("Xong.", 1.0)
        return CheckResult(
            title=title,
            text=text,
            source_path=source_path,
            char_count=len(text),
            elapsed=elapsed,
            overall_score=overall,
            verdict=verdict,
            matches=matches[:20],
            top_spans=top_spans,
            note="",
        )
