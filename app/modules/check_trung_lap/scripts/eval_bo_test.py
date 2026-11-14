from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_DIR = Path(r"c:\Users\AD\Desktop\bo_test_truyen_dai")
os.environ["TRUNGLAP_DATA"] = str(Path(tempfile.mkdtemp(prefix="trunglap-eval-")))

from src.checker import calibrate_cos, combine_score, verdict_for  # noqa: E402
from src.chunking import chunk_text  # noqa: E402
from src.config import COS_COVER, COS_HIT  # noqa: E402
from src.fingerprints import content_hash, minhash_jaccard, minhash_signature  # noqa: E402
from src.services import AppService  # noqa: E402
from src.text_extract import extract_file  # noqa: E402

FILES = [
    ("00_TRUYEN_GOC.txt", "00 nguồn"),
    ("01_BAN_SAO_GAN_NGUYEN_VAN.txt", "01 gần nguyên văn"),
    ("02_XAO_NAU_NHE.txt", "02 xào nấu nhẹ"),
    ("03_XAO_NAU_TRUNG_BINH.txt", "03 xào nấu TB"),
    ("04_VIET_LAI_SAU.txt", "04 viết lại sâu"),
    ("05_CUNG_COT_TRUYEN_DOI_BE_MAT.txt", "05 cùng cốt đổi bề mặt"),
    ("06_DOI_CHUNG_KHONG_LIEN_QUAN.txt", "06 đối chứng âm"),
]


def snippet(text: str, n: int = 180) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def diagnose(svc: AppService, notebook_id: int, text: str, title: str, warn: int, block: int) -> dict:
    started = time.perf_counter()
    official = svc.check_text(
        text, title, notebook_id=notebook_id, warn=warn, block=block
    )
    chunks = chunk_text(text)
    embs = svc.embedder.encode_passages(chunks)
    scores_mat, ids_mat = svc.store.search(embs, notebook_id=notebook_id, k=8)
    sig = minhash_signature(text)
    digest = content_hash(text)
    exact = bool(svc.db.find_by_hash(digest, notebook_id))
    jac = 0.0
    for _id, _nb, _title, _name, blob in svc.db.iter_minhashes(notebook_id):
        jac = minhash_jaccard(sig, blob)
        break

    best_per_q: list[float] = []
    spans: list[dict] = []
    chunk_cache: dict[int, str] = {}
    for q_idx, ch in enumerate(chunks):
        best = 0.0
        for j in range(ids_mat.shape[1]):
            cid = int(ids_mat[q_idx, j])
            sc = float(scores_mat[q_idx, j])
            if cid < 0 or sc < 0:
                continue
            if cid not in chunk_cache:
                row = svc.db.conn.execute(
                    "SELECT text FROM chunks WHERE id = ?", (cid,)
                ).fetchone()
                chunk_cache[cid] = str(row["text"]) if row else ""
            if sc > best:
                best = sc
            spans.append(
                {
                    "raw_cos": round(sc, 4),
                    "cal": round(calibrate_cos(sc) * 100, 1),
                    "query_i": q_idx,
                    "query": snippet(ch),
                    "source": snippet(chunk_cache[cid]),
                }
            )
        best_per_q.append(best)
    spans.sort(key=lambda x: x["raw_cos"], reverse=True)
    uniq: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for sp in spans:
        key = (sp["query_i"], sp["source"][:40])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sp)
        if len(uniq) >= 6:
            break

    import numpy as np

    arr = np.asarray(best_per_q, dtype=np.float32) if best_per_q else np.zeros(1)
    max_cos = float(arr.max()) if arr.size else 0.0
    mean_cos = float(arr.mean()) if arr.size else 0.0
    coverage = float(np.mean(arr >= COS_COVER)) if arr.size else 0.0
    hit_rate = float(np.mean(arr >= COS_HIT)) if arr.size else 0.0
    raw_score = combine_score(
        exact=exact,
        jaccard=jac,
        max_cos=max_cos,
        mean_cos=mean_cos,
        coverage=coverage,
    )
    elapsed = time.perf_counter() - started
    return {
        "title": title,
        "chars": len(text),
        "chunks": len(chunks),
        "elapsed_s": round(elapsed, 3),
        "official_score": official.overall_score,
        "official_verdict": official.verdict,
        "official_kind": official.matches[0].kind if official.matches else "—",
        "raw_score": raw_score,
        "raw_verdict": verdict_for(raw_score, warn, block),
        "exact": exact,
        "jaccard_pct": round(jac * 100, 2),
        "max_cos": round(max_cos, 4),
        "mean_best_cos": round(mean_cos, 4),
        "coverage_pct": round(coverage * 100, 1),
        "hit_rate_pct": round(hit_rate * 100, 1),
        "top_spans": uniq,
    }


def decision(score: float, warn: int, block: int) -> str:
    if score >= block:
        return "Từ chối (chặn)"
    if score >= warn:
        return "Cảnh báo — xem lại"
    return "Chấp nhận"


def rank_ok(scores: list[float]) -> tuple[bool, str]:
    # 00≈01 > 02 > 03 > 04 > 05 >> 06
    labels = ["00", "01", "02", "03", "04", "05", "06"]
    gaps = []
    for i in range(1, 6):
        if scores[i] > scores[i - 1] + 1.5:
            gaps.append(f"{labels[i]} ({scores[i]}) > {labels[i-1]} ({scores[i-1]}) — lệch thứ tự")
    if abs(scores[0] - scores[1]) > 8:
        gaps.append(f"00 ({scores[0]}) và 01 ({scores[1]}) lệch nhau nhiều")
    if scores[6] > 25:
        gaps.append(f"06 quá cao ({scores[6]})")
    if scores[5] - scores[6] < 8 and scores[5] > 0:
        gaps.append(f"05 không tách rõ 06 ({scores[5]} vs {scores[6]})")
    if scores[4] <= 0.1 and scores[5] <= 0.1:
        gaps.append("04 và 05 cùng ~0 — yếu cốt truyện")
    return (not gaps, "; ".join(gaps) if gaps else "Thứ tự đúng")


def main() -> None:
    warn, block = 70, 90
    print("DATA", os.environ["TRUNGLAP_DATA"])
    svc = AppService()
    t0 = time.perf_counter()
    svc.warmup(lambda msg, frac: print(f"  [{frac:.0%}] {msg}"))
    print(f"model {svc.device_label()} load={time.perf_counter()-t0:.1f}s")

    origin = extract_file(TEST_DIR / "00_TRUYEN_GOC.txt")
    nb = svc.create_notebook("Bo test truyen dai")
    t_add = time.perf_counter()
    svc.add_text(nb, "00_TRUYEN_GOC", origin, str(TEST_DIR / "00_TRUYEN_GOC.txt"))
    add_s = time.perf_counter() - t_add
    print(f"nap 00: {len(origin)} ky tu, {add_s:.2f}s, chunks={svc.store.n}")

    forward: dict[str, dict] = {}
    for fname, label in FILES:
        text = extract_file(TEST_DIR / fname)
        print(f"\n=== FORWARD {label} ===")
        rec = diagnose(svc, nb, text, fname, warn, block)
        rec["file"] = fname
        rec["label"] = label
        rec["decision"] = decision(rec["raw_score"], warn, block)
        forward[fname] = rec
        print(
            f"  official={rec['official_score']} {rec['official_verdict']} "
            f"raw={rec['raw_score']} jac={rec['jaccard_pct']} "
            f"max={rec['max_cos']} cover={rec['coverage_pct']}% {rec['elapsed_s']}s"
        )
        for i, sp in enumerate(rec["top_spans"][:3], 1):
            print(f"  span{i} cos={sp['raw_cos']} cal={sp['cal']}%")
            print(f"    Q: {sp['query']}")
            print(f"    S: {sp['source']}")

    # reverse order, warehouse unchanged
    reverse: dict[str, dict] = {}
    for fname, label in reversed(FILES):
        if fname.startswith("00"):
            continue
        text = extract_file(TEST_DIR / fname)
        print(f"\n=== REVERSE {label} ===")
        rec = diagnose(svc, nb, text, fname, warn, block)
        rec["file"] = fname
        rec["label"] = label
        rec["decision"] = decision(rec["raw_score"], warn, block)
        reverse[fname] = rec
        print(
            f"  raw={rec['raw_score']} official={rec['official_score']} "
            f"delta_vs_fwd={rec['raw_score']-forward[fname]['raw_score']:.1f} {rec['elapsed_s']}s"
        )

    order = [forward[f]["raw_score"] for f, _ in FILES]
    ok, why = rank_ok(order)
    out = {
        "warn": warn,
        "block": block,
        "device": svc.device_label(),
        "add_origin_s": round(add_s, 3),
        "origin_chars": len(origin),
        "index_chunks": svc.store.n,
        "rank_ok": ok,
        "rank_note": why,
        "order_scores": order,
        "forward": forward,
        "reverse_deltas": {
            f: round(reverse[f]["raw_score"] - forward[f]["raw_score"], 2)
            for f, _ in FILES
            if f in reverse
        },
        "reverse": reverse,
    }
    dest = ROOT / "data" / "eval_bo_test.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRANK", ok, why)
    print("WROTE", dest)
    svc.close()


if __name__ == "__main__":
    main()
