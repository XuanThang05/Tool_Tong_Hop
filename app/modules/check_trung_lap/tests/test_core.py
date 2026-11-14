from __future__ import annotations

from pathlib import Path

from src.chunking import chunk_text, fingerprint_text
from src.checker import classify_kind, combine_score, verdict_for
from src.fingerprints import content_hash, minhash_jaccard, minhash_signature
from src.plot_fp import fingerprint, fingerprint_sim


def test_fingerprint_and_hash() -> None:
    a = "Cô gái bước vào quán.\n\nTrời mưa."
    b = "  cô gái   bước vào quán. trời mưa. "
    assert fingerprint_text(a).lower() in fingerprint_text(a)
    assert content_hash(a) == content_hash(b)


def test_chunk_long_text() -> None:
    text = "Câu một. " * 400
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(chunks)


def test_minhash_similar_not_unrelated() -> None:
    a = (
        "Cô gái bước vào quán cà phê cũ và ngồi xuống chiếc ghế gỗ gần cửa sổ. "
        "Cô gọi một ly đen đá rồi mở cuốn sổ tay đã sờn góc."
    )
    b = (
        "Cô gái đi vào tiệm cà phê cũ rồi ngồi ở ghế gỗ bên cửa sổ. "
        "Cô gọi ly đen đá và mở cuốn sổ tay sờn góc."
    )
    c = "Bão đổ bộ miền Trung, hàng nghìn hộ dân được sơ tán trong đêm."
    ja = minhash_jaccard(minhash_signature(a), minhash_signature(b))
    jc = minhash_jaccard(minhash_signature(a), minhash_signature(c))
    assert ja > jc
    assert ja > 0.2


def test_scoring_verdict() -> None:
    exact = combine_score(exact=True, jaccard=1, max_cos=1, mean_cos=1, coverage=1)
    assert exact == 100
    unrelated = combine_score(exact=False, jaccard=0.05, max_cos=0.56, mean_cos=0.55, coverage=0.0)
    rewrite = combine_score(exact=False, jaccard=0.05, max_cos=0.77, mean_cos=0.77, coverage=1.0)
    plotty = combine_score(
        exact=False, jaccard=0.02, max_cos=0.67, mean_cos=0.67, coverage=1.0, plot_sim=0.72
    )
    assert unrelated < 30
    assert rewrite >= 60
    assert plotty >= 68
    assert verdict_for(92, 70, 90) == "block"
    assert verdict_for(75, 70, 90) == "warn"
    assert verdict_for(20, 70, 90) == "ok"
    assert classify_kind(exact=True, score=100, jaccard=1, max_cos=1, coverage=1) == "Nguyên bản"
    assert (
        classify_kind(exact=False, score=72, jaccard=0.02, max_cos=0.67, coverage=1, plot_sim=0.72)
        == "Cùng cốt"
    )


def test_plot_fingerprint_same_theme() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    probes = rng.standard_normal((8, 16)).astype(np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    a = rng.standard_normal((3, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = a + rng.standard_normal((3, 16)).astype(np.float32) * 0.02
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = rng.standard_normal((3, 16)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    fa, fb, fc = fingerprint(a, probes), fingerprint(b, probes), fingerprint(c, probes)
    assert fingerprint_sim(fa, fb) > fingerprint_sim(fa, fc)


def test_extract_txt(tmp_path: Path) -> None:
    from src.text_extract import cleanup_text, extract_file

    p = tmp_path / "truyen.txt"
    p.write_text("Nội dung demo.\nDòng hai.", encoding="utf-8")
    assert "Nội dung demo" in extract_file(p)
    assert "Dòng hai" in cleanup_text("Dòng hai.\r\n\r\n\r\n")
