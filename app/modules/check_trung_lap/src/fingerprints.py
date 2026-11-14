from __future__ import annotations

import hashlib
import struct

import numpy as np

from src.chunking import fingerprint_text
from src.config import MINHASH_NGRAM, MINHASH_PERM

_PRIME = np.uint64(2305843009213693951)
_RNG = np.random.default_rng(20260916)
_A = _RNG.integers(1, int(_PRIME), size=MINHASH_PERM, dtype=np.uint64)
_B = _RNG.integers(0, int(_PRIME), size=MINHASH_PERM, dtype=np.uint64)


def content_hash(text: str) -> str:
    body = fingerprint_text(text).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def minhash_signature(text: str, ngram: int = MINHASH_NGRAM) -> bytes:
    src = fingerprint_text(text)
    if not src:
        sig = np.zeros(MINHASH_PERM, dtype=np.uint64)
        return sig.tobytes()
    if len(src) < ngram:
        grams = [src]
    else:
        grams = [src[i : i + ngram] for i in range(len(src) - ngram + 1)]

    sig = np.full(MINHASH_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    seen: set[str] = set()
    for g in grams:
        if g in seen:
            continue
        seen.add(g)
        h = np.uint64(_blake_u64(g))
        vals = (_A * h + _B) % _PRIME
        np.minimum(sig, vals, out=sig)
    return sig.tobytes()


def minhash_jaccard(a: bytes, b: bytes) -> float:
    sa = np.frombuffer(a, dtype=np.uint64)
    sb = np.frombuffer(b, dtype=np.uint64)
    if sa.size == 0 or sa.size != sb.size:
        return 0.0
    return float(np.mean(sa == sb))


def _blake_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]
