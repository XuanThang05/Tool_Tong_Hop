from __future__ import annotations

import re
import unicodedata

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_SENT_RE = re.compile(r"(?<=[\.!?…:;])\s+|(?<=\n)")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fingerprint_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]
    if not sentences:
        sentences = [text]

    packed: list[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            buf = sent
            continue
        if len(buf) + 1 + len(sent) <= size:
            buf = f"{buf} {sent}" if not buf.endswith("\n") else f"{buf}{sent}"
        else:
            packed.append(buf)
            buf = sent
    if buf:
        packed.append(buf)

    pieces: list[str] = []
    for block in packed:
        if len(block) <= size * 1.4:
            pieces.append(block)
        else:
            start = 0
            while start < len(block):
                end = min(len(block), start + size)
                pieces.append(block[start:end].strip())
                if end >= len(block):
                    break
                start = max(end - overlap, start + 1)

    if overlap <= 0 or len(pieces) == 1:
        return [p for p in pieces if p]

    merged: list[str] = []
    for i, piece in enumerate(pieces):
        merged.append(piece)
        if i + 1 < len(pieces) and overlap:
            tail = piece[-overlap:]
            head = pieces[i + 1][:overlap]
            bridge = f"{tail} {head}".strip()
            if len(bridge) >= overlap:
                merged.append(bridge)
    return [p for p in merged if p]
