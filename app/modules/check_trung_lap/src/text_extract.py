from __future__ import annotations

import re
import zipfile
from html import unescape
from pathlib import Path

from charset_normalizer import from_bytes

from src.config import SUPPORTED_EXT

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+\n")


class ExtractError(ValueError):
    pass


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXT and path.is_file()


def collect_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.name.startswith("~$"):
                    continue
                if is_supported(p):
                    key = str(p.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(p)
        elif is_supported(path):
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(path)
        else:
            raise ExtractError(f"Không hỗ trợ định dạng: {path.suffix or path.name}")
    return out


def extract_file(path: Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".txt", ".text", ".md", ".markdown"}:
        text = _read_text_file(path)
    elif ext == ".docx":
        text = _read_docx(path)
    elif ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".rtf":
        text = _read_rtf(path)
    elif ext == ".odt":
        text = _read_odt(path)
    elif ext in {".html", ".htm"}:
        text = _read_html(path.read_bytes())
    elif ext == ".epub":
        text = _read_epub(path)
    else:
        raise ExtractError(f"Không hỗ trợ định dạng: {ext}")
    text = cleanup_text(text)
    if not text.strip():
        raise ExtractError(f"File không có nội dung chữ: {path.name}")
    return text


def cleanup_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    return _decode_bytes(data)


def _decode_bytes(data: bytes) -> str:
    if not data:
        return ""
    best = from_bytes(data).best()
    if best is None:
        return data.decode("utf-8", errors="replace")
    return str(best)


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        parts.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_rtf(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text

    raw = _decode_bytes(path.read_bytes())
    return rtf_to_text(raw)


def _read_odt(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("content.xml").decode("utf-8")
    xml = re.sub(r"<text:line-break[^/]*/>", "\n", xml)
    xml = re.sub(r"<text:tab[^/]*/>", "\t", xml)
    xml = re.sub(r"<text:p\b[^>]*>", "\n", xml)
    xml = re.sub(r"<text:h\b[^>]*>", "\n", xml)
    xml = _TAG_RE.sub("", xml)
    return unescape(xml)


def _read_html(data: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode_bytes(data), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def _read_epub(path: Path) -> str:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        parts.append(_read_html(item.get_content()))
    return "\n\n".join(parts)
