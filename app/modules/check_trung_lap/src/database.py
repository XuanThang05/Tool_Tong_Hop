from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import DB_PATH, DOCS_DIR, ensure_dirs


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path = DB_PATH) -> None:
        ensure_dirs()
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _init_schema(self) -> None:
        with self.tx() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS notebooks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notebook_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    content_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    minhash BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    notebook_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_docs_nb ON documents(notebook_id);
                CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_nb ON chunks(notebook_id);
                """
            )
            cols = {
                r["name"]
                for r in c.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "plot_fp" not in cols:
                c.execute("ALTER TABLE documents ADD COLUMN plot_fp BLOB")

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def list_notebooks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT n.id, n.name, n.created_at,
                       COUNT(d.id) AS doc_count,
                       COALESCE(SUM(d.char_count), 0) AS char_count
                FROM notebooks n
                LEFT JOIN documents d ON d.notebook_id = n.id
                GROUP BY n.id
                ORDER BY n.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def create_notebook(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Tên sổ không được trống.")
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO notebooks(name, created_at) VALUES(?, ?)",
                (name, _now()),
            )
            return int(cur.lastrowid)

    def rename_notebook(self, notebook_id: int, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Tên sổ không được trống.")
        with self.tx() as c:
            c.execute("UPDATE notebooks SET name = ? WHERE id = ?", (name, notebook_id))

    def delete_notebook(self, notebook_id: int) -> list[int]:
        with self._lock:
            docs = self.conn.execute(
                "SELECT id FROM documents WHERE notebook_id = ?", (notebook_id,)
            ).fetchall()
        doc_ids = [int(r["id"]) for r in docs]
        with self.tx() as c:
            c.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        for did in doc_ids:
            p = DOCS_DIR / f"{did}.txt"
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        return doc_ids

    def get_notebook(self, notebook_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, name, created_at FROM notebooks WHERE id = ?",
                (notebook_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self, notebook_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, notebook_id, title, source_path, content_hash,
                       char_count, created_at
                FROM documents
                WHERE notebook_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (notebook_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT d.id, d.notebook_id, d.title, d.source_path, d.content_hash,
                       d.char_count, d.minhash, d.created_at, n.name AS notebook_name
                FROM documents d
                JOIN notebooks n ON n.id = d.notebook_id
                WHERE d.id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_chunk_texts(self, chunk_ids: list[int]) -> dict[int, str]:
        ids = [int(i) for i in chunk_ids if int(i) >= 0]
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, text FROM chunks WHERE id IN ({q})", ids
            ).fetchall()
        return {int(r["id"]): str(r["text"]) for r in rows}

    def read_document_text(self, document_id: int) -> str:
        path = DOCS_DIR / f"{document_id}.txt"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def find_by_hash(
        self, content_hash: str, notebook_id: int | None = None
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT d.id, d.notebook_id, d.title, d.content_hash, d.char_count,
                   n.name AS notebook_name
            FROM documents d
            JOIN notebooks n ON n.id = d.notebook_id
            WHERE d.content_hash = ?
        """
        params: list[Any] = [content_hash]
        if notebook_id is not None:
            sql += " AND d.notebook_id = ?"
            params.append(notebook_id)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def iter_minhashes(
        self, notebook_id: int | None = None
    ) -> list[tuple[int, int, str, str, bytes]]:
        sql = """
            SELECT d.id, d.notebook_id, d.title, n.name AS notebook_name, d.minhash
            FROM documents d
            JOIN notebooks n ON n.id = d.notebook_id
        """
        params: list[Any] = []
        if notebook_id is not None:
            sql += " WHERE d.notebook_id = ?"
            params.append(notebook_id)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [
            (
                int(r["id"]),
                int(r["notebook_id"]),
                str(r["title"]),
                str(r["notebook_name"]),
                bytes(r["minhash"]),
            )
            for r in rows
        ]

    def iter_plot_fps(
        self, notebook_id: int | None = None
    ) -> list[tuple[int, bytes | None]]:
        sql = "SELECT id, plot_fp FROM documents"
        params: list[Any] = []
        if notebook_id is not None:
            sql += " WHERE notebook_id = ?"
            params.append(notebook_id)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [(int(r["id"]), bytes(r["plot_fp"]) if r["plot_fp"] else None) for r in rows]

    def update_plot_fp(self, document_id: int, plot_fp: bytes) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE documents SET plot_fp = ? WHERE id = ?",
                (plot_fp, document_id),
            )

    def insert_document(
        self,
        notebook_id: int,
        title: str,
        text: str,
        source_path: str | None,
        content_hash: str,
        minhash: bytes,
        chunks: list[str],
        plot_fp: bytes | None = None,
    ) -> tuple[int, list[int]]:
        ensure_dirs()
        with self.tx() as c:
            cur = c.execute(
                """
                INSERT INTO documents(
                    notebook_id, title, source_path, content_hash,
                    char_count, minhash, created_at, plot_fp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notebook_id,
                    title,
                    source_path,
                    content_hash,
                    len(text),
                    minhash,
                    _now(),
                    plot_fp,
                ),
            )
            doc_id = int(cur.lastrowid)
            chunk_ids: list[int] = []
            for i, ch in enumerate(chunks):
                cur = c.execute(
                    """
                    INSERT INTO chunks(document_id, notebook_id, chunk_index, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (doc_id, notebook_id, i, ch),
                )
                chunk_ids.append(int(cur.lastrowid))
            (DOCS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        return doc_id, chunk_ids

    def delete_document(self, document_id: int) -> dict[str, Any] | None:
        doc = self.get_document(document_id)
        if not doc:
            return None
        with self._lock:
            chunk_rows = self.conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
        chunk_ids = [int(r["id"]) for r in chunk_rows]
        with self.tx() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        path = DOCS_DIR / f"{document_id}.txt"
        if path.exists():
            path.unlink()
        doc["chunk_ids"] = chunk_ids
        return doc

    def stats(self) -> dict[str, int]:
        with self._lock:
            n_nb = self.conn.execute("SELECT COUNT(*) AS c FROM notebooks").fetchone()["c"]
            n_doc = self.conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
            n_chunk = self.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
            chars = self.conn.execute(
                "SELECT COALESCE(SUM(char_count), 0) AS c FROM documents"
            ).fetchone()["c"]
        return {
            "notebooks": int(n_nb),
            "documents": int(n_doc),
            "chunks": int(n_chunk),
            "chars": int(chars),
        }

    def list_chunks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, document_id, notebook_id, text
                FROM chunks
                ORDER BY id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def settings_json(self) -> dict[str, Any]:
        raw = self.get_setting("ui", "{}") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return data if isinstance(data, dict) else {}

    def save_settings_json(self, data: dict[str, Any]) -> None:
        self.set_setting("ui", json.dumps(data, ensure_ascii=False))
