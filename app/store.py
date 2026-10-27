"""SQLite persistence for batches and per-file results (survives page refreshes and restarts)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY, created REAL, label TEXT, status TEXT, report TEXT, config TEXT
);
CREATE TABLE IF NOT EXISTS files (
  batch_id TEXT, name TEXT, status TEXT, error TEXT, prediction TEXT, meta TEXT, label TEXT,
  started REAL, finished REAL, PRIMARY KEY (batch_id, name)
);
"""


def init(db_path: Path) -> None:
    global _conn
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.executescript(SCHEMA)
    _conn.commit()


def _q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    assert _conn is not None
    with _lock:
        cur = _conn.execute(sql, args)
        rows = cur.fetchall()
        _conn.commit()
        return rows


def create_batch(batch_id: str, label: str, report: dict, files: list[tuple[str, dict | None]]) -> None:
    _q("INSERT INTO batches VALUES (?,?,?,?,?,?)",
       (batch_id, time.time(), label, "validated", json.dumps(report), "{}"))
    for name, lab in files:
        _q("INSERT INTO files (batch_id, name, status, label) VALUES (?,?,?,?)",
           (batch_id, name, "pending", json.dumps(lab) if lab else None))


def set_batch(batch_id: str, **kw) -> None:
    for k, v in kw.items():
        if k in ("report", "config"):
            v = json.dumps(v)
        _q(f"UPDATE batches SET {k}=? WHERE id=?", (v, batch_id))  # keys are internal constants


def set_file(batch_id: str, name: str, **kw) -> None:
    for k, v in kw.items():
        if k in ("prediction", "meta") and v is not None:
            v = json.dumps(v)
        _q(f"UPDATE files SET {k}=? WHERE batch_id=? AND name=?", (v, batch_id, name))


def get_batch(batch_id: str) -> dict | None:
    rows = _q("SELECT * FROM batches WHERE id=?", (batch_id,))
    if not rows:
        return None
    b = dict(rows[0])
    b["report"] = json.loads(b["report"] or "{}")
    b["config"] = json.loads(b["config"] or "{}")
    return b


def list_batches(limit: int = 50) -> list[dict]:
    rows = _q("""SELECT b.id, b.created, b.label, b.status,
                 SUM(f.status='done') AS done, SUM(f.status='error') AS failed, COUNT(f.name) AS total
                 FROM batches b LEFT JOIN files f ON f.batch_id=b.id
                 GROUP BY b.id ORDER BY b.created DESC LIMIT ?""", (limit,))
    return [dict(r) for r in rows]


def get_files(batch_id: str) -> list[dict]:
    rows = _q("SELECT * FROM files WHERE batch_id=? ORDER BY name", (batch_id,))
    out = []
    for r in rows:
        d = dict(r)
        for k in ("prediction", "meta", "label"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def delete_batch(batch_id: str) -> bool:
    """Remove a batch and its per-file rows. Returns False if it did not exist."""
    if not _q("SELECT id FROM batches WHERE id=?", (batch_id,)):
        return False
    _q("DELETE FROM files WHERE batch_id=?", (batch_id,))
    _q("DELETE FROM batches WHERE id=?", (batch_id,))
    return True


def interrupted() -> list[tuple[str, str]]:
    rows = _q("SELECT batch_id, name FROM files WHERE status IN ('queued','processing')")
    return [(r["batch_id"], r["name"]) for r in rows]
