"""SQLite access layer.

The database stores three logical groups of data:

* Normative base – the catalogue of rascenki (codes + names).
* Knowledge base components:
    - association rules (FP-Growth output)
    - statistical profiles (frequency, contextual frequency, …)
    - transactions (parsed PTMs from historical XML estimates)
* User feedback used by the ranking layer as implicit training data.

All access goes through small, plain-SQL helper functions.  No ORM is
needed for the project scope.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

from common.config import DB_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
-- Catalogue of normative positions.
CREATE TABLE IF NOT EXISTS rascenka (
    obosn         TEXT PRIMARY KEY,
    naim          TEXT NOT NULL,
    tip           TEXT,
    ed_izm        TEXT,
    entity_type   TEXT DEFAULT 'rate',
    rate_group_id INTEGER,
    source_id     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rascenka_naim  ON rascenka(naim);
CREATE INDEX IF NOT EXISTS idx_rascenka_type  ON rascenka(entity_type);
CREATE INDEX IF NOT EXISTS idx_rascenka_group ON rascenka(rate_group_id);

-- Hierarchy of rate groups (chapters, sections, …).
CREATE TABLE IF NOT EXISTS rate_groups (
    id          INTEGER PRIMARY KEY,
    code        TEXT,
    name        TEXT NOT NULL,
    name_prefix TEXT,
    parent_id   INTEGER,
    FOREIGN KEY (parent_id) REFERENCES rate_groups(id)
);
CREATE INDEX IF NOT EXISTS idx_rg_parent ON rate_groups(parent_id);

-- Tracks which XML files have already been ingested.
-- Deduplication is file-level (by SHA-256 of content), NOT content-level.
-- This means: the same file is never imported twice, but 100 different
-- files that all contain a "Тёплый пол" PTM produce 100 transactions —
-- which is correct for FP-Growth support calculations.
CREATE TABLE IF NOT EXISTS ingested_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash   TEXT NOT NULL UNIQUE,
    file_path   TEXT NOT NULL,
    ptm_count   INTEGER DEFAULT 0,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One row per PTM from a historical XML estimate.
-- No UNIQUE constraint on content — the same composition in different
-- projects is intentionally counted multiple times.
CREATE TABLE IF NOT EXISTS transactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ptm_kod    TEXT,
    ptm_naim   TEXT,
    glava      TEXT,
    items      TEXT NOT NULL,   -- JSON array of obosn codes
    file_hash  TEXT             -- FK to ingested_files.file_hash
);
CREATE INDEX IF NOT EXISTS idx_tx_kod      ON transactions(ptm_kod);
CREATE INDEX IF NOT EXISTS idx_tx_glava    ON transactions(glava);
CREATE INDEX IF NOT EXISTS idx_tx_filehash ON transactions(file_hash);

-- FP-Growth association rules
CREATE TABLE IF NOT EXISTS association_rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    antecedent  TEXT NOT NULL,       -- JSON list of obosn
    consequent  TEXT NOT NULL,       -- JSON list of obosn
    support     REAL NOT NULL,
    confidence  REAL NOT NULL,
    lift        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rules_conf ON association_rules(confidence DESC);

-- Statistical profile for each rascenka
CREATE TABLE IF NOT EXISTS rascenka_profile (
    obosn      TEXT PRIMARY KEY,
    f_abs      INTEGER DEFAULT 0,
    f_norm     REAL    DEFAULT 0.0,
    ctx_json   TEXT    DEFAULT '{}', -- {ptm_kod: count}
    glava_json TEXT    DEFAULT '{}'  -- {glava: count}
);

-- User feedback (implicit signal for LTR)
CREATE TABLE IF NOT EXISTS user_feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    query     TEXT NOT NULL,
    obosn     TEXT NOT NULL,
    label     INTEGER NOT NULL,   -- 0/1/2/3 relevance
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated kit templates: a named PTM the user can recall by concept
CREATE TABLE IF NOT EXISTS kit_templates (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    members   TEXT NOT NULL  -- JSON list of obosn
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db():
    """Context manager that yields a SQLite connection."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path: Optional[Path] = None) -> None:
    """Create all tables (idempotent)."""
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rascenka catalogue
# ---------------------------------------------------------------------------

# Columns selected by every reader function – keep stable to simplify call sites.
_RASCENKA_COLS = "obosn, naim, tip, ed_izm, entity_type, rate_group_id, source_id"


def upsert_rascenki(rows: Iterable[Dict[str, Any]]) -> None:
    """rows: iterable of dicts with at least obosn and naim.

    Supported keys:
        obosn, naim, tip, ed_izm, entity_type, rate_group_id, source_id

    Missing keys take sensible defaults (entity_type='rate').
    """
    def _row(r: Dict[str, Any]) -> Tuple:
        return (
            r["obosn"],
            r.get("naim", ""),
            r.get("tip"),
            r.get("ed_izm"),
            r.get("entity_type", "rate"),
            r.get("rate_group_id"),
            r.get("source_id"),
        )

    with db() as conn:
        conn.executemany(
            """INSERT INTO rascenka (obosn, naim, tip, ed_izm, entity_type, rate_group_id, source_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(obosn) DO UPDATE SET
                   naim=excluded.naim,
                   tip=COALESCE(excluded.tip, rascenka.tip),
                   ed_izm=COALESCE(excluded.ed_izm, rascenka.ed_izm),
                   entity_type=excluded.entity_type,
                   rate_group_id=COALESCE(excluded.rate_group_id, rascenka.rate_group_id),
                   source_id=COALESCE(excluded.source_id, rascenka.source_id)""",
            [_row(r) for r in rows],
        )


def get_rascenka(obosn: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute(
            f"SELECT {_RASCENKA_COLS} FROM rascenka WHERE obosn = ?",
            (obosn,),
        ).fetchone()
        return dict(row) if row else None


def get_rascenki(obosn_list: List[str]) -> List[Dict[str, Any]]:
    if not obosn_list:
        return []
    placeholders = ",".join("?" for _ in obosn_list)
    with db() as conn:
        rows = conn.execute(
            f"SELECT {_RASCENKA_COLS} FROM rascenka WHERE obosn IN ({placeholders})",
            obosn_list,
        ).fetchall()
        return [dict(r) for r in rows]


def all_rascenki(entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all rascenki, optionally filtered by entity_type."""
    sql = f"SELECT {_RASCENKA_COLS} FROM rascenka"
    params: tuple = ()
    if entity_type:
        sql += " WHERE entity_type = ?"
        params = (entity_type,)
    sql += " ORDER BY obosn"
    with db() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def rascenki_count_by_type() -> Dict[str, int]:
    with db() as conn:
        rows = conn.execute(
            "SELECT entity_type, COUNT(*) AS c FROM rascenka GROUP BY entity_type"
        ).fetchall()
    return {r["entity_type"]: r["c"] for r in rows}


# ---------------------------------------------------------------------------
# Rate groups
# ---------------------------------------------------------------------------

def upsert_rate_groups(rows: Iterable[Dict[str, Any]]) -> None:
    """rows: dicts with keys id, code, name, name_prefix, parent_id."""
    def _row(r):
        return (
            r["id"],
            r.get("code"),
            r.get("name", ""),
            r.get("name_prefix"),
            r.get("parent_id"),
        )
    with db() as conn:
        conn.executemany(
            """INSERT INTO rate_groups (id, code, name, name_prefix, parent_id)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   code=excluded.code,
                   name=excluded.name,
                   name_prefix=excluded.name_prefix,
                   parent_id=excluded.parent_id""",
            [_row(r) for r in rows],
        )


def get_rate_group(group_id: int) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute(
            "SELECT id, code, name, name_prefix, parent_id FROM rate_groups WHERE id = ?",
            (group_id,),
        ).fetchone()
    return dict(row) if row else None


def get_rate_groups(group_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not group_ids:
        return {}
    placeholders = ",".join("?" for _ in group_ids)
    with db() as conn:
        rows = conn.execute(
            f"SELECT id, code, name, name_prefix, parent_id "
            f"FROM rate_groups WHERE id IN ({placeholders})",
            group_ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def all_rate_groups() -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, code, name, name_prefix, parent_id FROM rate_groups"
        ).fetchall()
    return [dict(r) for r in rows]


def rate_group_path(group_id: int, max_depth: int = 10) -> List[Dict[str, Any]]:
    """Walk up the parent chain. Returns root-first list of groups."""
    chain: List[Dict[str, Any]] = []
    current = group_id
    for _ in range(max_depth):
        g = get_rate_group(current)
        if not g:
            break
        chain.append(g)
        if not g["parent_id"]:
            break
        current = g["parent_id"]
    return list(reversed(chain))


# ---------------------------------------------------------------------------
# Ingested files registry
# ---------------------------------------------------------------------------

def is_file_ingested(file_hash: str) -> bool:
    with db() as conn:
        row = conn.execute(
            "SELECT 1 FROM ingested_files WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    return row is not None


def mark_file_ingested(file_hash: str, file_path: str, ptm_count: int) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO ingested_files (file_hash, file_path, ptm_count)
               VALUES (?, ?, ?)
               ON CONFLICT(file_hash) DO UPDATE SET
                   ptm_count=excluded.ptm_count""",
            (file_hash, str(file_path), ptm_count),
        )


def all_ingested_files() -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT file_hash, file_path, ptm_count, ingested_at "
            "FROM ingested_files ORDER BY ingested_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def insert_transaction(
    ptm_kod: str,
    ptm_naim: str,
    glava: Optional[str],
    items: List[str],
    file_hash: Optional[str] = None,
) -> int:
    """Insert a transaction. Returns the new row id.

    No content-level deduplication — the same set of codes from two
    different files counts as two transactions, which is correct for
    FP-Growth support calculations (each project is an independent
    observation). File-level deduplication is handled upstream via
    :func:`is_file_ingested` / :func:`mark_file_ingested`.
    """
    items_json = json.dumps(sorted(set(items)), ensure_ascii=False)
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO transactions (ptm_kod, ptm_naim, glava, items, file_hash)
               VALUES (?, ?, ?, ?, ?)""",
            (ptm_kod, ptm_naim, glava, items_json, file_hash),
        )
        return cur.lastrowid


def all_transactions() -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, ptm_kod, ptm_naim, glava, items, file_hash "
            "FROM transactions"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["items"] = json.loads(d["items"])
        out.append(d)
    return out


def transactions_count() -> int:
    with db() as conn:
        return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------

def replace_rules(rules: List[Dict[str, Any]]) -> None:
    """Atomic replacement of the association rules table."""
    with db() as conn:
        conn.execute("DELETE FROM association_rules")
        conn.executemany(
            """INSERT INTO association_rules
                (antecedent, consequent, support, confidence, lift)
                VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    json.dumps(r["antecedent"], ensure_ascii=False),
                    json.dumps(r["consequent"], ensure_ascii=False),
                    r["support"], r["confidence"], r["lift"],
                )
                for r in rules
            ],
        )


def find_rules_for(obosn: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Find association rules whose antecedent contains obosn."""
    with db() as conn:
        rows = conn.execute(
            """SELECT antecedent, consequent, support, confidence, lift
               FROM association_rules
               WHERE antecedent LIKE ?
               ORDER BY confidence DESC, lift DESC
               LIMIT ?""",
            (f'%"{obosn}"%', limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["antecedent"] = json.loads(d["antecedent"])
        d["consequent"] = json.loads(d["consequent"])
        if obosn in d["antecedent"]:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Statistical profiles
# ---------------------------------------------------------------------------

def replace_profiles(profiles: List[Dict[str, Any]]) -> None:
    with db() as conn:
        conn.execute("DELETE FROM rascenka_profile")
        conn.executemany(
            """INSERT INTO rascenka_profile
                (obosn, f_abs, f_norm, ctx_json, glava_json)
                VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    p["obosn"], p["f_abs"], p["f_norm"],
                    json.dumps(p.get("ctx", {}), ensure_ascii=False),
                    json.dumps(p.get("glava", {}), ensure_ascii=False),
                )
                for p in profiles
            ],
        )


def get_profile(obosn: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute(
            "SELECT obosn, f_abs, f_norm, ctx_json, glava_json "
            "FROM rascenka_profile WHERE obosn = ?",
            (obosn,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["ctx"] = json.loads(d.pop("ctx_json"))
    d["glava"] = json.loads(d.pop("glava_json"))
    return d


def get_profiles(obosn_list: List[str]) -> Dict[str, Dict[str, Any]]:
    if not obosn_list:
        return {}
    placeholders = ",".join("?" for _ in obosn_list)
    with db() as conn:
        rows = conn.execute(
            f"SELECT obosn, f_abs, f_norm, ctx_json, glava_json "
            f"FROM rascenka_profile WHERE obosn IN ({placeholders})",
            obosn_list,
        ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = dict(r)
        d["ctx"] = json.loads(d.pop("ctx_json"))
        d["glava"] = json.loads(d.pop("glava_json"))
        out[d["obosn"]] = d
    return out


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def insert_feedback(query: str, obosn: str, label: int) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO user_feedback (query, obosn, label) VALUES (?, ?, ?)",
            (query, obosn, int(label)),
        )


def all_feedback() -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT query, obosn, label FROM user_feedback"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Kit templates
# ---------------------------------------------------------------------------

def upsert_kit_template(name: str, members: List[str]) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO kit_templates (name, members) VALUES (?, ?)",
            (name, json.dumps(members, ensure_ascii=False)),
        )


def all_kit_templates() -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, name, members FROM kit_templates"
        ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "members": json.loads(r["members"])}
        for r in rows
    ]
