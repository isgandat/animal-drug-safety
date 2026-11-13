"""SQLite storage with idempotent, partition-level loads."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from etl.parse import Parsed

SCHEMA = """
CREATE TABLE IF NOT EXISTS partitions (
    key TEXT PRIMARY KEY,
    file_url TEXT,
    sha256 TEXT NOT NULL,
    records_in INTEGER NOT NULL,
    records_loaded INTEGER NOT NULL,
    records_rejected INTEGER NOT NULL,
    loaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    aer_id TEXT PRIMARY KEY,
    report_id TEXT,
    partition_key TEXT NOT NULL,
    receive_date TEXT NOT NULL,
    receive_year INTEGER NOT NULL,
    species TEXT, gender TEXT, repro_status TEXT, crossbred INTEGER,
    age_years REAL, weight_kg REAL, health_prior TEXT,
    reporter TEXT, type_info TEXT, origin TEXT,
    serious INTEGER, treated INTEGER, n_affected INTEGER, n_treated INTEGER
);
CREATE TABLE IF NOT EXISTS drugs (
    aer_id TEXT NOT NULL REFERENCES reports(aer_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    route TEXT, administered_by TEXT, off_label TEXT,
    used_per_label INTEGER, previous_exposure INTEGER,
    dosage_form TEXT, atc_code TEXT,
    PRIMARY KEY (aer_id, seq)
);
CREATE TABLE IF NOT EXISTS ingredients (
    aer_id TEXT NOT NULL REFERENCES reports(aer_id) ON DELETE CASCADE,
    drug_seq INTEGER NOT NULL,
    name_raw TEXT,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reactions (
    aer_id TEXT NOT NULL REFERENCES reports(aer_id) ON DELETE CASCADE,
    term_code TEXT,
    term TEXT NOT NULL,
    veddra_version TEXT,
    PRIMARY KEY (aer_id, term)
);
CREATE TABLE IF NOT EXISTS outcomes (
    aer_id TEXT NOT NULL REFERENCES reports(aer_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    n_animals INTEGER
);
CREATE INDEX IF NOT EXISTS ix_reports_year ON reports(receive_year);
CREATE INDEX IF NOT EXISTS ix_ingredients_aer ON ingredients(aer_id);
CREATE INDEX IF NOT EXISTS ix_ingredients_name ON ingredients(name);
CREATE INDEX IF NOT EXISTS ix_reactions_term ON reactions(term);
CREATE INDEX IF NOT EXISTS ix_outcomes_aer ON outcomes(aer_id);
"""

_COLUMNS = {
    "reports": ["aer_id", "report_id", "partition_key", "receive_date", "receive_year",
                "species", "gender", "repro_status", "crossbred", "age_years", "weight_kg",
                "health_prior", "reporter", "type_info", "origin", "serious", "treated",
                "n_affected", "n_treated"],
    "drugs": ["aer_id", "seq", "route", "administered_by", "off_label", "used_per_label",
              "previous_exposure", "dosage_form", "atc_code"],
    "ingredients": ["aer_id", "drug_seq", "name_raw", "name"],
    "reactions": ["aer_id", "term_code", "term", "veddra_version"],
    "outcomes": ["aer_id", "status", "n_animals"],
}


def _insert_sql(table: str) -> str:
    cols = _COLUMNS[table]
    return (f"INSERT INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join(':' + c for c in cols)})")


def connect(path: Path, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def loaded_sha(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT sha256 FROM partitions WHERE key = ?", (key,)).fetchone()
    return row["sha256"] if row else None


def load_partition(conn: sqlite3.Connection, key: str, records: List[Parsed],
                   sha: str, records_in: int, rejected: int, file_url: str = "") -> Dict[str, int]:
    """Upsert one partition in a single transaction.

    A report that is already stored (same aer_id, from this or another
    partition) is deleted first; ON DELETE CASCADE removes its child rows, so a
    re-load never leaves duplicates or stale reactions behind.
    """
    latest: Dict[str, Parsed] = {}
    for rec in records:  # last version wins if an id repeats inside the file
        latest[rec.report["aer_id"]] = rec
    batch = list(latest.values())
    with conn:
        before = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        conn.executemany("DELETE FROM reports WHERE aer_id = ?",
                         [(r.report["aer_id"],) for r in batch])
        replaced = before - conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        conn.executemany(_insert_sql("reports"), [r.report for r in batch])
        for table in ("drugs", "ingredients", "reactions", "outcomes"):
            rows = [row for r in batch for row in getattr(r, table)]
            if rows:
                conn.executemany(_insert_sql(table), rows)
        conn.execute(
            "INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, file_url, sha, records_in, len(batch), rejected,
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return {"loaded": len(batch), "replaced": replaced, "rejected": rejected}
