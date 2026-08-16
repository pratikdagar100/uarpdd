"""SQLite persistence for the user study.

No account system and no PII: participants are identified by a random
session-scoped UUID only.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    condition TEXT NOT NULL CHECK (condition IN ('POINT','UNCERTAINTY')),
    prediction TEXT NOT NULL,
    rain_probability REAL NOT NULL,
    confidence TEXT,
    uncertainty_level TEXT,
    prediction_set TEXT,
    user_decision TEXT NOT NULL CHECK (user_decision IN ('EVENT','NO_EVENT')),
    user_confidence INTEGER NOT NULL CHECK (user_confidence BETWEEN 1 AND 5),
    decision_time_seconds REAL NOT NULL,
    actual_outcome TEXT NOT NULL,
    decision_correct INTEGER NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);
"""


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    return conn


def log_decision(record: dict, db_path: Path = DB_PATH) -> None:
    record = dict(record)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    record.setdefault("is_demo", 0)
    cols = ["participant_id", "session_id", "scenario_id", "condition",
            "prediction", "rain_probability", "confidence",
            "uncertainty_level", "prediction_set", "user_decision",
            "user_confidence", "decision_time_seconds", "actual_outcome",
            "decision_correct", "is_demo", "timestamp"]
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO decisions ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [record.get(c) for c in cols])


def fetch_decisions(include_demo: bool = False,
                    db_path: Path = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        q = "SELECT * FROM decisions"
        if not include_demo:
            q += " WHERE is_demo = 0"
        return pd.read_sql_query(q, conn)


def counts(db_path: Path = DB_PATH) -> dict:
    with _connect(db_path) as conn:
        real = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT participant_id) "
            "FROM decisions WHERE is_demo = 0").fetchone()
        demo = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE is_demo = 1").fetchone()
    return {"real_decisions": real[0], "real_participants": real[1],
            "demo_decisions": demo[0]}


def delete_demo_data(db_path: Path = DB_PATH) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM decisions WHERE is_demo = 1")
        return cur.rowcount


def delete_all_data(db_path: Path = DB_PATH) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM decisions")
        return cur.rowcount
