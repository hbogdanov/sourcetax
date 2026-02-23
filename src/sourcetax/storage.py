import sqlite3
from pathlib import Path
import json
from typing import Dict, Any

DB_PATH = Path("data") / "store.db"


def ensure_db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    # Phase 2 schema: all canonical fields needed for receipt extraction, matching, categorization
    cur.execute("""
    CREATE TABLE IF NOT EXISTS canonical_records (
        rowid INTEGER PRIMARY KEY,
        id TEXT,
        source TEXT,
        source_record_id TEXT,
        transaction_date TEXT,
        merchant_raw TEXT,
        merchant_norm TEXT,
        amount REAL,
        currency TEXT,
        direction TEXT,
        payment_method TEXT,
        category_pred TEXT,
        category_final TEXT,
        confidence REAL,
        matched_transaction_id TEXT,
        match_score REAL,
        evidence_keys TEXT,
        raw_payload TEXT,
        tags TEXT
    )
    """)
    conn.commit()
    conn.close()


def insert_record(rec: Dict[str, Any], path: Path = DB_PATH):
    ensure_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO canonical_records 
           (id, source, source_record_id, transaction_date, merchant_raw, merchant_norm, 
            amount, currency, direction, payment_method, category_pred, category_final, 
            confidence, matched_transaction_id, match_score, evidence_keys, raw_payload, tags) 
           VALUES (WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:,WARNING:)""",
        (
            rec.get("id"),
            rec.get("source"),
            rec.get("source_record_id"),
            rec.get("transaction_date"),
            rec.get("merchant_raw"),
            rec.get("merchant_norm"),
            rec.get("amount"),
            rec.get("currency"),
            rec.get("direction"),
            rec.get("payment_method"),
            rec.get("category_pred"),
            rec.get("category_final"),
            rec.get("confidence"),
            rec.get("matched_transaction_id"),
            rec.get("match_score"),
            json.dumps(rec.get("evidence_keys") or []),
            json.dumps(rec.get("raw_payload") or {}),
            json.dumps(rec.get("tags") or []),
        ),
    )
    conn.commit()
    conn.close()
