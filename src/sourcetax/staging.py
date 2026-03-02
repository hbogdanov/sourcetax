"""Staging-layer utilities for external dataset ingestion.

Staging tables are intentionally source-agnostic and preserve raw fields so we can:
1) map external taxonomies to SourceTax categories deterministically, and
2) sample realistic records for gold labeling.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_STAGING_DB = Path("data") / "staging.db"


def _json_dumps(value: Any, default: Any) -> str:
    return json.dumps(value if value is not None else default, ensure_ascii=False)


def ensure_staging_db(path: Path = DEFAULT_STAGING_DB) -> None:
    """Create staging tables used by the data-focused workflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS staging_transactions (
            rowid INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            source_record_id TEXT,
            txn_ts TEXT,
            amount REAL,
            currency TEXT,
            merchant_raw TEXT,
            description TEXT,
            mcc TEXT,
            mcc_description TEXT,
            category_external TEXT,
            subcategory_external TEXT,
            raw_payload_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_staging_txn_source
        ON staging_transactions(source)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_staging_txn_mcc
        ON staging_transactions(mcc)
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS staging_receipts (
            rowid INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            source_record_id TEXT,
            receipt_ts TEXT,
            merchant_raw TEXT,
            total REAL,
            tax REAL,
            currency TEXT,
            ocr_text TEXT,
            structured_fields_json TEXT NOT NULL DEFAULT '{}',
            raw_payload_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_staging_receipt_source
        ON staging_receipts(source)
        """
    )

    conn.commit()
    conn.close()


def insert_staging_transaction(
    record: Dict[str, Any], path: Path = DEFAULT_STAGING_DB
) -> None:
    """Insert a bank-like staging transaction row."""
    ensure_staging_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO staging_transactions (
            source, source_record_id, txn_ts, amount, currency, merchant_raw,
            description, mcc, mcc_description, category_external, subcategory_external,
            raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("source"),
            record.get("source_record_id"),
            record.get("txn_ts"),
            record.get("amount"),
            record.get("currency"),
            record.get("merchant_raw"),
            record.get("description"),
            record.get("mcc"),
            record.get("mcc_description"),
            record.get("category_external"),
            record.get("subcategory_external"),
            _json_dumps(record.get("raw_payload_json"), {}),
        ),
    )
    conn.commit()
    conn.close()


def insert_staging_transactions(
    records: list[Dict[str, Any]],
    path: Path = DEFAULT_STAGING_DB,
    batch_size: int = 1000,
) -> int:
    """Insert many bank-like staging rows in batches for fast loaders."""
    if not records:
        return 0
    ensure_staging_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    sql = """
        INSERT INTO staging_transactions (
            source, source_record_id, txn_ts, amount, currency, merchant_raw,
            description, mcc, mcc_description, category_external, subcategory_external,
            raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    inserted = 0
    for i in range(0, len(records), max(batch_size, 1)):
        chunk = records[i : i + max(batch_size, 1)]
        params = [
            (
                rec.get("source"),
                rec.get("source_record_id"),
                rec.get("txn_ts"),
                rec.get("amount"),
                rec.get("currency"),
                rec.get("merchant_raw"),
                rec.get("description"),
                rec.get("mcc"),
                rec.get("mcc_description"),
                rec.get("category_external"),
                rec.get("subcategory_external"),
                _json_dumps(rec.get("raw_payload_json"), {}),
            )
            for rec in chunk
        ]
        cur.executemany(sql, params)
        inserted += len(params)
    conn.commit()
    conn.close()
    return inserted


def insert_staging_receipt(record: Dict[str, Any], path: Path = DEFAULT_STAGING_DB) -> None:
    """Insert a receipt-like staging document row."""
    ensure_staging_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO staging_receipts (
            source, source_record_id, receipt_ts, merchant_raw, total, tax, currency,
            ocr_text, structured_fields_json, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("source"),
            record.get("source_record_id"),
            record.get("receipt_ts"),
            record.get("merchant_raw"),
            record.get("total"),
            record.get("tax"),
            record.get("currency"),
            record.get("ocr_text"),
            _json_dumps(record.get("structured_fields_json"), {}),
            _json_dumps(record.get("raw_payload_json"), {}),
        ),
    )
    conn.commit()
    conn.close()


def insert_staging_receipts(
    records: list[Dict[str, Any]],
    path: Path = DEFAULT_STAGING_DB,
    batch_size: int = 500,
) -> int:
    """Insert many receipt-like staging rows in batches."""
    if not records:
        return 0
    ensure_staging_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    sql = """
        INSERT INTO staging_receipts (
            source, source_record_id, receipt_ts, merchant_raw, total, tax, currency,
            ocr_text, structured_fields_json, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    inserted = 0
    for i in range(0, len(records), max(batch_size, 1)):
        chunk = records[i : i + max(batch_size, 1)]
        params = [
            (
                rec.get("source"),
                rec.get("source_record_id"),
                rec.get("receipt_ts"),
                rec.get("merchant_raw"),
                rec.get("total"),
                rec.get("tax"),
                rec.get("currency"),
                rec.get("ocr_text"),
                _json_dumps(rec.get("structured_fields_json"), {}),
                _json_dumps(rec.get("raw_payload_json"), {}),
            )
            for rec in chunk
        ]
        cur.executemany(sql, params)
        inserted += len(params)
    conn.commit()
    conn.close()
    return inserted


def get_staging_counts(path: Path = DEFAULT_STAGING_DB) -> Dict[str, int]:
    """Return row counts for quick ingestion validation."""
    ensure_staging_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM staging_transactions")
    txns = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM staging_receipts")
    receipts = int(cur.fetchone()[0])
    conn.close()
    return {"staging_transactions": txns, "staging_receipts": receipts}
