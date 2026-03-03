#!/usr/bin/env python
"""Build enriched staging transactions with normalization + deterministic mapping.

Outputs:
- SQLite table (default: staging_transactions_enriched) in the staging DB
- CSV export (default: artifacts/exports/staging_transactions_enriched.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from sourcetax import mapping
from sourcetax.normalization import normalize_merchant_name


GENERIC_DESCRIPTION_TOKENS = (
    "POS PURCHASE",
    "DEBIT CARD",
    "ONLINE PAYMENT",
    "PURCHASE",
    "PAYMENT",
    "CHECKCARD",
)


def _mapping_confidence(reasons: List[str]) -> float:
    if not reasons:
        return 0.3
    r = reasons[0]
    if r.startswith("financial_high:"):
        return 0.9
    if r.startswith("financial_medium:"):
        return 0.7
    if "_high:" in r:
        return 0.9
    if "_medium:" in r:
        return 0.7
    if r.startswith("keyword:"):
        return 0.9
    if r.startswith("mcc:") or r.startswith("mcc_description:"):
        return 0.85
    if r.startswith("external:"):
        return 0.7
    return 0.3


def _is_generic_description(description: str) -> bool:
    d = (description or "").upper()
    if not d:
        return False
    return any(token in d for token in GENERIC_DESCRIPTION_TOKENS)


def _row_to_enriched(row: sqlite3.Row) -> Dict[str, Any]:
    merchant_raw = (row["merchant_raw"] or "").strip()
    description = (row["description"] or "").strip()
    merchant_for_norm = merchant_raw or description
    merchant_norm = normalize_merchant_name(merchant_for_norm, case="lower") if merchant_for_norm else ""

    category, reasons = mapping.resolve_category_with_reason(
        merchant_raw=merchant_raw or None,
        description=description or None,
        mcc=row["mcc"],
        mcc_description=row["mcc_description"],
        external_category=row["category_external"],
        amount=row["amount"],
        fallback="Other Expense",
    )
    confidence = _mapping_confidence(reasons)
    missing_merchant = 1 if not merchant_raw else 0
    missing_description = 1 if not description else 0
    missing_mcc = 1 if not (row["mcc"] or "").strip() else 0
    merchant_only = 1 if merchant_raw and not description else 0
    generic_description = 1 if _is_generic_description(description) else 0

    return {
        "rowid": row["rowid"],
        "source": row["source"],
        "source_record_id": row["source_record_id"],
        "txn_ts": row["txn_ts"],
        "amount": row["amount"],
        "currency": row["currency"],
        "merchant_raw": row["merchant_raw"],
        "merchant_norm": merchant_norm or None,
        "description": row["description"],
        "mcc": row["mcc"],
        "mcc_description": row["mcc_description"],
        "category_external": row["category_external"],
        "subcategory_external": row["subcategory_external"],
        "rule_category": category,
        "rule_confidence": confidence,
        "mapping_reason": json.dumps(reasons, ensure_ascii=False),
        "missing_merchant": missing_merchant,
        "missing_description": missing_description,
        "missing_mcc": missing_mcc,
        "merchant_only": merchant_only,
        "generic_description": generic_description,
        "raw_payload_json": row["raw_payload_json"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--table", default="staging_transactions_enriched")
    parser.add_argument("--out-csv", default="artifacts/exports/staging_transactions_enriched.csv")
    args = parser.parse_args()

    db_path = Path(args.staging_db)
    if not db_path.exists():
        print(f"Missing staging DB: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid, source, source_record_id, txn_ts, amount, currency, merchant_raw, description,
               mcc, mcc_description, category_external, subcategory_external, raw_payload_json
        FROM staging_transactions
        ORDER BY rowid
        """
    )
    rows = cur.fetchall()

    enriched = [_row_to_enriched(r) for r in rows]

    cur.execute(f"DROP TABLE IF EXISTS {args.table}")
    cur.execute(
        f"""
        CREATE TABLE {args.table} (
            rowid INTEGER PRIMARY KEY,
            source TEXT,
            source_record_id TEXT,
            txn_ts TEXT,
            amount REAL,
            currency TEXT,
            merchant_raw TEXT,
            merchant_norm TEXT,
            description TEXT,
            mcc TEXT,
            mcc_description TEXT,
            category_external TEXT,
            subcategory_external TEXT,
            rule_category TEXT,
            rule_confidence REAL,
            mapping_reason TEXT,
            missing_merchant INTEGER,
            missing_description INTEGER,
            missing_mcc INTEGER,
            merchant_only INTEGER,
            generic_description INTEGER,
            raw_payload_json TEXT
        )
        """
    )
    insert_sql = f"""
        INSERT INTO {args.table} (
            rowid, source, source_record_id, txn_ts, amount, currency, merchant_raw, merchant_norm, description,
            mcc, mcc_description, category_external, subcategory_external, rule_category, rule_confidence,
            mapping_reason, missing_merchant, missing_description, missing_mcc, merchant_only, generic_description,
            raw_payload_json
        ) VALUES (
            :rowid, :source, :source_record_id, :txn_ts, :amount, :currency, :merchant_raw, :merchant_norm, :description,
            :mcc, :mcc_description, :category_external, :subcategory_external, :rule_category, :rule_confidence,
            :mapping_reason, :missing_merchant, :missing_description, :missing_mcc, :merchant_only, :generic_description,
            :raw_payload_json
        )
    """
    cur.executemany(insert_sql, enriched)
    conn.commit()
    conn.close()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(enriched[0].keys()) if enriched else []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(enriched)

    print("Enriched staging build complete.")
    print(f"- rows_enriched: {len(enriched)}")
    print(f"- table: {args.table}")
    print(f"- csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

