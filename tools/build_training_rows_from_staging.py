#!/usr/bin/env python
"""Build mapped training rows from staging transactions with explainability.

Each emitted row includes:
- category_mapped
- mapping_reason (list[str])
- evidence_keys (same as mapping_reason for downstream compatibility)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from sourcetax import mapping
from sourcetax.normalization import normalize_merchant_name


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return default
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _row_to_training_row(row: sqlite3.Row) -> Dict[str, Any]:
    merchant_raw = row["merchant_raw"] or row["description"] or ""
    merchant_norm = normalize_merchant_name(merchant_raw, case="lower") if merchant_raw else ""
    category, reasons = mapping.resolve_category_with_reason(
        merchant_raw=merchant_raw,
        description=row["description"],
        mcc=row["mcc"],
        mcc_description=row["mcc_description"],
        external_category=row["category_external"],
        fallback="Other Expense",
    )
    raw_payload = _safe_json_loads(row["raw_payload_json"], {})
    if not isinstance(raw_payload, dict):
        raw_payload = {"raw_payload_value": raw_payload}
    raw_payload["mapping_reason"] = reasons

    return {
        "source": row["source"],
        "source_record_id": row["source_record_id"],
        "transaction_date": row["txn_ts"],
        "merchant_raw": merchant_raw or None,
        "merchant_norm": merchant_norm or None,
        "description": row["description"],
        "amount": row["amount"],
        "currency": row["currency"],
        "mcc": row["mcc"],
        "mcc_description": row["mcc_description"],
        "category_external": row["category_external"],
        "category_mapped": category,
        "mapping_reason": reasons,
        "evidence_keys": reasons,
        "raw_payload": raw_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--out", default="data/ml/staging_training_rows.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--where",
        default="1=1",
        help="SQL WHERE clause for staging_transactions filtering",
    )
    args = parser.parse_args()

    db_path = Path(args.staging_db)
    if not db_path.exists():
        print(f"Missing staging DB: {db_path}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = f"""
        SELECT source, source_record_id, txn_ts, amount, currency, merchant_raw, description,
               mcc, mcc_description, category_external, raw_payload_json
        FROM staging_transactions
        WHERE {args.where}
        ORDER BY rowid
    """
    if args.limit and args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            tr = _row_to_training_row(row)
            fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
            written += 1

    print("Training row build complete.")
    print(f"- rows_written: {written}")
    print(f"- out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
