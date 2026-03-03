#!/usr/bin/env python
"""Backfill shadow-mode inference fields at scale.

Supports:
- canonical_records (data/interim/store.db-style)
- staging_transactions (data/interim/staging.db-style)

Writes rule/ml/hybrid shadow fields into payload JSON and keeps production
final category unchanged (rules) unless explicitly asked otherwise.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from sourcetax import categorization


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _deterministic_key(
    *,
    rowid: int,
    record_id: str,
    source: str,
    source_record_id: str,
    merchant_raw: str,
    amount: Any,
    description: str,
) -> str:
    rid = str(record_id or "").strip()
    if rid:
        return f"id:{rid}"
    src = str(source or "").strip()
    src_id = str(source_record_id or "").strip()
    if src and src_id:
        return f"source:{src}:{src_id}"
    fp = f"{merchant_raw or ''}\t{description or ''}\t{amount if amount is not None else ''}"
    if fp.strip():
        return f"fp:{fp.lower()}"
    return f"rowid:{rowid}"


def _backfill_canonical(conn: sqlite3.Connection, overwrite_category_pred: bool) -> Tuple[int, int]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid, id, source, source_record_id, merchant_raw, amount, raw_payload
        FROM canonical_records
        """
    )
    rows = cur.fetchall()
    seen = 0
    logged = 0
    for rowid, rid, source, source_record_id, merchant_raw, amount, raw_payload_raw in rows:
        seen += 1
        raw_payload = _safe_json_loads(raw_payload_raw, {})
        if not isinstance(raw_payload, dict):
            raw_payload = {}
        description = str(raw_payload.get("description") or raw_payload.get("ocr_text") or "").strip()
        mcc = str(raw_payload.get("mcc") or "").strip()
        mcc_description = str(raw_payload.get("mcc_description") or "").strip()
        category_external = str(raw_payload.get("category_external") or "").strip()

        shadow = categorization.build_shadow_decisions(
            merchant_raw=str(merchant_raw or "").strip(),
            amount=amount,
            description=description,
            mcc=mcc,
            mcc_description=mcc_description,
            category_external=category_external,
            raw_payload=raw_payload,
            threshold_primary=0.85,
            threshold_alt=0.70,
        )

        shadow_key = _deterministic_key(
            rowid=int(rowid),
            record_id=str(rid or ""),
            source=str(source or ""),
            source_record_id=str(source_record_id or ""),
            merchant_raw=str(merchant_raw or ""),
            amount=amount,
            description=description,
        )
        raw_payload.update(shadow)
        raw_payload["shadow_record_key"] = shadow_key
        raw_payload["shadow_table"] = "canonical_records"

        if overwrite_category_pred:
            cur.execute(
                """
                UPDATE canonical_records
                SET category_pred = ?, confidence = ?, raw_payload = ?
                WHERE rowid = ?
                """,
                (
                    shadow.get("final_category"),
                    float(shadow.get("rule_confidence") or 0.0),
                    json.dumps(raw_payload, ensure_ascii=False),
                    rowid,
                ),
            )
        else:
            cur.execute(
                "UPDATE canonical_records SET raw_payload = ? WHERE rowid = ?",
                (json.dumps(raw_payload, ensure_ascii=False), rowid),
            )
        logged += 1
    conn.commit()
    return seen, logged


def _backfill_staging(conn: sqlite3.Connection) -> Tuple[int, int]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid, source, source_record_id, merchant_raw, amount, description, mcc, mcc_description, category_external, raw_payload_json
        FROM staging_transactions
        """
    )
    rows = cur.fetchall()
    seen = 0
    logged = 0
    for (
        rowid,
        source,
        source_record_id,
        merchant_raw,
        amount,
        description,
        mcc,
        mcc_description,
        category_external,
        raw_payload_raw,
    ) in rows:
        seen += 1
        raw_payload = _safe_json_loads(raw_payload_raw, {})
        if not isinstance(raw_payload, dict):
            raw_payload = {}

        desc = str(description or "").strip()
        mcc_s = str(mcc or "").strip()
        mcc_desc_s = str(mcc_description or "").strip()
        ext = str(category_external or "").strip()

        shadow = categorization.build_shadow_decisions(
            merchant_raw=str(merchant_raw or "").strip(),
            amount=amount,
            description=desc,
            mcc=mcc_s,
            mcc_description=mcc_desc_s,
            category_external=ext,
            raw_payload=raw_payload,
            threshold_primary=0.85,
            threshold_alt=0.70,
        )

        shadow_key = _deterministic_key(
            rowid=int(rowid),
            record_id="",
            source=str(source or ""),
            source_record_id=str(source_record_id or ""),
            merchant_raw=str(merchant_raw or ""),
            amount=amount,
            description=desc,
        )
        raw_payload.update(shadow)
        raw_payload["shadow_record_key"] = shadow_key
        raw_payload["shadow_table"] = "staging_transactions"

        cur.execute(
            "UPDATE staging_transactions SET raw_payload_json = ? WHERE rowid = ?",
            (json.dumps(raw_payload, ensure_ascii=False), rowid),
        )
        logged += 1
    conn.commit()
    return seen, logged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/interim/staging.db")
    parser.add_argument(
        "--table",
        default="staging_transactions",
        choices=["staging_transactions", "canonical_records"],
    )
    parser.add_argument(
        "--overwrite-category-pred",
        action="store_true",
        help="Only for canonical_records: overwrite category_pred/confidence with rules.",
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}")
        return 1

    conn = sqlite3.connect(str(db))
    try:
        if args.table == "canonical_records":
            seen, logged = _backfill_canonical(conn, overwrite_category_pred=bool(args.overwrite_category_pred))
        else:
            seen, logged = _backfill_staging(conn)
    finally:
        conn.close()

    print("Shadow backfill complete.")
    print(f"- db: {db}")
    print(f"- table: {args.table}")
    print(f"- records_seen: {seen}")
    print(f"- records_logged: {logged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


