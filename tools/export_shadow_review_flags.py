#!/usr/bin/env python
"""Export ML-vs-rule disagreement rows for human review while hybrid stays shadow-only."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


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


def _fetch_rows(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if table == "canonical_records":
        cur.execute(
            """
            SELECT rowid, id, source, source_record_id, merchant_raw, amount, raw_payload
            FROM canonical_records
            """
        )
    else:
        cur.execute(
            """
            SELECT rowid, NULL AS id, source, source_record_id, merchant_raw, amount, description, mcc_description, raw_payload_json AS raw_payload
            FROM staging_transactions
            """
        )
    return cur.fetchall()


def _record_key(row: sqlite3.Row, payload: Dict[str, Any]) -> str:
    key = str(payload.get("shadow_record_key") or "").strip()
    if key:
        return key
    rid = str(row["id"] or "").strip() if "id" in row.keys() else ""
    if rid:
        return f"id:{rid}"
    src = str(row["source"] or "").strip()
    src_id = str(row["source_record_id"] or "").strip()
    if src and src_id:
        return f"source:{src}:{src_id}"
    return f"rowid:{int(row['rowid'])}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/staging.db")
    parser.add_argument(
        "--table",
        default="staging_transactions",
        choices=["staging_transactions", "canonical_records"],
    )
    parser.add_argument("--rule-conf-max", type=float, default=0.70)
    parser.add_argument("--ml-conf-min", type=float, default=0.30)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--out-csv", default="outputs/reports/shadow_review_candidates.csv")
    parser.add_argument("--out-json", default="outputs/reports/shadow_review_candidates_summary.json")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}")
        return 1

    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_rows(conn, args.table)
    finally:
        conn.close()

    candidates: List[Dict[str, Any]] = []
    shadow_rows = 0
    eligible_disagreements = 0
    for row in rows:
        payload = _safe_json_loads(row["raw_payload"], {})
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("shadow_mode") is not True:
            continue
        shadow_rows += 1

        rule = str(payload.get("rule_category") or "").strip()
        ml = str(payload.get("ml_prediction") or "").strip()
        if not rule or not ml or rule == ml:
            continue

        try:
            rule_conf = float(payload.get("rule_confidence") or 0.0)
        except Exception:
            rule_conf = 0.0
        try:
            ml_conf = float(payload.get("ml_confidence") or 0.0)
        except Exception:
            ml_conf = 0.0

        if rule_conf < float(args.rule_conf_max) and ml_conf >= float(args.ml_conf_min):
            eligible_disagreements += 1
            description = str(payload.get("description") or payload.get("ocr_text") or (row["description"] if "description" in row.keys() else "") or "").strip()
            mcc_desc = str(payload.get("mcc_description") or (row["mcc_description"] if "mcc_description" in row.keys() else "") or "").strip()
            candidates.append(
                {
                    "shadow_record_key": _record_key(row, payload),
                    "source": str(row["source"] or "").strip() if "source" in row.keys() else "",
                    "source_record_id": str(row["source_record_id"] or "").strip() if "source_record_id" in row.keys() else "",
                    "merchant_raw": str(row["merchant_raw"] or "").strip() if "merchant_raw" in row.keys() else "",
                    "description": description,
                    "mcc_description": mcc_desc,
                    "amount": row["amount"] if "amount" in row.keys() else None,
                    "rule_category": rule,
                    "rule_confidence": rule_conf,
                    "ml_prediction": ml,
                    "ml_confidence": ml_conf,
                    "hybrid_prediction_t70": str(payload.get("hybrid_prediction_t70") or "").strip(),
                    "hybrid_prediction_t85": str(payload.get("hybrid_prediction_t85") or payload.get("hybrid_prediction") or "").strip(),
                    "mapping_reason": "|".join(payload.get("rule_reason") or []),
                    "ml_model_status": str(payload.get("ml_model_status") or "").strip(),
                }
            )

    candidates.sort(key=lambda x: (x["rule_confidence"], -x["ml_confidence"]))
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if candidates:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(candidates[0].keys()))
            writer.writeheader()
            writer.writerows(candidates)
    else:
        out_csv.write_text("", encoding="utf-8")

    summary = {
        "db": str(db),
        "table": args.table,
        "shadow_rows": shadow_rows,
        "rule_conf_max": float(args.rule_conf_max),
        "ml_conf_min": float(args.ml_conf_min),
        "eligible_disagreements": eligible_disagreements,
        "exported_rows": len(candidates),
        "out_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Shadow review candidate export complete.")
    print(f"- shadow_rows: {shadow_rows}")
    print(f"- eligible_disagreements: {eligible_disagreements}")
    print(f"- exported_rows: {len(candidates)}")
    print(f"- out_csv: {out_csv}")
    print(f"- out_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

