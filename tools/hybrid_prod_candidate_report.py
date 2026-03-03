#!/usr/bin/env python
"""Report and export Stage-1 hybrid prod-candidate flips from shadow logs."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
            SELECT rowid, NULL AS id, source, source_record_id, merchant_raw, amount,
                   description, mcc_description, raw_payload_json AS raw_payload
            FROM staging_transactions
            """
        )
    return cur.fetchall()


def _record_key(row: sqlite3.Row, payload: Dict[str, Any]) -> str:
    k = str(payload.get("shadow_record_key") or "").strip()
    if k:
        return k
    rid = str(row["id"] or "").strip() if "id" in row.keys() else ""
    if rid:
        return f"id:{rid}"
    source = str(row["source"] or "").strip() if "source" in row.keys() else ""
    source_id = str(row["source_record_id"] or "").strip() if "source_record_id" in row.keys() else ""
    if source and source_id:
        return f"source:{source}:{source_id}"
    return f"rowid:{int(row['rowid'])}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/staging.db")
    parser.add_argument(
        "--table",
        default="staging_transactions",
        choices=["staging_transactions", "canonical_records"],
    )
    parser.add_argument("--review-sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="outputs/reports/hybrid_prod_candidate_flips.csv")
    parser.add_argument("--out-sample-csv", default="outputs/reports/hybrid_prod_candidate_flips_sample.csv")
    parser.add_argument("--out-json", default="outputs/reports/hybrid_prod_candidate_summary.json")
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

    total = 0
    shadow_rows = 0
    eligible: List[Dict[str, Any]] = []
    gate_rule_conf = 0
    gate_disagree = 0
    gate_ml_conf = 0
    gate_disagree_rule_conf = 0
    gate_disagree_ml_conf = 0
    gate_rule_conf_ml_conf = 0
    for row in rows:
        total += 1
        payload = _safe_json_loads(row["raw_payload"], {})
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("shadow_mode") is not True:
            continue
        shadow_rows += 1

        rule = str(payload.get("rule_category") or "").strip()
        ml = str(payload.get("ml_prediction") or "").strip()
        prod = str(payload.get("hybrid_prediction_prod_candidate") or "").strip()
        if not rule or not ml or not prod:
            continue

        try:
            rule_conf = float(payload.get("rule_confidence") or 0.0)
        except Exception:
            rule_conf = 0.0
        try:
            ml_conf = float(payload.get("ml_confidence") or 0.0)
        except Exception:
            ml_conf = 0.0
        rule_gate = rule_conf < 0.95
        disagree_gate = rule != ml
        ml_gate = ml_conf >= 0.30
        if rule_gate:
            gate_rule_conf += 1
        if disagree_gate:
            gate_disagree += 1
        if ml_gate:
            gate_ml_conf += 1
        if disagree_gate and rule_gate:
            gate_disagree_rule_conf += 1
        if disagree_gate and ml_gate:
            gate_disagree_ml_conf += 1
        if rule_gate and ml_gate:
            gate_rule_conf_ml_conf += 1

        eligible_flip = bool(payload.get("hybrid_prod_candidate_eligible_flip")) or (prod != rule)
        if not eligible_flip:
            continue

        eligible.append(
            {
                "shadow_record_key": _record_key(row, payload),
                "source": str(row["source"] or "").strip() if "source" in row.keys() else "",
                "source_record_id": str(row["source_record_id"] or "").strip() if "source_record_id" in row.keys() else "",
                "merchant_raw": str(row["merchant_raw"] or "").strip() if "merchant_raw" in row.keys() else "",
                "description": str(
                    payload.get("description")
                    or payload.get("ocr_text")
                    or (row["description"] if "description" in row.keys() else "")
                    or ""
                ).strip(),
                "mcc_description": str(
                    payload.get("mcc_description")
                    or (row["mcc_description"] if "mcc_description" in row.keys() else "")
                    or ""
                ).strip(),
                "amount": row["amount"] if "amount" in row.keys() else None,
                "rule_category": rule,
                "rule_confidence": rule_conf,
                "ml_prediction": ml,
                "ml_confidence": ml_conf,
                "hybrid_prediction_prod_candidate": prod,
                "mapping_reason": "|".join(payload.get("rule_reason") or []),
                "ml_model_status": str(payload.get("ml_model_status") or "").strip(),
            }
        )

    eligible.sort(key=lambda x: (x["rule_confidence"], -x["ml_confidence"]))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if eligible:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(eligible[0].keys()))
            writer.writeheader()
            writer.writerows(eligible)
    else:
        out_csv.write_text("", encoding="utf-8")

    rng = random.Random(args.seed)
    sample_n = max(0, int(args.review_sample_size))
    sample_rows = rng.sample(eligible, min(sample_n, len(eligible))) if eligible else []
    out_sample = Path(args.out_sample_csv)
    out_sample.parent.mkdir(parents=True, exist_ok=True)
    if sample_rows:
        with out_sample.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)
    else:
        out_sample.write_text("", encoding="utf-8")

    flip_rate = (len(eligible) / shadow_rows) if shadow_rows else 0.0
    summary = {
        "db": str(db),
        "table": args.table,
        "total_rows": total,
        "shadow_rows": shadow_rows,
        "eligible_flips": len(eligible),
        "eligible_flip_rate": flip_rate,
        "gate_counts": {
            "rule_conf_lt_0_95": gate_rule_conf,
            "rule_ne_ml": gate_disagree,
            "ml_conf_gte_0_30": gate_ml_conf,
            "rule_ne_ml_and_rule_conf_lt_0_95": gate_disagree_rule_conf,
            "rule_ne_ml_and_ml_conf_gte_0_30": gate_disagree_ml_conf,
            "rule_conf_lt_0_95_and_ml_conf_gte_0_30": gate_rule_conf_ml_conf,
        },
        "review_sample_size": len(sample_rows),
        "out_csv": str(out_csv),
        "out_sample_csv": str(out_sample),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Hybrid prod-candidate report complete.")
    print(f"- total_rows: {total}")
    print(f"- shadow_rows: {shadow_rows}")
    print(f"- eligible_flips: {len(eligible)}")
    print(f"- eligible_flip_rate: {flip_rate:.4f}")
    print(f"- rule_conf_lt_0_95: {gate_rule_conf}")
    print(f"- rule_ne_ml: {gate_disagree}")
    print(f"- ml_conf_gte_0_30: {gate_ml_conf}")
    print(f"- rule_ne_ml_and_rule_conf_lt_0_95: {gate_disagree_rule_conf}")
    print(f"- rule_ne_ml_and_ml_conf_gte_0_30: {gate_disagree_ml_conf}")
    print(f"- rule_conf_lt_0_95_and_ml_conf_gte_0_30: {gate_rule_conf_ml_conf}")
    print(f"- out_csv: {out_csv}")
    print(f"- out_sample_csv: {out_sample}")
    print(f"- out_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
