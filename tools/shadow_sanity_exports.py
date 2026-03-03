#!/usr/bin/env python
"""Export high-signal shadow sanity artifacts from staging shadow logs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from sourcetax.normalization import normalize_merchant_name
from sourcetax.taxonomy import normalize_category_name


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


def _fetch_rows(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid, source, source_record_id, merchant_raw, description, mcc_description, amount, raw_payload_json
        FROM staging_transactions
        """
    )
    return cur.fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/interim/staging.db")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts/reports")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}")
        return 1

    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_rows(conn)
    finally:
        conn.close()

    enriched: List[Dict[str, Any]] = []
    rule_dist = Counter()
    ml_dist = Counter()
    invalid_rule = 0
    invalid_ml = 0

    for r in rows:
        payload = _safe_json_loads(r["raw_payload_json"], {})
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("shadow_mode") is not True:
            continue
        merchant_raw = str(r["merchant_raw"] or "").strip()
        merchant_norm = normalize_merchant_name(merchant_raw, case="lower") if merchant_raw else ""
        description = str(r["description"] or payload.get("description") or payload.get("ocr_text") or "").strip()
        mcc_description = str(r["mcc_description"] or payload.get("mcc_description") or "").strip()
        rule_category = str(payload.get("rule_category") or "").strip()
        ml_prediction = str(payload.get("ml_prediction") or "").strip()
        rule_conf = payload.get("rule_confidence")
        ml_conf = payload.get("ml_confidence")
        hyb70 = str(payload.get("hybrid_prediction_t70") or "").strip()
        hyb85 = str(payload.get("hybrid_prediction_t85") or payload.get("hybrid_prediction") or "").strip()
        reason = payload.get("rule_reason") or payload.get("mapping_reason") or []
        if isinstance(reason, list):
            mapping_reason = "|".join(str(x) for x in reason)
        else:
            mapping_reason = str(reason)

        rule_dist[rule_category or "<blank>"] += 1
        ml_dist[ml_prediction or "<blank>"] += 1
        if rule_category and not normalize_category_name(rule_category):
            invalid_rule += 1
        if ml_prediction and not normalize_category_name(ml_prediction):
            invalid_ml += 1

        enriched.append(
            {
                "shadow_record_key": str(payload.get("shadow_record_key") or f"rowid:{r['rowid']}"),
                "merchant_norm": merchant_norm,
                "description": description,
                "mcc_description": mcc_description,
                "amount": r["amount"],
                "rule_category": rule_category,
                "rule_confidence": rule_conf,
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_conf,
                "hybrid_prediction_t70": hyb70,
                "hybrid_prediction_t85": hyb85,
                "mapping_reason": mapping_reason,
            }
        )

    rng = random.Random(args.seed)
    sample = list(enriched)
    rng.shuffle(sample)
    sample = sample[: max(0, args.sample_size)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "shadow_sanity_sample.csv"
    ml_path = out_dir / "ml_pred_distribution.csv"
    rule_path = out_dir / "rule_distribution.csv"

    if sample:
        with sample_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(sample[0].keys()))
            writer.writeheader()
            writer.writerows(sample)
    else:
        sample_path.write_text("", encoding="utf-8")

    with ml_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "count"])
        writer.writeheader()
        for k, v in ml_dist.most_common():
            writer.writerow({"category": k, "count": v})

    with rule_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "count"])
        writer.writeheader()
        for k, v in rule_dist.most_common():
            writer.writerow({"category": k, "count": v})

    print("Shadow sanity exports complete.")
    print(f"- shadow_rows: {len(enriched)}")
    print(f"- sample_out: {sample_path}")
    print(f"- ml_distribution_out: {ml_path}")
    print(f"- rule_distribution_out: {rule_path}")
    print(f"- invalid_rule_labels: {invalid_rule}")
    print(f"- invalid_ml_labels: {invalid_ml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


