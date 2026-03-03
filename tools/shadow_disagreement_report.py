#!/usr/bin/env python
"""Report shadow-mode disagreement rates and slices."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
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


def _conf_bin(v: float) -> str:
    if v < 0.5:
        return "<0.50"
    if v < 0.7:
        return "0.50-0.69"
    if v < 0.85:
        return "0.70-0.84"
    return ">=0.85"


def _fetch_rows(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if table == "canonical_records":
        cur.execute(
            """
            SELECT rowid, merchant_raw, amount, raw_payload
            FROM canonical_records
            """
        )
    else:
        cur.execute(
            """
            SELECT rowid, merchant_raw, amount, description, mcc_description, raw_payload_json AS raw_payload
            FROM staging_transactions
            """
        )
    return cur.fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/interim/staging.db")
    parser.add_argument(
        "--table",
        default="staging_transactions",
        choices=["staging_transactions", "canonical_records"],
    )
    parser.add_argument("--out", default="artifacts/reports/shadow_disagreement_report.json")
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
    with_shadow = 0
    with_both = 0
    rule_ml_disagree = 0
    hybrid85_rule_diff = 0
    hybrid70_rule_diff = 0

    by_rule_category = Counter()
    by_rule_category_disagree = Counter()
    by_missing_mcc = Counter()
    by_missing_mcc_disagree = Counter()
    by_merchant_only = Counter()
    by_merchant_only_disagree = Counter()
    by_conf_bin = Counter()
    by_conf_bin_disagree = Counter()

    for row in rows:
        total += 1
        payload = _safe_json_loads(row["raw_payload"], {})
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("shadow_mode") is not True:
            continue
        with_shadow += 1
        rule = str(payload.get("rule_category") or "").strip()
        ml = str(payload.get("ml_prediction") or "").strip()
        h85 = str(payload.get("hybrid_prediction_t85") or payload.get("hybrid_prediction") or "").strip()
        h70 = str(payload.get("hybrid_prediction_t70") or "").strip()
        conf = float(payload.get("rule_confidence") or 0.0)

        description = str(payload.get("description") or payload.get("ocr_text") or (row["description"] if "description" in row.keys() else "") or "").strip()
        mcc_desc = str(payload.get("mcc_description") or (row["mcc_description"] if "mcc_description" in row.keys() else "") or "").strip()
        merchant = str((row["merchant_raw"] if "merchant_raw" in row.keys() else "") or "").strip()

        missing_mcc = "missing_mcc" if not mcc_desc else "has_mcc"
        merchant_only = "merchant_only" if merchant and not description else "has_description"
        cbin = _conf_bin(conf)

        by_missing_mcc[missing_mcc] += 1
        by_merchant_only[merchant_only] += 1
        by_conf_bin[cbin] += 1
        if rule:
            by_rule_category[rule] += 1

        if rule and ml:
            with_both += 1
            disagree = rule != ml
            if disagree:
                rule_ml_disagree += 1
                by_missing_mcc_disagree[missing_mcc] += 1
                by_merchant_only_disagree[merchant_only] += 1
                by_conf_bin_disagree[cbin] += 1
                by_rule_category_disagree[rule] += 1
        if rule and h85 and rule != h85:
            hybrid85_rule_diff += 1
        if rule and h70 and rule != h70:
            hybrid70_rule_diff += 1

    report = {
        "db": str(db),
        "table": args.table,
        "total_rows": total,
        "shadow_rows": with_shadow,
        "rows_with_rule_and_ml": with_both,
        "rule_vs_ml_disagreement_rate": (rule_ml_disagree / with_both) if with_both else 0.0,
        "hybrid_t85_vs_rule_rate": (hybrid85_rule_diff / with_shadow) if with_shadow else 0.0,
        "hybrid_t70_vs_rule_rate": (hybrid70_rule_diff / with_shadow) if with_shadow else 0.0,
        "by_rule_category": dict(by_rule_category),
        "by_rule_category_disagree": dict(by_rule_category_disagree),
        "by_missing_mcc": dict(by_missing_mcc),
        "by_missing_mcc_disagree": dict(by_missing_mcc_disagree),
        "by_merchant_only": dict(by_merchant_only),
        "by_merchant_only_disagree": dict(by_merchant_only_disagree),
        "by_rule_conf_bin": dict(by_conf_bin),
        "by_rule_conf_bin_disagree": dict(by_conf_bin_disagree),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Shadow disagreement report complete.")
    print(f"- out: {out}")
    print(f"- total_rows: {total}")
    print(f"- shadow_rows: {with_shadow}")
    print(f"- rows_with_rule_and_ml: {with_both}")
    print(f"- rule_vs_ml_disagreement_rate: {report['rule_vs_ml_disagreement_rate']:.3f}")
    print(f"- hybrid_t85_vs_rule_rate: {report['hybrid_t85_vs_rule_rate']:.3f}")
    print(f"- hybrid_t70_vs_rule_rate: {report['hybrid_t70_vs_rule_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

