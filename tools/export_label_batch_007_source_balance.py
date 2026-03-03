#!/usr/bin/env python
"""Export source-diversity labeling queues (bank and toast) with rare-category oversampling."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple


RARE_TARGETS = {
    "Income",
    "Rent & Utilities",
    "COGS",
    "Professional Services",
}


def _fetch_all(conn: sqlite3.Connection, table: str) -> List[Dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY rowid")
    return [dict(r) for r in cur.fetchall()]


def _fingerprint_from_row(r: Dict) -> str:
    merchant = str(r.get("merchant_raw") or "").strip().lower()
    description = str(r.get("description") or "").strip().lower()
    amount = str(r.get("amount") if r.get("amount") is not None else "").strip().lower()
    return f"{merchant}\t{description}\t{amount}"


def _load_gold_fingerprints(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            out.add(_fingerprint_from_row(row))
    return out


def _source_bucket(source: str) -> str:
    s = str(source or "").strip().lower()
    if "toast" in s:
        return "toast"
    if s.startswith("bank") or "chase" in s or "bofa" in s:
        return "bank"
    return "other"


def _score_row(r: Dict) -> float:
    score = 0.0
    conf = float(r.get("rule_confidence") or 0.0)
    score += (1.0 - conf) * 2.0
    if int(r.get("missing_mcc") or 0):
        score += 0.8
    if int(r.get("merchant_only") or 0):
        score += 0.6
    rule_cat = str(r.get("rule_category") or "")
    if rule_cat in RARE_TARGETS:
        score += 2.0
    if "fallback:Other Expense" in str(r.get("mapping_reason") or ""):
        score += 0.7
    return score


def _pick_for_bucket(
    rows: List[Dict],
    *,
    target: int,
    seen_fps: Set[str],
) -> Tuple[List[Dict], int]:
    rows_sorted = sorted(rows, key=_score_row, reverse=True)
    picked: List[Dict] = []
    used: Set[int] = set()
    for r in rows_sorted:
        if len(picked) >= target:
            break
        rid = int(r.get("rowid") or 0)
        if rid in used:
            continue
        fp = _fingerprint_from_row(r)
        if fp in seen_fps:
            continue
        used.add(rid)
        seen_fps.add(fp)
        picked.append(r)
    shortage = max(0, target - len(picked))
    return picked, shortage


def _to_row(r: Dict, bucket: str) -> Dict:
    return {
        "rowid": r.get("rowid"),
        "batch_tag": "B007_source_balance",
        "target_bucket": bucket,
        "source": r.get("source"),
        "source_record_id": r.get("source_record_id"),
        "transaction_date": r.get("txn_ts"),
        "merchant_raw": r.get("merchant_raw"),
        "merchant_norm": r.get("merchant_norm"),
        "description": r.get("description"),
        "amount": r.get("amount"),
        "currency": r.get("currency"),
        "mcc": r.get("mcc"),
        "mcc_description": r.get("mcc_description"),
        "category_external": r.get("category_external"),
        "rule_category_suggested": r.get("rule_category"),
        "rule_confidence": r.get("rule_confidence"),
        "mapping_reason": r.get("mapping_reason"),
        "sourcetax_category_v1": "",
        "label_confidence": "",
        "label_notes": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--table", default="staging_transactions_enriched")
    parser.add_argument("--gold-path", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--bank-target", type=int, default=150)
    parser.add_argument("--toast-target", type=int, default=150)
    parser.add_argument("--out", default="outputs/label_batch_007_source_balance.csv")
    parser.add_argument("--summary-out", default="outputs/reports/label_batch_007_source_balance_summary.json")
    args = parser.parse_args()

    db = Path(args.staging_db)
    if not db.exists():
        print(f"Missing staging DB: {db}")
        return 1
    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_all(conn, args.table)
    except sqlite3.OperationalError:
        print(f"Missing table {args.table}. Run tools/build_staging_enriched.py first.")
        conn.close()
        return 1
    conn.close()

    bank_rows = [r for r in rows if _source_bucket(str(r.get("source") or "")) == "bank"]
    toast_rows = [r for r in rows if _source_bucket(str(r.get("source") or "")) == "toast"]

    seen_fps = _load_gold_fingerprints(Path(args.gold_path))
    picked_bank, bank_shortage = _pick_for_bucket(bank_rows, target=int(args.bank_target), seen_fps=seen_fps)
    picked_toast, toast_shortage = _pick_for_bucket(toast_rows, target=int(args.toast_target), seen_fps=seen_fps)

    output_rows = [_to_row(r, "bank") for r in picked_bank] + [_to_row(r, "toast") for r in picked_toast]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
    else:
        out.write_text("", encoding="utf-8")

    summary = {
        "staging_counts": {
            "bank_rows_available": len(bank_rows),
            "toast_rows_available": len(toast_rows),
        },
        "targets": {
            "bank_target": int(args.bank_target),
            "toast_target": int(args.toast_target),
        },
        "selected": {
            "bank_selected": len(picked_bank),
            "toast_selected": len(picked_toast),
            "total_selected": len(output_rows),
        },
        "shortages": {
            "bank_shortage": bank_shortage,
            "toast_shortage": toast_shortage,
        },
        "rare_category_focus": sorted(RARE_TARGETS),
        "out": str(out),
    }
    s_out = Path(args.summary_out)
    s_out.parent.mkdir(parents=True, exist_ok=True)
    s_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Batch 007 source-balance export complete.")
    print(f"- out: {out}")
    print(f"- bank_selected: {len(picked_bank)} / target {args.bank_target}")
    print(f"- toast_selected: {len(picked_toast)} / target {args.toast_target}")
    print(f"- total_selected: {len(output_rows)}")
    print(f"- shortages: bank={bank_shortage}, toast={toast_shortage}")
    print(f"- summary: {s_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

