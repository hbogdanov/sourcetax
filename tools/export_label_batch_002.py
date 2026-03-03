#!/usr/bin/env python
"""Export targeted Batch 002 for missing categories.

Targets:
- Financial Fees
- Insurance
- Payroll & Contractors

Selection is heuristic-driven from merchant/description/MCC text and prefers
low/medium confidence rows (default max confidence = 0.85).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Set


TARGET_CATEGORIES = ["Financial Fees", "Insurance", "Payroll & Contractors"]

PATTERNS = {
    "Financial Fees": [
        r"\bFEE\b",
        r"\bSERVICE FEE\b",
        r"\bWIRE FEE\b",
        r"\bOVERDRAFT\b",
        r"\bINTEREST\b",
        r"\bNSF\b",
        r"\bATM\b",
        r"\bBANK CHARGE\b",
        r"\bTOLLS?\b",
    ],
    "Insurance": [
        r"\bINSURANCE\b",
        r"\bGEICO\b",
        r"\bPROGRESSIVE\b",
        r"\bSTATE FARM\b",
        r"\bTRAVELERS?\b",
        r"\bHISCOX\b",
    ],
    "Payroll & Contractors": [
        r"\bPAYROLL\b",
        r"\bADP\b",
        r"\bGUSTO\b",
        r"\bPAYCHEX\b",
        r"\bCONTRACTOR\b",
        r"\b1099\b",
        r"\bFREELANCE\b",
        r"\bSTAFFING\b",
    ],
}


def _fingerprint(row: Dict) -> str:
    merchant = str(row.get("merchant_raw") or "").strip().lower()
    description = str(row.get("description") or "").strip().lower()
    amount = str(row.get("amount") if row.get("amount") is not None else "").strip().lower()
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
            out.add(_fingerprint(row))
    return out


def _text_blob(row: Dict) -> str:
    return " ".join(
        [
            str(row.get("merchant_raw") or ""),
            str(row.get("description") or ""),
            str(row.get("mcc_description") or ""),
        ]
    ).upper()


def _match_target(row: Dict, category: str) -> bool:
    blob = _text_blob(row)
    for pat in PATTERNS.get(category, []):
        if re.search(pat, blob):
            return True
    return False


def _score_row(row: Dict, category: str, max_conf: float) -> float:
    blob = _text_blob(row)
    score = 0.0
    for pat in PATTERNS.get(category, []):
        if re.search(pat, blob):
            score += 1.0
    conf = float(row.get("rule_confidence") or 0.0)
    if conf <= max_conf:
        score += 2.0
    score += max(0.0, 0.85 - conf)
    if int(row.get("missing_mcc") or 0):
        score += 0.4
    return score


def _fetch_rows(conn: sqlite3.Connection, table: str) -> List[Dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY rowid")
    return [dict(r) for r in cur.fetchall()]


def _to_output_row(row: Dict, target_category: str) -> Dict:
    return {
        "rowid": row.get("rowid"),
        "batch_tag": "BATCH_002_missing_categories",
        "target_category": target_category,
        "source": row.get("source"),
        "source_record_id": row.get("source_record_id"),
        "transaction_date": row.get("txn_ts"),
        "merchant_raw": row.get("merchant_raw"),
        "merchant_norm": row.get("merchant_norm"),
        "description": row.get("description"),
        "amount": row.get("amount"),
        "currency": row.get("currency"),
        "mcc": row.get("mcc"),
        "mcc_description": row.get("mcc_description"),
        "category_external": row.get("category_external"),
        "rule_category_suggested": row.get("rule_category"),
        "rule_confidence": row.get("rule_confidence"),
        "mapping_reason": row.get("mapping_reason"),
        "sourcetax_category_v1": "",
        "label_confidence": "",
        "label_notes": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--table", default="staging_transactions_enriched")
    parser.add_argument("--out", default="outputs/label_batch_002.csv")
    parser.add_argument("--gold-path", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--max-confidence", type=float, default=0.85)
    args = parser.parse_args()

    db = Path(args.staging_db)
    if not db.exists():
        print(f"Missing staging DB: {db}")
        return 1

    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_rows(conn, args.table)
    except sqlite3.OperationalError:
        print(f"Missing table {args.table}. Run tools/build_staging_enriched.py first.")
        conn.close()
        return 1
    conn.close()

    gold_fps = _load_gold_fingerprints(Path(args.gold_path))
    used_fps: Set[str] = set(gold_fps)
    used_rowids: Set[int] = set()
    selected: List[Dict] = []
    shortage: Dict[str, int] = {}

    for cat in TARGET_CATEGORIES:
        candidates = [r for r in rows if _match_target(r, cat)]
        candidates.sort(key=lambda r: _score_row(r, cat, args.max_confidence), reverse=True)

        picked = 0
        deferred_high_conf: List[Dict] = []
        for r in candidates:
            rid = int(r.get("rowid") or 0)
            if rid in used_rowids:
                continue
            fp = _fingerprint(r)
            if fp in used_fps:
                continue
            conf = float(r.get("rule_confidence") or 0.0)
            if conf > args.max_confidence:
                deferred_high_conf.append(r)
                continue
            used_rowids.add(rid)
            used_fps.add(fp)
            selected.append(_to_output_row(r, cat))
            picked += 1
            if picked >= args.per_category:
                break

        # Backfill from high-confidence candidates if low/medium pool is insufficient.
        if picked < args.per_category:
            for r in deferred_high_conf:
                rid = int(r.get("rowid") or 0)
                if rid in used_rowids:
                    continue
                fp = _fingerprint(r)
                if fp in used_fps:
                    continue
                used_rowids.add(rid)
                used_fps.add(fp)
                selected.append(_to_output_row(r, cat))
                picked += 1
                if picked >= args.per_category:
                    break
        if picked < args.per_category:
            shortage[cat] = args.per_category - picked

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0].keys()) if selected else []
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(selected)

    print("Targeted label batch export complete.")
    print(f"- out: {out}")
    print(f"- total_rows: {len(selected)}")
    for cat in TARGET_CATEGORIES:
        n = sum(1 for r in selected if r["target_category"] == cat)
        print(f"- {cat}: {n}")
    if shortage:
        print("- shortages:")
        for k, v in shortage.items():
            print(f"  - {k}: missing {v} rows (insufficient candidates under filters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
