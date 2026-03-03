#!/usr/bin/env python
"""Export sparse-category-only labeling batch with aggressive merchant/MCC heuristics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Set


TARGETS = [
    "Income",
    "Rent & Utilities",
    "Financial Fees",
    "Professional Services",
]

PATTERNS = {
    "Income": [
        r"\bINVOICE\b",
        r"\bPAYMENT RECEIVED\b",
        r"\bDEPOSIT\b",
        r"\bACH CREDIT\b",
        r"\bCLIENT PAYMENT\b",
        r"\bSTRIPE\b",
        r"\bSQUARE\b",
        r"\bPAYPAL\b",
        r"\bREVENUE\b",
        r"\bSALES\b",
    ],
    "Rent & Utilities": [
        r"\bRENT\b",
        r"\bLEASE\b",
        r"\bLANDLORD\b",
        r"\bPROPERTY MGMT\b",
        r"\bELECTRIC\b",
        r"\bPOWER\b",
        r"\bWATER\b",
        r"\bSEWER\b",
        r"\bGAS BILL\b",
        r"\bINTERNET\b",
        r"\bCOMCAST\b",
        r"\bVERIZON\b",
        r"\bAT&T\b",
        r"\bUTILITY\b",
        r"\bTELECOMMUNICATION\b",
    ],
    "Financial Fees": [
        r"\bFEE\b",
        r"\bINTEREST\b",
        r"\bFINANCE CHARGE\b",
        r"\bSERVICE CHARGE\b",
        r"\bOVERDRAFT\b",
        r"\bNSF\b",
        r"\bWIRE FEE\b",
        r"\bACH FEE\b",
        r"\bMONTHLY MAINTENANCE\b",
        r"\bPROCESSING FEE\b",
        r"\bCHARGEBACK\b",
        r"\bMERCHANT FEE\b",
    ],
    "Professional Services": [
        r"\bCONSULT(ING|ANT)?\b",
        r"\bADVIS(OR|ORY)\b",
        r"\bLEGAL\b",
        r"\bLAW\b",
        r"\bATTORNEY\b",
        r"\bCPA\b",
        r"\bACCOUNT(ING|ANT)\b",
        r"\bBOOKKEEP(ING|ER)\b",
        r"\bAUDIT\b",
        r"\bNOTARY\b",
        r"\bCOMPLIANCE\b",
        r"\bPROFESSIONAL SERVICES\b",
    ],
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


def _blob(r: Dict) -> str:
    return " | ".join(
        [
            str(r.get("merchant_raw") or ""),
            str(r.get("merchant_norm") or ""),
            str(r.get("description") or ""),
            str(r.get("mcc_description") or ""),
            str(r.get("category_external") or ""),
            str(r.get("mapping_reason") or ""),
        ]
    ).upper()


def _score(r: Dict, target: str) -> float:
    text = _blob(r)
    conf = float(r.get("rule_confidence") or 0.0)
    rule_cat = str(r.get("rule_category") or "")
    score = 0.0
    for p in PATTERNS.get(target, []):
        if re.search(p, text):
            score += 1.0
    if rule_cat == target:
        score += 1.0
    if "mcc:" in str(r.get("mapping_reason") or ""):
        score += 0.5
    score += (1.0 - conf) * 1.5
    if int(r.get("missing_mcc") or 0):
        score += 0.3
    return score


def _to_row(r: Dict, target: str) -> Dict:
    return {
        "rowid": r.get("rowid"),
        "batch_tag": "B006_sparse_only",
        "target_category": target,
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
    parser.add_argument("--out", default="outputs/label_batch_006_sparse.csv")
    parser.add_argument("--summary-out", default="outputs/reports/label_batch_006_sparse_summary.json")
    parser.add_argument("--per-category", type=int, default=25)
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

    seen_fps = _load_gold_fingerprints(Path(args.gold_path))
    used_rowids: Set[int] = set()
    selected: List[Dict] = []
    shortages: Dict[str, int] = {}

    for target in TARGETS:
        cands = rows[:]
        cands.sort(key=lambda r: _score(r, target), reverse=True)
        picked = 0
        for r in cands:
            rid = int(r.get("rowid") or 0)
            if rid in used_rowids:
                continue
            fp = _fingerprint_from_row(r)
            if fp in seen_fps:
                continue
            if _score(r, target) <= 0:
                continue
            used_rowids.add(rid)
            seen_fps.add(fp)
            selected.append(_to_row(r, target))
            picked += 1
            if picked >= args.per_category:
                break
        if picked < args.per_category:
            shortages[target] = args.per_category - picked

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if selected:
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
    else:
        out.write_text("", encoding="utf-8")

    summary = {
        "out": str(out),
        "total_rows": len(selected),
        "per_category": args.per_category,
        "counts": {t: sum(1 for r in selected if r["target_category"] == t) for t in TARGETS},
        "shortages": shortages,
    }
    s_out = Path(args.summary_out)
    s_out.parent.mkdir(parents=True, exist_ok=True)
    s_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Batch 006 sparse export complete.")
    print(f"- out: {out}")
    print(f"- total_rows: {len(selected)}")
    for t in TARGETS:
        n = sum(1 for r in selected if r["target_category"] == t)
        print(f"- {t}: {n}")
    if shortages:
        print(f"- shortages: {shortages}")
    print(f"- summary: {s_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

