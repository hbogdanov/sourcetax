#!/usr/bin/env python
"""Export targeted Batch 003 for low-support categories in gold.

Targets:
- Education & Training
- Vehicle Expenses
- COGS
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Set


TARGET_CATEGORIES = ["Education & Training", "Vehicle Expenses", "COGS"]

PATTERNS = {
    "Education & Training": [
        r"\bUDEMY\b",
        r"\bCOURSERA\b",
        r"\bEDX\b",
        r"\bPLURALSIGHT\b",
        r"\bDATACAMP\b",
        r"\bO[' ]?REILLY\b",
        r"\bPACKT\b",
        r"\bSAFARIBOOKSONLINE\b",
        r"\bTRAIN(ING)?\b",
        r"\bCOURSE\b",
        r"\bCERT(IFICATION)?\b",
        r"\bEXAM\b",
        r"\bTEST FEE\b",
        r"\bCONFERENCE\b",
        r"\bSUMMIT\b",
        r"\bWORKSHOP\b",
        r"\bBOOTCAMP\b",
        r"\bTUITION\b",
        r"\bENROLLMENT\b",
        r"\bEDUCATIONAL SERVICES\b",
        r"\bBOOK STORES?\b",
        r"\bBOOKS?/PERIODICALS?\b",
    ],
    "Vehicle Expenses": [
        r"\bSHELL\b",
        r"\bCHEVRON\b",
        r"\bEXXON\b",
        r"\bBP\b",
        r"\bMARATHON\b",
        r"\bSUNOCO\b",
        r"\bRACETRAC\b",
        r"\bQT\b",
        r"\bSPEEDWAY\b",
        r"\bGAS\b",
        r"\bFUEL\b",
        r"\bJIFFY LUBE\b",
        r"\bPEP BOYS\b",
        r"\bAUTOZONE\b",
        r"\bADVANCE AUTO\b",
        r"\bNAPA\b",
        r"\bPARKING\b",
        r"\bTOLL\b",
        r"\bE[- ]?ZPASS\b",
        r"\bSUNPASS\b",
        r"\bCAR WASH\b",
        r"\bSERVICE STATIONS?\b",
        r"\bAUTOMOTIVE FUEL\b",
        r"\bAUTO PARTS?\b",
        r"\bAUTO SERVICE\b",
        r"\bPARKING LOTS?/GARAGES?\b",
    ],
    "COGS": [
        r"\bWHOLESALE\b",
        r"\bDISTRIBUTOR\b",
        r"\bRAW MATERIALS?\b",
        r"\bMANUFACTUR(ING|ER)?\b",
        r"\bINDUSTRIAL SUPPL(IES|Y)\b",
        r"\bINVENTORY\b",
        r"\bMATERIALS?\b",
        r"\bRESALE\b",
        r"\bWHOLESALE SUPPLIERS?\b",
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


def _fetch_rows(conn: sqlite3.Connection, table: str) -> List[Dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY rowid")
    return [dict(r) for r in cur.fetchall()]


def _text_blob(row: Dict) -> str:
    return " ".join(
        [
            str(row.get("merchant_raw") or ""),
            str(row.get("description") or ""),
            str(row.get("mcc_description") or ""),
        ]
    ).upper()


def _score(row: Dict, category: str) -> float:
    txt = _text_blob(row)
    score = 0.0
    for pat in PATTERNS.get(category, []):
        if re.search(pat, txt):
            score += 1.0
    # Prefer low/medium-confidence edge cases first.
    conf = float(row.get("rule_confidence") or 0.0)
    score += max(0.0, 0.9 - conf)
    # Weakly prefer rows where suggestion already aligns.
    if (row.get("rule_category") or "") == category:
        score += 0.5
    return score


def _to_output_row(row: Dict, target_category: str) -> Dict:
    return {
        "rowid": row.get("rowid"),
        "batch_tag": "BATCH_003_low_support",
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
    parser.add_argument("--gold-path", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--out", default="outputs/label_batch_003.csv")
    parser.add_argument("--per-category", type=int, default=20)
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

    used_fps: Set[str] = _load_gold_fingerprints(Path(args.gold_path))
    used_rowids: Set[int] = set()
    selected: List[Dict] = []
    shortages: Dict[str, int] = {}

    for cat in TARGET_CATEGORIES:
        candidates = [r for r in rows if any(re.search(p, _text_blob(r)) for p in PATTERNS[cat])]
        candidates.sort(key=lambda r: _score(r, cat), reverse=True)
        picked = 0
        for r in candidates:
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
            shortages[cat] = args.per_category - picked

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0].keys()) if selected else []
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(selected)

    print("Batch 003 export complete.")
    print(f"- out: {out}")
    print(f"- total_rows: {len(selected)}")
    for cat in TARGET_CATEGORIES:
        n = sum(1 for r in selected if r["target_category"] == cat)
        print(f"- {cat}: {n}")
    if shortages:
        print("- shortages:")
        for k, v in shortages.items():
            print(f"  - {k}: missing {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

