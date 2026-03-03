#!/usr/bin/env python
"""
Merge labeled CSV exports into `data/gold/gold_transactions.jsonl` safely.

Usage:
  python tools/merge_labeled.py --input labeled.csv --gold data/gold/gold_transactions.jsonl

Behavior:
 - Reads existing gold JSONL, builds a fingerprint set (merchant+description+amount)
 - Reads input CSV, normalizes fields, appends non-duplicates to gold JSONL
 - Prints summary counts
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sourcetax import taxonomy
from sourcetax.gold import normalize_label_confidence, normalize_label_notes


def fingerprint(rec: dict) -> str:
    merchant = rec.get("merchant_raw") or rec.get("merchant") or ""
    description = (rec.get("raw_payload") or {}).get("description", "") if isinstance(rec.get("raw_payload"), dict) else rec.get("description", "")
    amount = rec.get("amount", "")
    return f"{merchant}\t{description}\t{amount}".lower()


def load_gold(gold_path: Path) -> list:
    if not gold_path.exists():
        return []
    out = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def append_to_gold(gold_path: Path, records: list) -> int:
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with gold_path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            appended += 1
    return appended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    args = parser.parse_args()

    in_path = Path(args.input)
    gold_path = Path(args.gold)

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return 1

    import csv

    # Load existing gold
    gold = load_gold(gold_path)
    seen = set(fingerprint(r) for r in gold)

    new_records = []
    with in_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            category = taxonomy.normalize_category_name(r.get("sourcetax_category_v1") or r.get("category"))
            if not category:
                continue
            rec = {
                "id": r.get("id") or None,
                "source_record_id": r.get("source_record_id") or "",
                "merchant_raw": r.get("merchant_raw") or r.get("merchant") or r.get("vendor") or r.get("Vendor") or "",
                "merchant_norm": r.get("merchant_norm") or r.get("merchant") or r.get("vendor") or r.get("Vendor") or "",
                "transaction_date": r.get("transaction_date") or "",
                "amount": r.get("amount") or r.get("Amount") or None,
                "currency": r.get("currency") or "USD",
                "mcc": r.get("mcc") or "",
                "mcc_description": r.get("mcc_description") or "",
                "category_external": r.get("category_external") or "",
                "category_final": category,
                "sourcetax_category_v1": category,
                "label_confidence": normalize_label_confidence(r.get("label_confidence")),
                "label_notes": normalize_label_notes(r.get("label_notes")),
                "source": r.get("source") or "labeled",
                "raw_payload": {
                    "description": r.get("description") or r.get("text") or "",
                    "label_source": "human",
                    "mapping_reason": r.get("mapping_reason") or "",
                    "rule_category_suggested": r.get("rule_category_suggested") or "",
                    "rule_confidence": r.get("rule_confidence") or "",
                    "mcc": r.get("mcc") or "",
                    "mcc_description": r.get("mcc_description") or "",
                    "category_external": r.get("category_external") or "",
                },
            }
            fp = fingerprint(rec)
            if fp in seen:
                continue
            # Assign an id if missing
            if not rec["id"]:
                rec["id"] = f"gold_{len(gold) + len(new_records) + 1:06d}"
            new_records.append(rec)
            seen.add(fp)

    appended = append_to_gold(gold_path, new_records)
    print(f"Appended {appended} records to {gold_path} (input: {in_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
