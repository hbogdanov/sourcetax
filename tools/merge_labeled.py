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


def fingerprint(rec: dict) -> str:
    return f"{rec.get('merchant','')}	{rec.get('description','')}	{rec.get('amount','')}".lower()


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
            rec = {
                "id": r.get("id") or None,
                "merchant": r.get("merchant") or r.get("vendor") or r.get("Vendor") or "",
                "description": r.get("description") or r.get("text") or "",
                "amount": r.get("amount") or r.get("Amount") or None,
                "category": r.get("category") or None,
                "source": r.get("source") or "labeled",
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
