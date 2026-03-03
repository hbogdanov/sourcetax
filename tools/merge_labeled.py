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
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sourcetax import taxonomy
from sourcetax.gold import normalize_label_confidence, normalize_label_notes


def _to_float(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _infer_direction(direction_value, amount_value, category_value):
    d = str(direction_value or "").strip().lower()
    if d in {"expense", "income"}:
        return d
    amt = _to_float(amount_value)
    if amt is not None:
        if amt < 0:
            return "expense"
        if amt > 0:
            return "income"
    c = str(category_value or "").strip().lower()
    if c == "income":
        return "income"
    return "expense"


def _stable_source_record_id(rec: dict) -> str:
    source = str(rec.get("source") or "").strip().lower()
    merchant = str(rec.get("merchant_raw") or "").strip().lower()
    date = str(rec.get("transaction_date") or "").strip()
    amount = _to_float(rec.get("amount"))
    amount_txt = "" if amount is None else f"{abs(amount):.2f}"
    direction = str(rec.get("direction") or "").strip().lower()
    base = f"{source}|{merchant}|{date}|{amount_txt}|{direction}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"hash_{digest}"


def canonicalize_gold_record(rec: dict) -> dict:
    out = dict(rec)
    out["currency"] = str(out.get("currency") or "USD").strip() or "USD"
    out["direction"] = _infer_direction(out.get("direction"), out.get("amount"), out.get("sourcetax_category_v1") or out.get("category_final"))
    amt = _to_float(out.get("amount"))
    out["amount"] = abs(amt) if amt is not None else None
    if not str(out.get("source_record_id") or "").strip():
        out["source_record_id"] = _stable_source_record_id(out)
    return out


def fingerprint(rec: dict) -> str:
    rec = canonicalize_gold_record(rec)
    merchant = rec.get("merchant_raw") or rec.get("merchant") or ""
    description = (rec.get("raw_payload") or {}).get("description", "") if isinstance(rec.get("raw_payload"), dict) else rec.get("description", "")
    amount = rec.get("amount", "")
    direction = rec.get("direction", "")
    return f"{merchant}\t{description}\t{amount}\t{direction}".lower()


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
                "direction": r.get("direction") or "",
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
            rec = canonicalize_gold_record(rec)
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
