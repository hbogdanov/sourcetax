#!/usr/bin/env python
"""Report progress and balance quality for the SourceTax gold dataset.

Usage:
  python tools/gold_balance_report.py
  python tools/gold_balance_report.py --target 200 --batch-size 25
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_GOLD = Path("data/gold/gold_transactions.jsonl")
DEFAULT_TAXONOMY = Path("data/taxonomy/sourcetax_v1.json")


SOURCE_ALIASES = {
    "bank": "bank",
    "receipt": "receipt",
    "toast": "pos",
    "pos": "pos",
    "quickbooks": "quickbooks",
    "qb": "quickbooks",
}

CATEGORY_ALIASES = {
    "Meals and Lodging": "Meals & Entertainment",
    "Meals (50% limit)": "Meals & Entertainment",
    "Repairs and Maintenance": "Repairs & Maintenance",
    "Legal and Professional Services": "Professional Services",
    "Office Expense": "Office Supplies",
    "Supplies": "Office Supplies",
    "Car and Truck Expenses": "Vehicle Expenses",
    "Utilities": "Rent & Utilities",
    "Other Expenses": "Other Expense",
}


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_taxonomy_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    names = [str(x.get("name", "")).strip() for x in raw if isinstance(x, dict)]
    return [n for n in names if n and n != "Uncategorized"]


def _normalize_source(value: str) -> str:
    if not value:
        return "unknown"
    return SOURCE_ALIASES.get(value.strip().lower(), value.strip().lower())


def _category_counts(rows: Iterable[dict]) -> Counter:
    c = Counter()
    for row in rows:
        raw = (row.get("category_final") or "").strip()
        cat = CATEGORY_ALIASES.get(raw, raw)
        if cat:
            c[cat] += 1
    return c


def _source_counts(rows: Iterable[dict]) -> Counter:
    c = Counter()
    for row in rows:
        src = _normalize_source(str(row.get("source") or ""))
        c[src] += 1
    return c


def _print_sorted_counts(title: str, counts: Counter, total: int) -> None:
    print(f"\n{title}")
    if not counts:
        print("  (none)")
        return
    for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = (value / total) * 100 if total else 0.0
        print(f"  - {key}: {value} ({pct:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    gold_path = Path(args.gold)
    taxonomy_path = Path(args.taxonomy)

    rows = _read_jsonl(gold_path)
    total = len(rows)
    remaining = max(args.target - total, 0)

    categories = _category_counts(rows)
    sources = _source_counts(rows)
    taxonomy_names = _load_taxonomy_names(taxonomy_path)

    print("SourceTax Gold Balance Report")
    print(f"- Gold file: {gold_path}")
    print(f"- Labeled rows: {total}")
    print(f"- Target: {args.target}")
    print(f"- Remaining: {remaining}")
    print(f"- Suggested next batch size: {args.batch_size}")

    _print_sorted_counts("Category distribution", categories, total)
    _print_sorted_counts("Source distribution", sources, total)

    if taxonomy_names:
        expected_per_category = args.target / len(taxonomy_names)
        print(f"\nTarget balance heuristic: ~{expected_per_category:.1f} per category at target={args.target}")
        missing = [name for name in taxonomy_names if categories.get(name, 0) == 0]
        if missing:
            print("Missing categories in current gold set:")
            for name in missing:
                print(f"  - {name}")
        else:
            print("Missing categories in current gold set: none")

        underfilled = []
        for name in taxonomy_names:
            count = categories.get(name, 0)
            if count < expected_per_category * 0.5:
                underfilled.append((name, count))
        if underfilled:
            print("\nPriority categories for next sampling batch:")
            for name, count in sorted(underfilled, key=lambda x: (x[1], x[0]))[: args.batch_size]:
                print(f"  - {name}: current={count}")

    source_targets = ["bank", "receipt", "pos", "quickbooks"]
    if total > 0:
        expected_per_source = total / len(source_targets)
        weak_sources = [s for s in source_targets if sources.get(s, 0) < expected_per_source * 0.5]
        if weak_sources:
            print("\nUnderrepresented sources:")
            for s in weak_sources:
                print(f"  - {s}: current={sources.get(s, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
