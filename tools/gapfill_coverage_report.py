#!/usr/bin/env python
"""Report synthetic gap-fill category coverage from staging_transactions."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


WEAK_CATEGORIES = [
    "COGS",
    "Payroll & Contractors",
    "Insurance",
    "Taxes & Licenses",
    "Professional Services",
]


def _fetch_counts(conn: sqlite3.Connection) -> List[Tuple[str, int]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(category_external, '') AS category, COUNT(*) AS c
        FROM staging_transactions
        WHERE source = 'synthetic_gapfill'
        GROUP BY COALESCE(category_external, '')
        ORDER BY c DESC, category
        """
    )
    return [(str(cat), int(c)) for cat, c in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--target-per-category", type=int, default=200)
    args = parser.parse_args()

    db_path = Path(args.staging_db)
    if not db_path.exists():
        print(f"Staging DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        counts = _fetch_counts(conn)
    finally:
        conn.close()

    total = sum(c for _, c in counts)
    print("Synthetic Gap-Fill Coverage Report")
    print(f"- staging_db: {db_path}")
    print(f"- total_synthetic_gapfill_rows: {total}")
    print(f"- target_per_category: {args.target_per_category}")

    by_category: Dict[str, int] = {k: v for k, v in counts}
    print("\nObserved categories")
    if not counts:
        print("  (none)")
    for cat, c in counts:
        print(f"  - {cat or '(empty)'}: {c}")

    print("\nWeak-category coverage")
    for cat in WEAK_CATEGORIES:
        c = by_category.get(cat, 0)
        status = "OK" if c >= args.target_per_category else "UNDER"
        print(f"  - {cat}: {c} ({status})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

