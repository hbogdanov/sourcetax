#!/usr/bin/env python
"""Profile staging values to drive gradual mapping updates.

Reports:
- top external categories observed in staging_transactions
- top MCC descriptions observed in staging_transactions
- which values are already mapped vs unmapped
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sourcetax import mapping


def _fetch_counts(conn: sqlite3.Connection, column: str, limit: int) -> list[tuple[str, int]]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT TRIM({column}) AS value, COUNT(*) AS c
        FROM staging_transactions
        WHERE {column} IS NOT NULL AND TRIM({column}) <> ''
        GROUP BY TRIM({column})
        ORDER BY c DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = []
    for value, c in cur.fetchall():
        rows.append((str(value), int(c)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    db_path = Path(args.staging_db)
    if not db_path.exists():
        print(f"Staging DB not found: {db_path}")
        print("Run importers first, then rerun this profiler.")
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        ext_rows = _fetch_counts(conn, "category_external", args.limit)
        mcc_desc_rows = _fetch_counts(conn, "mcc_description", args.limit)
    finally:
        conn.close()

    print("Top external categories (category_external)")
    if not ext_rows:
        print("  (none)")
    for value, count in ext_rows:
        mapped = mapping.map_external_category(value)
        status = mapped if mapped else "UNMAPPED"
        print(f"  - {value} : {count} -> {status}")

    print("\nTop MCC descriptions (mcc_description)")
    if not mcc_desc_rows:
        print("  (none)")
    for value, count in mcc_desc_rows:
        mapped = mapping.map_mcc_description(value)
        status = mapped if mapped else "UNMAPPED"
        print(f"  - {value} : {count} -> {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


