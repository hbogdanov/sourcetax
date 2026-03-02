#!/usr/bin/env python
"""Import Hugging Face Mitul Shah transaction corpus into staging_transactions.

Target mapping:
- source='hf_mitulshah'
- description <- transaction_description
- category_external <- category
- currency/country kept in raw_payload_json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from sourcetax import staging


def _to_staging_row(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    description = str(example.get("transaction_description") or "").strip()
    category = str(example.get("category") or "").strip() or None
    currency = str(example.get("currency") or "").strip() or "USD"
    country = str(example.get("country") or "").strip() or None
    return {
        "source": "hf_mitulshah",
        "source_record_id": f"hf_mitulshah_{idx}",
        "txn_ts": None,
        "amount": None,
        "currency": currency,
        "merchant_raw": None,
        "description": description or None,
        "mcc": None,
        "mcc_description": None,
        "category_external": category,
        "subcategory_external": None,
        "raw_payload_json": {
            "dataset": "mitulshah/transaction-categorization",
            "country": country,
            "currency": currency,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use HF streaming API to avoid full local download (default: true).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming and load split in-memory.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "datasets package is required. Install with: pip install datasets\n"
            f"Underlying import error: {exc}"
        )

    db_path = Path(args.staging_db)
    staging.ensure_staging_db(db_path)

    print("Loading dataset mitulshah/transaction-categorization...")
    ds = load_dataset(
        "mitulshah/transaction-categorization",
        split=args.split,
        streaming=bool(args.streaming),
    )

    batch: List[Dict[str, Any]] = []
    inserted_total = 0
    for idx, ex in enumerate(ds):
        if idx >= args.max_rows:
            break
        batch.append(_to_staging_row(ex, idx))
        if len(batch) >= args.batch_size:
            inserted_total += staging.insert_staging_transactions(
                batch, path=db_path, batch_size=args.batch_size
            )
            batch = []
            if inserted_total % (args.batch_size * 10) == 0:
                print(f"Inserted {inserted_total} rows...")

    if batch:
        inserted_total += staging.insert_staging_transactions(
            batch, path=db_path, batch_size=args.batch_size
        )

    counts = staging.get_staging_counts(db_path)
    print("Import complete.")
    print(f"- inserted_rows: {inserted_total}")
    print(f"- staging_transactions_total: {counts['staging_transactions']}")
    print(f"- staging_receipts_total: {counts['staging_receipts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

