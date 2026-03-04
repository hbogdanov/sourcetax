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
import os
import json
import subprocess
from collections import OrderedDict
from datetime import datetime, timezone
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


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return (result.stdout or "").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _write_manifest(
    *,
    out_path: Path,
    dataset: str,
    revision: str,
    split_stats: Dict[str, Dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = OrderedDict(
        {
            "dataset": dataset,
            "revision": revision or "default",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "tools/import_hf_mitulshah.py",
            "git_commit": _git_commit(),
            "split_stats": split_stats,
        }
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mitulshah/transaction-categorization")
    parser.add_argument(
        "--revision",
        default="",
        help="Optional HF dataset revision/commit/tag for reproducibility.",
    )
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--mirror-dir",
        default="data/external/mitulshah_transaction_categorization",
        help="Local on-disk mirror path for HF dataset.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing HF access token.",
    )
    parser.add_argument(
        "--mirror-only",
        action="store_true",
        help="Download and save dataset locally, then exit (no staging import).",
    )
    parser.add_argument(
        "--from-disk",
        action="store_true",
        help="Load from --mirror-dir instead of fetching from HF.",
    )
    parser.add_argument(
        "--save-to-disk",
        action="store_true",
        help="Save a local mirror to --mirror-dir before importing to staging.",
    )
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
            "datasets package is required. Install with: pip install datasets huggingface_hub\n"
            f"Underlying import error: {exc}"
        )

    token = os.getenv(args.token_env)
    mirror_dir = Path(args.mirror_dir)
    manifest_path = mirror_dir / "MANIFEST.json"

    if args.mirror_only or args.save_to_disk:
        print(f"Loading dataset {args.dataset} for local mirror...")
        ds_dict = load_dataset(args.dataset, token=token, revision=(args.revision or None))
        mirror_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving local mirror to: {mirror_dir}")
        ds_dict.save_to_disk(str(mirror_dir))
        split_stats: Dict[str, Dict[str, Any]] = {}
        for split_name, split_ds in ds_dict.items():
            split_stats[split_name] = {
                "rows": int(len(split_ds)),
                "columns": list(split_ds.column_names),
            }
        _write_manifest(
            out_path=manifest_path,
            dataset=args.dataset,
            revision=args.revision,
            split_stats=split_stats,
        )
        print(f"Wrote manifest: {manifest_path}")
        if args.mirror_only:
            print("Mirror complete. Exiting without staging import (--mirror-only).")
            return 0

    db_path = Path(args.staging_db)
    staging.ensure_staging_db(db_path)

    if args.from_disk:
        if not mirror_dir.exists():
            raise SystemExit(
                f"--from-disk requested but mirror not found at {mirror_dir}. "
                "Run with --save-to-disk or --mirror-only first."
            )
        print(f"Loading local mirror from: {mirror_dir}")
        from datasets import load_from_disk

        ds = load_from_disk(str(mirror_dir))[args.split]
    else:
        print(f"Loading dataset {args.dataset}...")
        ds = load_dataset(
            args.dataset,
            split=args.split,
            streaming=bool(args.streaming),
            token=token,
            revision=(args.revision or None),
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
    if manifest_path.exists():
        print(f"- mirror_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


