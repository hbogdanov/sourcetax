#!/usr/bin/env python
"""Build a compact local training corpus from mirrored HF Mitul Shah dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="data/external/mitulshah_transaction_categorization",
        help="Path produced by tools/import_hf_mitulshah.py --mirror-only",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out-path",
        default="data/external/mitulshah_corpus_train.parquet",
        help="Parquet output path for fast downstream training.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sample size for faster iteration (0 = full split).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--min-rows", type=int, default=100000)
    parser.add_argument("--min-labels", type=int, default=10)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
        import pandas as pd  # noqa: F401
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Required packages missing. Install with: pip install datasets huggingface_hub pandas pyarrow\n"
            f"Underlying import error: {exc}"
        )

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input mirror not found: {in_dir}")

    ds_dict = load_from_disk(str(in_dir))
    if args.split not in ds_dict:
        raise SystemExit(f"Split '{args.split}' not found in mirror. Available: {list(ds_dict.keys())}")
    split_ds = ds_dict[args.split]

    cols = ["transaction_description", "category", "country", "currency"]
    available_cols = [c for c in cols if c in split_ds.column_names]
    split_ds = split_ds.select_columns(available_cols)
    rename = {"transaction_description": "text", "category": "label"}
    if "transaction_description" not in split_ds.column_names or "category" not in split_ds.column_names:
        raise SystemExit("Input split missing required columns: transaction_description, category")

    total_rows = len(split_ds)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    label_counts: Counter = Counter()
    country_counts: Counter = Counter()
    currency_counts: Counter = Counter()
    written_rows = 0
    removed_rows = 0
    null_rows_after_cleaning = 0
    writer = None

    for start in range(0, total_rows, max(1, int(args.batch_size))):
        stop = min(start + max(1, int(args.batch_size)), total_rows)
        batch = split_ds.select(range(start, stop)).to_pandas()
        batch = batch.rename(columns=rename)
        batch["text"] = batch["text"].astype(str).str.strip()
        batch["label"] = batch["label"].astype(str).str.strip()
        before = len(batch)
        batch = batch[(batch["text"].str.len() > 0) & (batch["label"].str.len() > 0)]
        removed_rows += before - len(batch)

        null_mask = batch[["text", "label"]].isnull().any(axis=1)
        null_rows_after_cleaning += int(null_mask.sum())
        if null_rows_after_cleaning:
            batch = batch[~null_mask]

        if args.sample_size and args.sample_size > 0:
            remaining = int(args.sample_size) - written_rows
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.sample(n=remaining, random_state=args.seed)

        if batch.empty:
            continue

        label_counts.update(batch["label"].tolist())
        if "country" in batch.columns:
            country_counts.update([str(x).strip() for x in batch["country"].fillna("").tolist() if str(x).strip()])
        if "currency" in batch.columns:
            currency_counts.update([str(x).strip() for x in batch["currency"].fillna("").tolist() if str(x).strip()])

        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
        written_rows += len(batch)

    if writer is not None:
        writer.close()

    if written_rows == 0:
        raise SystemExit("No rows written after cleaning.")

    empty_removed_pct = (removed_rows / total_rows) * 100 if total_rows else 0.0
    unique_labels = len(label_counts)

    if written_rows <= int(args.min_rows):
        raise SystemExit(
            f"Integrity check failed: written_rows={written_rows:,} <= min_rows={int(args.min_rows):,}"
        )
    if unique_labels < int(args.min_labels):
        raise SystemExit(
            f"Integrity check failed: unique_labels={unique_labels} < min_labels={int(args.min_labels)}"
        )
    if null_rows_after_cleaning > 0:
        raise SystemExit(
            f"Integrity check failed: {null_rows_after_cleaning} null rows remained after cleaning."
        )

    print("Mitul Shah corpus build complete.")
    print(f"- input_rows: {total_rows:,}")
    print(f"- output_rows: {written_rows:,}")
    print(f"- empty_removed_rows: {removed_rows:,} ({empty_removed_pct:.2f}%)")
    print(f"- unique_labels: {unique_labels}")
    print(f"- top10_labels: {label_counts.most_common(10)}")
    print(f"- top5_country: {country_counts.most_common(5)}")
    print(f"- top5_currency: {currency_counts.most_common(5)}")
    out_path = Path(args.out_path)
    print(f"- parquet_out: {out_path}")

    manifest = OrderedDict(
        {
            "dataset": "mitulshah/transaction-categorization",
            "source_mirror": str(in_dir),
            "split": args.split,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "tools/build_mitulshah_corpus.py",
            "git_commit": _git_commit(),
            "rows_input": int(total_rows),
            "rows_output": int(written_rows),
            "rows_removed_empty": int(removed_rows),
            "empty_removed_pct": round(float(empty_removed_pct), 4),
            "unique_labels": int(unique_labels),
            "top10_labels": label_counts.most_common(10),
            "top5_country": country_counts.most_common(5),
            "top5_currency": currency_counts.most_common(5),
            "columns_output": ["text", "label", "country", "currency"],
            "sample_size": int(args.sample_size),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
        }
    )
    manifest_path = in_dir / "MANIFEST.json"
    mirror_manifest = {}
    if manifest_path.exists():
        try:
            mirror_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            mirror_manifest = {}
    manifest["mirror_manifest"] = mirror_manifest
    corpus_manifest_path = out_path.with_suffix(".manifest.json")
    corpus_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"- manifest_out: {corpus_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
