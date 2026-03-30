#!/usr/bin/env python
"""Single-entry pipeline runner for SourceTax demos and local product use."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sourcetax import categorization, exporter, ingest, storage


SUPPORTED_SOURCES = ("bank", "toast", "quickbooks", "receipt")


def detect_source_from_headers(fieldnames: Iterable[str]) -> str:
    headers = {str(name or "").strip() for name in fieldnames}
    normalized = {header.lower() for header in headers}

    if {"order_id", "location", "total"}.issubset(normalized):
        return "toast"
    if {"date", "description", "amount"}.issubset(normalized):
        return "bank"
    if {"date", "amount"}.issubset(normalized) and (
        "payee" in normalized or "description" in normalized
    ):
        return "quickbooks"
    if {"merchant", "date", "total"}.issubset(normalized):
        return "receipt"
    raise ValueError(
        "Could not auto-detect input format. Use --source with one of: "
        + ", ".join(SUPPORTED_SOURCES)
    )


def detect_source_from_csv(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return detect_source_from_headers(reader.fieldnames or [])


def tax_class_for_row(direction: str, category: str) -> str:
    if str(direction).strip().lower() == "income":
        return "Business Income"
    if category and category != "Other Expense":
        return "Deductible Expense"
    return "Needs Review"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SourceTax normalization -> classification -> export in one command."
    )
    parser.add_argument("--input", required=True, help="Path to an input CSV file.")
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto",) + SUPPORTED_SOURCES,
        help="Input format. Defaults to auto-detect from CSV headers.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/pipeline_run",
        help="Output directory for exports and the temporary SQLite store.",
    )
    parser.add_argument(
        "--db",
        default="",
        help="Optional SQLite path. Defaults to <out-dir>/pipeline.db.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep an existing DB file instead of starting with a clean run.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=3,
        help="How many enriched rows to print after the pipeline finishes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    source = args.source
    if source == "auto":
        source = detect_source_from_csv(input_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db) if args.db else out_dir / "pipeline.db"
    if db_path.exists() and not args.keep_db:
        db_path.unlink()
    storage.ensure_db(db_path)

    ingested = ingest.ingest_and_store(str(input_path), source, db_path=str(db_path))
    categorized = categorization.categorize_all_records(str(db_path))
    run_id = exporter.generate_run_id("pipeline")
    bundle = exporter.export_accounting_grade_bundle(
        db_path=str(db_path),
        out_dir=str(out_dir),
        pipeline_version="product_demo_v1",
        run_id=run_id,
    )
    quickbooks_csv = exporter.generate_quickbooks_csv(
        out_path=str(out_dir / "quickbooks_import.csv"),
        db_path=str(db_path),
    )

    print(f"SourceTax pipeline complete for {input_path}")
    print(f"Detected source: {source}")
    print(f"Normalized records: {ingested}")
    print(f"Classified records: {categorized}")
    print("Exports:")
    print(f"  Enriched transactions: {bundle['transactions_enriched']}")
    print(f"  GL lines: {bundle['gl_lines']}")
    print(f"  Audit trail: {bundle['audit_trail_jsonl']}")
    print(f"  QuickBooks import: {quickbooks_csv}")

    preview_rows = exporter.fetch_records_full(str(db_path))[: max(int(args.preview_rows), 0)]
    if preview_rows:
        print("\nPreview:")
        for row in preview_rows:
            category = exporter._effective_category(row)
            merchant = row.get("merchant_norm") or row.get("merchant_raw") or ""
            print(f"- Raw: {row.get('merchant_raw') or ''}  {row.get('amount') or ''}")
            print(f"  Merchant: {merchant}")
            print(f"  Category: {category}")
            print(f"  Tax Class: {tax_class_for_row(row.get('direction') or '', category)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
