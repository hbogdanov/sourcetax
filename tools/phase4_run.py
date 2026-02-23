#!/usr/bin/env python
"""Phase 4 runner: accounting-grade exports + reconciliation reports (+ optional adapters)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax import exporter, reconciliation
from sourcetax.adapters import MockQuickBooksApi, QboLikeExportAdapter


def _read_csv_rows(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/store.db")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--low-conf-threshold", type=float, default=0.7)
    parser.add_argument("--mock-qbo", action="store_true")
    args = parser.parse_args()

    run_id = exporter.generate_run_id("phase4")
    bundle = exporter.export_accounting_grade_bundle(
        db_path=args.db,
        out_dir=args.out_dir,
        pipeline_version=exporter.PIPELINE_VERSION,
        run_id=run_id,
    )
    recon = reconciliation.export_reconciliation_reports(
        db_path=args.db,
        out_dir=str(Path(args.out_dir) / "reconciliation"),
        low_conf_threshold=args.low_conf_threshold,
    )

    # Phase 4c minimal adapter exports.
    qbo_like_path = str(Path(args.out_dir) / "qbo_like_transactions.csv")
    qbo_like = QboLikeExportAdapter().export(_read_csv_rows(bundle["transactions_enriched"]), qbo_like_path)

    outputs = {**bundle, **recon, "qbo_like_export": qbo_like}

    if args.mock_qbo:
        mock_path = MockQuickBooksApi(root=str(Path(args.out_dir) / "mock_qbo_api")).push_transactions(
            _read_csv_rows(bundle["transactions_enriched"]),
            batch_name=f"phase4_{run_id}",
        )
        outputs["mock_qbo_api_payload"] = mock_path

    print("Phase 4 outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
