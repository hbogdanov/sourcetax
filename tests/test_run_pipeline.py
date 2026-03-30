from __future__ import annotations

import csv
from pathlib import Path

import run_pipeline


def test_detect_source_from_headers():
    assert run_pipeline.detect_source_from_headers(["date", "description", "amount"]) == "bank"
    assert run_pipeline.detect_source_from_headers(["order_id", "location", "total"]) == "toast"


def test_run_pipeline_end_to_end(tmp_path: Path):
    input_path = tmp_path / "bank.csv"
    with input_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["date", "description", "amount", "transaction_type", "balance"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date": "2026-03-01",
                "description": "STARBUCKS #1234 ATLANTA GA",
                "amount": "-5.87",
                "transaction_type": "debit",
                "balance": "1200.00",
            }
        )

    out_dir = tmp_path / "outputs"
    exit_code = run_pipeline.main(["--input", str(input_path), "--out-dir", str(out_dir), "--preview-rows", "1"])
    assert exit_code == 0

    enriched_path = out_dir / "accounting_transactions_enriched.csv"
    assert enriched_path.exists()

    rows = list(csv.DictReader(enriched_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["effective_category"] == "Meals & Entertainment"
