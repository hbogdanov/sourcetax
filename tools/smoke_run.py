#!/usr/bin/env python
"""
Lightweight smoke test that runs an end-to-end pipeline without requiring EasyOCR or SBERT.

Steps:
 - Ingest sample CSVs (bank, toast)
 - Run matching (receipt <-> bank)
 - Run categorization (rules)
 - Export QuickBooks CSV
 - Run evaluation scripts (best effort)

This tool should not import heavy optional deps at module import time.
"""
import logging
import os
from pathlib import Path
import sys
import sqlite3
import subprocess
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from sourcetax import ingest, matching, categorization, exporter, storage, receipts
from sourcetax.normalization import normalize_merchant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_run")


def _assert_smoke_outputs(
    db_path: Path,
    out_csv: Path,
    strict_min_records: int = 5,
    strict_min_matches: int = 1,
) -> None:
    if not db_path.exists():
        raise AssertionError(f"Smoke DB missing: {db_path}")
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        raise AssertionError(f"Smoke export missing/empty: {out_csv}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM canonical_records")
    total_records = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM canonical_records WHERE source = 'receipt' AND matched_transaction_id IS NOT NULL")
    matched_receipts = int(cur.fetchone()[0])
    conn.close()

    if total_records < strict_min_records:
        raise AssertionError(f"Expected at least {strict_min_records} records, found {total_records}")
    if matched_receipts < strict_min_matches:
        raise AssertionError(f"Expected at least {strict_min_matches} matched receipts, found {matched_receipts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Fail loudly if core smoke outputs are missing")
    args = parser.parse_args()

    logger.info("Starting smoke run: ingest -> match -> categorize -> export")
    db_path = Path("tmp/smoke_store.db")
    out_csv = Path("outputs/smoke_quickbooks_import.csv")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()
    storage.ensure_db(db_path)

    # Optional lightweight "OCR parse" stage from sample text (no OCR engine needed).
    try:
        sample_receipt_text = Path("data/samples/receipt_sample.txt")
        if sample_receipt_text.exists():
            parsed = receipts.parse_receipt_text(sample_receipt_text.read_text(encoding="utf-8"))
            logger.info(
                "Parsed receipt sample text (optional OCR parse stage): merchant=%s date=%s total=%s",
                parsed.get("merchant"),
                parsed.get("date"),
                parsed.get("total"),
            )
    except Exception as e:
        logger.warning(f"Receipt parse sample skipped: {e}")

    # Ingest sample files if present
    n = 0
    try:
        n = ingest.ingest_and_store("data/samples/bank_sample.csv", "bank", db_path=str(db_path))
        logger.info(f"Ingested bank samples: {n}")
    except Exception as e:
        logger.warning(f"Skipping bank ingest: {e}")

    try:
        n = ingest.ingest_and_store("data/samples/toast_sample.csv", "toast", db_path=str(db_path))
        logger.info(f"Ingested toast samples: {n}")
    except Exception as e:
        logger.warning(f"Skipping toast ingest: {e}")

    # Insert a deterministic synthetic receipt so smoke demos always exercise matching.
    # This targets the Toast sample row id=1001 ("Main", 2026-02-20, 4.50), because
    # bank sample rows may not have stable IDs in the current fixture.
    try:
        synthetic_receipt = {
            "id": "smoke_receipt_1",
            "source": "receipt",
            "source_record_id": "smoke_receipt_1",
            "transaction_date": "2026-02-20",
            "merchant_raw": "Main",
            "merchant_norm": normalize_merchant("Main"),
            "amount": 4.50,
            "currency": "USD",
            "direction": "expense",
            "payment_method": "card",
            "category_pred": None,
            "category_final": None,
            "confidence": None,
            "matched_transaction_id": None,
            "match_score": None,
            "evidence_keys": ["smoke_synthetic_receipt"],
            "raw_payload": {"note": "Synthetic receipt inserted for deterministic smoke matching"},
            "tags": ["smoke", "synthetic"],
        }
        storage.insert_record(synthetic_receipt, path=db_path)
        logger.info("Inserted synthetic smoke receipt for deterministic matching demo")
    except Exception as e:
        logger.warning(f"Synthetic receipt insert failed: {e}")

    # Run matching
    try:
        matched = matching.match_all_receipts(str(db_path))
        logger.info(f"Matched receipts: {matched}")
    except Exception as e:
        logger.warning(f"Matching failed: {e}")

    # Categorize
    try:
        categorized = categorization.categorize_all_records(str(db_path))
        logger.info(f"Categorized {categorized} records")
    except Exception as e:
        logger.warning(f"Categorization failed: {e}")

    # Export (best-effort)
    try:
        out_path = exporter.generate_quickbooks_csv(str(out_csv), str(db_path))
        logger.info(f"Exported QuickBooks CSV: {out_path}")
    except Exception as e:
        logger.warning(f"Export failed: {e}")

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM canonical_records")
        total = cur.fetchone()[0]
        conn.close()
        logger.info(f"Smoke DB records: {total}")
    except Exception as e:
        logger.warning(f"DB verification failed: {e}")

    # Best-effort evaluations. These should not fail the smoke run.
    eval_exit_codes = []
    for cmd in (
        [sys.executable, "tools/eval.py"],
        [sys.executable, "tools/phase3_benchmark.py", "--allow-small"],
    ):
        try:
            logger.info("Running evaluation step: %s", " ".join(cmd[1:]))
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                check=False,
                env=env,
            )
            logger.info("Eval step exited with code %s", result.returncode)
            eval_exit_codes.append((cmd[1], result.returncode))
        except Exception as e:
            logger.warning("Eval step failed to execute (%s): %s", " ".join(cmd[1:]), e)
            eval_exit_codes.append((cmd[1], -1))

    if args.strict:
        _assert_smoke_outputs(db_path, out_csv)
        benchmark_report = Path("reports/phase3_eval.md")
        if not benchmark_report.exists() or benchmark_report.stat().st_size == 0:
            raise AssertionError("Benchmark report missing/empty: reports/phase3_eval.md")
        for step_name, code in eval_exit_codes:
            if code != 0:
                raise AssertionError(f"Eval step failed in strict mode: {step_name} (exit={code})")
        logger.info("Strict smoke assertions passed")

    logger.info("Smoke run complete")


if __name__ == "__main__":
    main()
