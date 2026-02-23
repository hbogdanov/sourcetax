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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_run")


def main():
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
        except Exception as e:
            logger.warning("Eval step failed to execute (%s): %s", " ".join(cmd[1:]), e)

    logger.info("Smoke run complete")


if __name__ == "__main__":
    main()
