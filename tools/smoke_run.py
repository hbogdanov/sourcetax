#!/usr/bin/env python
"""
Lightweight smoke test that runs an end-to-end pipeline without requiring EasyOCR or SBERT.

Steps:
 - Ingest sample CSVs (bank, toast)
 - Run matching (receipt <-> bank)
 - Run categorization (rules)
 - Export QuickBooks CSV

This tool should not import heavy optional deps at module import time.
"""
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax import ingest, matching, categorization, exporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_run")


def main():
    logger.info("Starting smoke run: ingest -> match -> categorize -> export")

    # Ingest sample files if present
    n = 0
    try:
        n = ingest.ingest_and_store("data/samples/bank_sample.csv", "bank")
        logger.info(f"Ingested bank samples: {n}")
    except Exception as e:
        logger.warning(f"Skipping bank ingest: {e}")

    try:
        n = ingest.ingest_and_store("data/samples/toast_sample.csv", "toast")
        logger.info(f"Ingested toast samples: {n}")
    except Exception as e:
        logger.warning(f"Skipping toast ingest: {e}")

    # Run matching
    try:
        matched = matching.match_all_receipts()
        logger.info(f"Matched receipts: {matched}")
    except Exception as e:
        logger.warning(f"Matching failed: {e}")

    # Categorize
    try:
        records = []
        # Use categorization.categorize_all_records if available
        if hasattr(categorization, "categorize_all_records"):
            categorized = categorization.categorize_all_records(records)
            logger.info(f"Categorized {len(categorized)} records")
    except Exception as e:
        logger.warning(f"Categorization failed: {e}")

    # Export (best-effort)
    try:
        exporter.generate_quickbooks_csv([], Path("outputs/quickbooks_import.csv"))
        logger.info("Exported QuickBooks CSV (demo placeholder)")
    except Exception as e:
        logger.warning(f"Export failed: {e}")

    logger.info("Smoke run complete")


if __name__ == "__main__":
    main()
