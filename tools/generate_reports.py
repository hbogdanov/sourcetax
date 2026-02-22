"""Ingest sample data (fresh start) and generate QuickBooks CSV + Schedule C totals."""

from pathlib import Path
from sourcetax.ingest import ingest_and_store
from sourcetax import exporter


def ensure_data():
    """Ingest sample data from multiple sources into a fresh DB."""
    db_path = Path("data/store.db")
    # Start fresh: delete existing DB for deterministic runs
    if db_path.exists():
        db_path.unlink()

    # Ingest samples in order
    n = ingest_and_store("data/samples/toast_sample.csv", "toast", db_path=str(db_path))
    print("Ingested toast:", n)
    n = ingest_and_store("data/samples/bank_sample.csv", "bank", db_path=str(db_path))
    print("Ingested bank:", n)
    n = ingest_and_store("data/samples/quickbooks_sample.csv", "quickbooks", db_path=str(db_path))
    print("Ingested quickbooks:", n)


def generate():
    """Generate exports from canonical DB."""
    print("Generating QuickBooks CSV...")
    qb = exporter.generate_quickbooks_csv(
        out_path="outputs/quickbooks_import.csv", db_path="data/store.db"
    )
    print("QuickBooks CSV:", qb)
    
    print("Computing Schedule C totals...")
    totals, counts = exporter.compute_schedule_c_totals(db_path="data/store.db")
    sc = exporter.write_schedule_c_csv(totals, count_by_category=counts, out_path="outputs/schedule_c_totals.csv")
    print("Schedule C totals:", sc)
    
    print("Exporting audit pack...")
    audit = exporter.export_audit_pack(db_path="data/store.db")
    print("Audit pack:", audit)
    
    print("\nExport Metrics:")
    metrics = exporter.export_metrics(db_path="data/store.db")
    print(f"  Total records: {metrics['total_records']}")
    print(f"  Total expenses: {metrics['total_expenses']} (${metrics['total_amount']:.2f})")
    print(f"  Receipts: {metrics['total_receipts']} (matched: {metrics['matched_receipts']}, unmatched: {metrics['unmatched_receipts']})")
    print(f"  Match rate: {metrics['match_rate']:.1%}")
    print(f"  Needs review: {metrics['needs_review']}")


if __name__ == "__main__":
    ensure_data()
    generate()
