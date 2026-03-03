import csv
import sqlite3
from pathlib import Path
from typing import Optional

from sourcetax import categorization, exporter, storage, taxonomy


def _insert_min_record(db_path: Path, record_id: str, category_pred: Optional[str] = None) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO canonical_records (
            id, source, source_record_id, transaction_date, merchant_raw, merchant_norm,
            amount, currency, direction, payment_method, category_pred, category_final,
            confidence, matched_transaction_id, match_score, evidence_keys, raw_payload, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            "bank",
            f"src_{record_id}",
            "2026-03-02",
            "TEST MERCHANT",
            "test merchant",
            10.0,
            "USD",
            "expense",
            "card",
            category_pred,
            None,
            0.8,
            None,
            None,
            "[]",
            "{}",
            "[]",
        ),
    )
    conn.commit()
    conn.close()


def test_taxonomy_contract_categories_available():
    categories = taxonomy.load_sourcetax_categories()
    assert "Uncategorized" not in categories
    assert "Meals & Entertainment" in categories
    assert taxonomy.is_valid_category("Rent & Utilities")
    assert taxonomy.normalize_category_name("Meals and Lodging") == "Meals & Entertainment"


def test_save_category_override_rejects_invalid_category(tmp_path: Path):
    db_path = tmp_path / "store.db"
    storage.ensure_db(db_path)
    _insert_min_record(db_path, "rec_1")

    try:
        categorization.save_category_override("rec_1", "Not A Real Category", str(db_path))
        assert False, "Expected ValueError for invalid category"
    except ValueError:
        pass


def test_exports_emit_taxonomy_categories_only(tmp_path: Path):
    db_path = tmp_path / "store.db"
    out_path = tmp_path / "quickbooks.csv"
    storage.ensure_db(db_path)
    _insert_min_record(db_path, "rec_alias", category_pred="Rent")
    _insert_min_record(db_path, "rec_invalid", category_pred="foo bar baz")

    exporter.generate_quickbooks_csv(out_path=str(out_path), db_path=str(db_path))

    with out_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cats = [r["Category"] for r in rows]

    assert "Rent & Utilities" in cats
    assert "Other Expense" in cats
    for c in cats:
        assert taxonomy.is_valid_category(c)
