from pathlib import Path

from sourcetax import mapping, staging


def test_mapping_precedence_keyword_overrides_mcc_and_external():
    category = mapping.resolve_category_with_precedence(
        merchant_raw="GUSTO PAYROLL*FEB",
        description="monthly payroll run",
        mcc="5812",
        external_category="Food & Dining",
    )
    assert category == "Payroll & Contractors"


def test_mapping_precedence_mcc_then_external_then_fallback():
    mcc_first = mapping.resolve_category_with_precedence(
        merchant_raw="UNKNOWN MERCHANT",
        mcc="5812",
        external_category="Utilities & Services",
    )
    assert mcc_first == "Meals & Entertainment"

    external_second = mapping.resolve_category_with_precedence(
        merchant_raw="UNKNOWN MERCHANT",
        mcc=None,
        external_category="Utilities & Services",
    )
    assert external_second == "Rent & Utilities"

    fallback = mapping.resolve_category_with_precedence(
        merchant_raw="UNKNOWN MERCHANT",
        mcc=None,
        external_category=None,
    )
    assert fallback == "Other Expense"


def test_staging_create_insert_and_count(tmp_path: Path):
    db_path = tmp_path / "staging.db"
    staging.ensure_staging_db(db_path)

    staging.insert_staging_transaction(
        {
            "source": "hf_mitulshah",
            "source_record_id": "txn_001",
            "txn_ts": "2026-03-02",
            "amount": -42.19,
            "currency": "USD",
            "merchant_raw": "STARBUCKS",
            "description": "STARBUCKS STORE 123",
            "mcc": "5814",
            "category_external": "Food & Dining",
            "raw_payload_json": {"country": "US"},
        },
        path=db_path,
    )

    staging.insert_staging_receipt(
        {
            "source": "sroie",
            "source_record_id": "receipt_001",
            "receipt_ts": "2026-03-02",
            "merchant_raw": "ABC CAFE",
            "total": 42.19,
            "tax": 3.50,
            "currency": "USD",
            "ocr_text": "TOTAL 42.19",
            "structured_fields_json": {"total": "42.19"},
        },
        path=db_path,
    )

    counts = staging.get_staging_counts(path=db_path)
    assert counts["staging_transactions"] == 1
    assert counts["staging_receipts"] == 1

