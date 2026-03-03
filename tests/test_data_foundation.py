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

    category2, reasons = mapping.resolve_category_with_reason(
        merchant_raw="GUSTO PAYROLL*FEB",
        description="monthly payroll run",
        mcc="5812",
        external_category="Food & Dining",
    )
    assert category2 == "Payroll & Contractors"
    assert reasons and reasons[0].startswith("keyword:")


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

    c_mcc, r_mcc = mapping.resolve_category_with_reason(
        merchant_raw="UNKNOWN MERCHANT",
        mcc="5812",
        external_category="Utilities & Services",
    )
    assert c_mcc == "Meals & Entertainment"
    assert r_mcc == ["mcc:5812"]

    c_ext, r_ext = mapping.resolve_category_with_reason(
        merchant_raw="UNKNOWN MERCHANT",
        mcc=None,
        external_category="Utilities & Services",
    )
    assert c_ext == "Rent & Utilities"
    assert r_ext == ["external:Utilities & Services"]

    c_fb, r_fb = mapping.resolve_category_with_reason(
        merchant_raw="UNKNOWN MERCHANT",
        mcc=None,
        external_category=None,
    )
    assert c_fb == "Other Expense"
    assert r_fb == ["fallback:Other Expense"]


def test_mapping_supports_mcc_description_mapping():
    category = mapping.resolve_category_with_precedence(
        merchant_raw="UNKNOWN MERCHANT",
        mcc=None,
        mcc_description="RESTAURANTS",
        external_category=None,
    )
    assert category == "Meals & Entertainment"


def test_transportation_external_split_with_keyword_override():
    # Default external mapping now favors Vehicle Expenses.
    c_default, r_default = mapping.resolve_category_with_reason(
        merchant_raw="UNKNOWN MERCHANT",
        external_category="Transportation",
    )
    assert c_default == "Vehicle Expenses"
    assert r_default == ["external:Transportation"]

    # Travel keywords should override external mapping.
    c_travel, r_travel = mapping.resolve_category_with_reason(
        merchant_raw="DELTA AIR LINES",
        external_category="Transportation",
    )
    assert c_travel == "Travel"
    assert r_travel[0].startswith("keyword:")

    # Vehicle keywords should still resolve to Vehicle Expenses.
    c_vehicle, r_vehicle = mapping.resolve_category_with_reason(
        merchant_raw="SHELL GAS STATION",
        external_category="Transportation",
    )
    assert c_vehicle == "Vehicle Expenses"
    assert r_vehicle[0].startswith("keyword:")


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
