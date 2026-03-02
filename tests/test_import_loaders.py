from sourcetax import staging


def test_hf_row_mapping_shape():
    from tools.import_hf_mitulshah import _to_staging_row

    row = _to_staging_row(
        {
            "transaction_description": "McDonald's #1234",
            "category": "Food & Dining",
            "currency": "USD",
            "country": "US",
        },
        idx=7,
    )
    assert row["source"] == "hf_mitulshah"
    assert row["source_record_id"] == "hf_mitulshah_7"
    assert row["description"] == "McDonald's #1234"
    assert row["category_external"] == "Food & Dining"
    assert row["raw_payload_json"]["country"] == "US"


def test_dc_feature_mapping_shape():
    from tools.import_dc_pcard import _feature_to_staging

    row = _feature_to_staging(
        {
            "OBJECTID": 42,
            "VENDOR_NAME": "ACME SUPPLY",
            "TRANSACTION_AMOUNT": 123.45,
            "TRANSACTION_DATE": 1709251200000,  # 2024-03-01 UTC
            "MCC_DESCRIPTION": "STATIONERY STORES",
        },
        idx=0,
    )
    assert row["source"] == "dc_pcard"
    assert row["source_record_id"] == "42"
    assert row["merchant_raw"] == "ACME SUPPLY"
    assert row["amount"] == -123.45
    assert row["mcc_description"] == "STATIONERY STORES"
    assert row["txn_ts"] == "2024-03-01"


def test_bulk_insert_transactions(tmp_path):
    db_path = tmp_path / "staging.db"
    rows = [
        {
            "source": "hf_mitulshah",
            "source_record_id": "a",
            "txn_ts": None,
            "amount": None,
            "currency": "USD",
            "merchant_raw": None,
            "description": "A",
            "mcc": None,
            "mcc_description": None,
            "category_external": "Food & Dining",
            "subcategory_external": None,
            "raw_payload_json": {"country": "US"},
        },
        {
            "source": "dc_pcard",
            "source_record_id": "b",
            "txn_ts": "2026-03-02",
            "amount": -10.0,
            "currency": "USD",
            "merchant_raw": "VENDOR",
            "description": None,
            "mcc": None,
            "mcc_description": "MCC",
            "category_external": None,
            "subcategory_external": None,
            "raw_payload_json": {"country": "US"},
        },
    ]
    inserted = staging.insert_staging_transactions(rows, path=db_path, batch_size=1)
    counts = staging.get_staging_counts(path=db_path)
    assert inserted == 2
    assert counts["staging_transactions"] == 2


def test_sroie_row_mapping_shape():
    from tools.import_receipts_sroie import _to_staging_row

    row = _to_staging_row(
        {
            "id": "sroie_001",
            "words": ["ABC", "TRADING", "TOTAL", "31.00"],
            "entities": {
                "company": "ABC HO TRADING",
                "date": "2018-07-21",
                "address": "KL",
                "total": "31.00",
            },
        },
        idx=0,
        currency="MYR",
    )
    assert row["source"] == "sroie"
    assert row["source_record_id"] == "sroie_001"
    assert row["merchant_raw"] == "ABC HO TRADING"
    assert row["total"] == 31.0
    assert row["receipt_ts"] == "2018-07-21"
    assert "ABC TRADING TOTAL 31.00" in (row["ocr_text"] or "")


def test_bulk_insert_receipts(tmp_path):
    db_path = tmp_path / "staging.db"
    rows = [
        {
            "source": "sroie",
            "source_record_id": "r1",
            "receipt_ts": "2026-03-02",
            "merchant_raw": "A",
            "total": 10.0,
            "tax": None,
            "currency": "MYR",
            "ocr_text": "TOTAL 10",
            "structured_fields_json": {"total": "10"},
            "raw_payload_json": {"dataset": "sroie"},
        },
        {
            "source": "sroie",
            "source_record_id": "r2",
            "receipt_ts": "2026-03-03",
            "merchant_raw": "B",
            "total": 11.0,
            "tax": None,
            "currency": "MYR",
            "ocr_text": "TOTAL 11",
            "structured_fields_json": {"total": "11"},
            "raw_payload_json": {"dataset": "sroie"},
        },
    ]
    inserted = staging.insert_staging_receipts(rows, path=db_path, batch_size=1)
    counts = staging.get_staging_counts(path=db_path)
    assert inserted == 2
    assert counts["staging_receipts"] == 2
