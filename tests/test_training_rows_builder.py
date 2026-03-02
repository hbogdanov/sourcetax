from tools.build_training_rows_from_staging import _row_to_training_row


def test_training_row_contains_mapping_reason_and_evidence_keys():
    row = {
        "source": "hf_mitulshah",
        "source_record_id": "x1",
        "txn_ts": "2026-03-02",
        "amount": None,
        "currency": "USD",
        "merchant_raw": "UBER TRIP",
        "description": "Ride charge",
        "mcc": None,
        "mcc_description": None,
        "category_external": "Transportation",
        "raw_payload_json": '{"country":"US"}',
    }
    out = _row_to_training_row(row)  # dict supports row["field"] access
    assert out["category_mapped"] == "Travel"
    assert isinstance(out["mapping_reason"], list)
    assert out["mapping_reason"]
    assert out["evidence_keys"] == out["mapping_reason"]
    assert out["raw_payload"]["mapping_reason"] == out["mapping_reason"]

