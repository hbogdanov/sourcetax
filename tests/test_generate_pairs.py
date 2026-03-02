import random

from tools.generate_pairs import _make_positive_pair, _sample_offset_days


def test_offset_sampler_returns_only_expected_values():
    rng = random.Random(123)
    values = {_sample_offset_days(rng) for _ in range(200)}
    assert values.issubset({0, 1, 2})
    assert values == {0, 1, 2}


def test_make_positive_pair_has_group_id_and_linked_ids():
    rng = random.Random(42)
    base = {
        "source": "sroie",
        "source_record_id": "receipt_001",
        "receipt_ts": "2026-03-02",
        "merchant_raw": "ABC CAFE",
        "total": 31.0,
        "tax": 2.0,
        "currency": "MYR",
        "ocr_text": "TOTAL 31.00",
        "structured_fields_json": {},
        "raw_payload_json": {},
    }
    receipt_row, bank_row, mini = _make_positive_pair(base_receipt=base, pair_idx=1, rng=rng)
    assert mini["group_id"] == "pair_00001"
    assert mini["should_match"] is True
    assert receipt_row["raw_payload_json"]["group_id"] == "pair_00001"
    assert bank_row["raw_payload_json"]["group_id"] == "pair_00001"
    assert receipt_row["source_record_id"] == "pair_00001_r"
    assert bank_row["source_record_id"] == "pair_00001_b"
    assert bank_row["amount"] < 0

