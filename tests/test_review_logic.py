from sourcetax.review_logic import (
    build_issue_flags,
    confidence_value,
    merchant_ambiguity_reason,
    primary_issue_type,
)


def test_confidence_value_clamps_and_parses():
    assert confidence_value("0.9") == 0.9
    assert confidence_value(None) == 0.0
    assert confidence_value(4) == 1.0
    assert confidence_value(-1) == 0.0


def test_build_issue_flags_prefers_low_confidence_without_receipts():
    record = {
        "transaction_date": "2026-02-01",
        "merchant_raw": "AMAZON BUSINESS",
        "merchant_norm": "amazon",
        "amount": 42.0,
        "source": "bank",
        "confidence": 0.3,
        "category_pred": "Other Expense",
        "category_final": None,
        "matched_transaction_id": None,
        "raw_payload": {
            "rule_category": "Other Expense",
            "rule_confidence": 0.3,
            "ml_prediction": "Meals & Entertainment",
            "ml_confidence": 0.21,
        },
    }
    flags = build_issue_flags(record, has_receipts=False, is_conflict=True)
    assert "low_confidence" in flags
    assert "conflict" in flags
    assert "ambiguous_merchant" in flags
    assert "unmatched_bank_txn" not in flags
    assert primary_issue_type(flags) == "low_confidence"


def test_merchant_ambiguity_reason_detects_processor_and_fallback_cases():
    processor_record = {
        "merchant_raw": "STRIPE TRANSFER",
        "merchant_norm": "transfer",
        "category_pred": "Other Expense",
        "raw_payload": {},
    }
    fallback_record = {
        "merchant_raw": "SWEET HUT BAKERY",
        "merchant_norm": "sweet hut bakery",
        "category_pred": "Other Expense",
        "raw_payload": {
            "rule_category": "Other Expense",
            "rule_confidence": 0.3,
            "ml_prediction": "Meals & Entertainment",
            "ml_confidence": 0.20,
        },
    }
    assert merchant_ambiguity_reason(processor_record) != ""
    assert merchant_ambiguity_reason(fallback_record) != ""
