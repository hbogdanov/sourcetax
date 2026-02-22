from sourcetax import ingest, matching, receipts


def test_normalize_to_canonical_bank_record():
    row = {
        "date": "02/21/2026",
        "description": "SQ *STARBUCKS COFFEE 123 SF CA",
        "amount": "-4.50",
        "transaction_type": "card",
    }

    res = ingest.normalize_to_canonical(row, source="bank")

    assert res["transaction_date"] == "2026-02-21"
    assert res["merchant_raw"] == row["description"]
    assert res["merchant_norm"] == "starbucks"
    assert res["amount"] == 4.5
    assert res["direction"] == "expense"


def test_matching_normalize_merchant_uses_shared_rules():
    assert matching.normalize_merchant("AMZN MKTP MERCHANT LLC") == "amazon"
    assert matching.normalize_merchant("SQ *STARBUCKS COFFEE 123 SF CA") == "starbucks"


def test_receipt_parsing_extracts_structured_fields():
    text = "\n".join(
        [
            "COFFEE SHOP",
            "Date: 02/21/2026",
            "Tax 0.40",
            "Tip 1.00",
            "TOTAL $4.50",
        ]
    )

    parsed = receipts.parse_receipt_text(text)

    assert parsed["merchant"] == "COFFEE SHOP"
    assert parsed["date"] == "2026-02-21"
    assert parsed["total"] == 4.5
    assert parsed["tax"] == 0.4
    assert parsed["tip"] == 1.0
