# Gold Standard Dataset

This directory contains hand-labeled transaction data for evaluating SourceTax categorization, matching, and extraction.

## Files

- `gold_transactions.jsonl` — Canonical transactions with ground truth labels

## Format

Each line is a JSON object conforming to `CanonicalRecord` schema with these guaranteed fields:

```json
{
  "id": "gold_001",
  "source": "bank|receipt|toast|quickbooks",
  "source_record_id": "original_id",
  "transaction_date": "2024-01-15",
  "merchant_raw": "ORIGINAL NAME",
  "merchant_norm": "Normalized Name",
  "amount": 49.99,
  "currency": "USD",
  "direction": "expense|income",
  "payment_method": "debit_card|credit_card|...",
  "category_pred": null,
  "category_final": "Meals and Lodging",
  "confidence": null,
  "matched_transaction_id": "gold_003",
  "match_score": 0.87,
  "evidence_keys": ["date_match", "amount_close"],
  "raw_payload": {}
}
```

## Ground Truth Fields

These fields are gold standard (human-verified):

- **`category_final`** — Correct Schedule C category (e.g., "Meals and Lodging", "Travel", "Office Supplies")
- **`matched_transaction_id`** — If set, this receipt links to that bank transaction (or vice versa)
- **`merchant_norm`** — Human-approved merchant normalization
- **`amount`, `transaction_date`, `direction`** — Verified correct

## Expanding the Gold Set

Use `app_review.py` to interactively build the gold set:

```bash
streamlit run app_review.py
```

Workflow:
1. Review unmatched receipts/transactions
2. Approve or override matches
3. Override category predictions where rules are wrong
4. Save overrides to database

Export reviewed transactions with:
```python
from sourcetax.storage import get_all_records
from pathlib import Path

records = get_all_records(Path("data/store.db"))
# Filter to high-confidence, manually-reviewed records
gold_records = [r for r in records if r.confidence > 0.8 or r.category_final]

# Export to gold_transactions.jsonl
import json
with open("data/gold/gold_transactions.jsonl", "w") as f:
    for r in gold_records:
        f.write(json.dumps(r.__dict__) + "\n")
```

## Evaluation

Run evaluation:
```bash
python tools/eval.py
```

Outputs metrics:
- Categorization accuracy (overall + by category)
- Matching precision/recall/F1
- Extraction accuracy (merchant, date, amount)

## Contribution Guidelines

When adding to gold set:
- Ensure all `category_final` values are valid Schedule C categories
- For matched receipts, verify the `matched_transaction_id` and `match_score` manually
- Include diverse merchant types and edge cases
- Annotate tricky cases in `raw_payload["notes"]`
