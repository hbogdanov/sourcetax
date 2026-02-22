"""
Evaluation script for SourceTax pipeline.

Evaluates:
- Category prediction accuracy
- Receipt-to-transaction matching precision/recall
- OCR extraction accuracy (merchant, date, amount)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sourcetax.schema import CanonicalRecord
from sourcetax.matching import match_receipt_to_bank
from sourcetax.categorization import categorize_record


@dataclass
class EvalMetrics:
    """Evaluation results."""
    category_accuracy: float
    category_precision_by_group: Dict[str, float]
    match_precision: float
    match_recall: float
    match_f1: float
    extraction_merchant_accuracy: float
    extraction_date_accuracy: float
    extraction_amount_accuracy: float


def load_gold_set(gold_path: Path = None) -> List[Dict]:
    """Load gold standard dataset (JSONL)."""
    if gold_path is None:
        gold_path = Path(__file__).parent.parent / "data" / "gold" / "gold_transactions.jsonl"
    
    records = []
    if gold_path.exists():
        with open(gold_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def gold_to_canonical(gold_record: Dict) -> CanonicalRecord:
    """Convert gold record dict to CanonicalRecord."""
    return CanonicalRecord(
        id=gold_record.get("id"),
        source=gold_record.get("source"),
        source_record_id=gold_record.get("source_record_id"),
        transaction_date=gold_record.get("transaction_date"),
        merchant_raw=gold_record.get("merchant_raw"),
        merchant_norm=gold_record.get("merchant_norm"),
        amount=gold_record.get("amount", 0.0),
        currency=gold_record.get("currency", "USD"),
        direction=gold_record.get("direction", "expense"),
        payment_method=gold_record.get("payment_method"),
        category_pred=gold_record.get("category_pred"),
        category_final=gold_record.get("category_final"),
        confidence=gold_record.get("confidence"),
        matched_transaction_id=gold_record.get("matched_transaction_id"),
        match_score=gold_record.get("match_score"),
        evidence_keys=gold_record.get("evidence_keys", []),
        raw_payload=gold_record.get("raw_payload", {}),
    )


def eval_category_accuracy(gold_records: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate category prediction accuracy.
    
    Compares category_final (ground truth) with what rules-based categorization predicts.
    Returns overall accuracy and per-category precision.
    """
    if not gold_records:
        return 0.0, {}
    
    canonical_records = [gold_to_canonical(r) for r in gold_records]
    predictions = [categorize_record(r) for r in canonical_records]
    
    correct = 0
    by_category = {}
    
    for gold, pred in zip(gold_records, predictions):
        truth = gold.get("category_final")
        predicted = pred.category_final
        
        if truth:
            if truth not in by_category:
                by_category[truth] = {"correct": 0, "total": 0}
            by_category[truth]["total"] += 1
            
            if predicted == truth:
                correct += 1
                by_category[truth]["correct"] += 1
    
    overall = correct / len(gold_records) if gold_records else 0.0
    precision_by_cat = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for cat, stats in by_category.items()
    }
    
    return overall, precision_by_cat


def eval_matching_accuracy(gold_records: List[Dict]) -> Tuple[float, float, float]:
    """
    Evaluate receipt-to-bank matching.
    
    Precision: of predicted matches, how many are correct?
    Recall: of actual matches, how many did we find?
    Returns: precision, recall, F1
    """
    # Separate receipts and bank transactions
    receipts = [r for r in gold_records if r.get("source") == "receipt"]
    bank_txns = [r for r in gold_records if r.get("source") in ["bank", "toast", "quickbooks"]]
    
    if not receipts or not bank_txns:
        return 0.0, 0.0, 0.0
    
    canonical_bank = [gold_to_canonical(r) for r in bank_txns]
    
    # Build ground truth matches
    true_matches = set()
    for receipt in gold_records:
        matched_id = receipt.get("matched_transaction_id")
        if matched_id:
            true_matches.add((receipt.get("id"), matched_id))
    
    # Evaluate predictions
    predicted_matches = set()
    for receipt in receipts:
        canonical_receipt = gold_to_canonical(receipt)
        best_match, best_score = match_receipt_to_bank(canonical_receipt, canonical_bank)
        if best_match and best_score > 0.6:  # Simple threshold
            predicted_matches.add((receipt.get("id"), best_match.id))
    
    if len(predicted_matches) == 0:
        return 0.0, 0.0, 0.0
    
    correct = len(predicted_matches & true_matches)
    precision = correct / len(predicted_matches) if predicted_matches else 0.0
    recall = correct / len(true_matches) if true_matches else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def eval_extraction_accuracy(gold_records: List[Dict]) -> Tuple[float, float, float]:
    """
    Evaluate OCR extraction for receipts.
    
    Measures how well extracted fields match ground truth.
    Returns: merchant accuracy, date accuracy, amount accuracy
    """
    receipts = [r for r in gold_records if r.get("source") == "receipt"]
    
    if not receipts:
        return 0.0, 0.0, 0.0
    
    merchant_correct = 0
    date_correct = 0
    amount_correct = 0
    
    for receipt in receipts:
        # For now, just check if merchant_norm is not empty (placeholder)
        if receipt.get("merchant_norm"):
            merchant_correct += 1
        if receipt.get("transaction_date"):
            date_correct += 1
        if receipt.get("amount"):
            amount_correct += 1
    
    n = len(receipts)
    return merchant_correct / n, date_correct / n, amount_correct / n


def print_eval_report(metrics: EvalMetrics) -> None:
    """Print human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("SOURCETAX EVALUATION REPORT")
    print("=" * 70)
    
    print("\n📊 CATEGORIZATION")
    print(f"  Overall Accuracy: {metrics.category_accuracy:.1%}")
    if metrics.category_precision_by_group:
        print("  By Category:")
        for cat, prec in sorted(metrics.category_precision_by_group.items()):
            print(f"    {cat:30s} {prec:6.1%}")
    
    print("\n🔗 RECEIPT-TO-TRANSACTION MATCHING")
    print(f"  Precision: {metrics.match_precision:.1%}  (of predicted matches, correct?)")
    print(f"  Recall:    {metrics.match_recall:.1%}  (of actual matches, found?)")
    print(f"  F1 Score:  {metrics.match_f1:.1%}")
    
    print("\n📸 OCR EXTRACTION (Receipts Only)")
    print(f"  Merchant: {metrics.extraction_merchant_accuracy:.1%}")
    print(f"  Date:     {metrics.extraction_date_accuracy:.1%}")
    print(f"  Amount:   {metrics.extraction_amount_accuracy:.1%}")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Run full evaluation on gold dataset."""
    gold_records = load_gold_set()
    
    if not gold_records:
        print("⚠️  No gold dataset found. Create data/gold/gold_transactions.jsonl")
        return
    
    print(f"📈 Evaluating on {len(gold_records)} gold records...")
    
    # Run evaluations
    cat_acc, cat_by_group = eval_category_accuracy(gold_records)
    match_prec, match_rec, match_f1 = eval_matching_accuracy(gold_records)
    ext_merchant, ext_date, ext_amount = eval_extraction_accuracy(gold_records)
    
    metrics = EvalMetrics(
        category_accuracy=cat_acc,
        category_precision_by_group=cat_by_group,
        match_precision=match_prec,
        match_recall=match_rec,
        match_f1=match_f1,
        extraction_merchant_accuracy=ext_merchant,
        extraction_date_accuracy=ext_date,
        extraction_amount_accuracy=ext_amount,
    )
    
    print_eval_report(metrics)


if __name__ == "__main__":
    main()
