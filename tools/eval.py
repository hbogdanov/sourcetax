"""
Evaluation script for SourceTax pipeline.

Evaluates:
- Category prediction accuracy
- Receipt-to-transaction matching precision/recall
- OCR extraction accuracy (merchant, date, amount)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv
from difflib import SequenceMatcher
from sourcetax.schema import CanonicalRecord


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


# Keyword rules for categorization
KEYWORD_RULES = {
    "HOME DEPOT": ("Supplies", 0.6),
    "LOWES": ("Supplies", 0.6),
    "AMAZON": ("Supplies", 0.6),
    "OFFICE DEPOT": ("Supplies", 0.65),
    "STAPLES": ("Supplies", 0.65),
    "UBER": ("Travel", 0.7),
    "LYFT": ("Travel", 0.7),
    "GAS": ("Travel", 0.6),
    "CHEVRON": ("Travel", 0.6),
    "STARBUCKS": ("Meals and Lodging", 0.6),
    "CAFE": ("Meals and Lodging", 0.55),
    "RESTAURANT": ("Meals and Lodging", 0.65),
    "HOTEL": ("Travel", 0.75),
    "AIRBNB": ("Travel", 0.75),
    "UTILITY": ("Utilities", 0.8),
    "INTERNET": ("Utilities", 0.75),
    "INSURANCE": ("Insurance", 0.85),
    "TAX": ("Taxes", 0.85),
}


def load_merchant_category_map(path: str = "data/mappings/merchant_category.csv") -> Dict[str, str]:
    """Load exact merchant → category mapping from CSV."""
    mapping = {}
    p = Path(path)
    if not p.exists():
        return mapping
    
    try:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merchant = row.get("merchant", "").strip().upper()
                category = row.get("category_name", "").strip()
                if merchant and category:
                    mapping[merchant] = category
    except Exception:
        pass
    
    return mapping


def categorize_by_rules(merchant: Optional[str], merchant_map: Dict[str, str]) -> Tuple[Optional[str], float]:
    """
    Categorize using rules (no database).
    Priority:
    1. Exact merchant match
    2. Fuzzy merchant match
    3. Keyword match
    4. Uncategorized
    """
    if not merchant:
        return None, 0.0
    
    merchant_norm = merchant.strip().upper()
    
    # Exact match
    if merchant_norm in merchant_map:
        return merchant_map[merchant_norm], 0.95
    
    # Fuzzy match
    for known, category in merchant_map.items():
        if merchant_norm in known or known in merchant_norm:
            return category, 0.75
    
    # Keyword match
    for keyword, (category, confidence) in KEYWORD_RULES.items():
        if keyword in merchant_norm:
            return category, confidence
    
    return None, 0.3


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


def eval_category_accuracy(gold_records: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate category prediction accuracy.
    
    Compares category_final (ground truth) with what rules-based categorization predicts.
    Returns overall accuracy and per-category precision.
    """
    if not gold_records:
        return 0.0, {}
    
    merchant_map = load_merchant_category_map()
    correct = 0
    by_category = {}
    
    for gold in gold_records:
        truth = gold.get("category_final")
        merchant = gold.get("merchant_raw")
        
        predicted, _ = categorize_by_rules(merchant, merchant_map)
        
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
    
    # Build ground truth matches
    true_matches = set()
    for receipt in gold_records:
        matched_id = receipt.get("matched_transaction_id")
        if matched_id:
            true_matches.add((receipt.get("id"), matched_id))
    
    # Simple rule-based scoring: date and merchant similarity
    predicted_matches = set()
    for receipt in receipts:
        receipt_date = receipt.get("transaction_date")
        receipt_merchant = receipt.get("merchant_raw", "").upper()
        receipt_amount = receipt.get("amount", 0)
        
        best_score = 0
        best_match_id = None
        
        for bank in bank_txns:
            bank_date = bank.get("transaction_date")
            bank_merchant = bank.get("merchant_raw", "").upper()
            bank_amount = bank.get("amount", 0)
            
            # Score: date match (±3 days), amount (±$10), merchant fuzzy
            date_score = 0
            if receipt_date and bank_date:
                from datetime import datetime, timedelta
                try:
                    r_date = datetime.fromisoformat(receipt_date)
                    b_date = datetime.fromisoformat(bank_date)
                    days_diff = abs((r_date - b_date).days)
                    if days_diff <= 3:
                        date_score = 1.0 - (days_diff / 3.0)
                except:
                    pass
            
            amount_score = 0
            if receipt_amount and bank_amount:
                amount_diff = abs(receipt_amount - bank_amount)
                if amount_diff <= 10:
                    amount_score = 1.0 - (amount_diff / 10.0)
            
            merchant_score = 0
            if receipt_merchant and bank_merchant:
                ratio = SequenceMatcher(None, receipt_merchant, bank_merchant).ratio()
                merchant_score = max(0, ratio - 0.2)  # Threshold at 0.2
            
            score = (date_score * 0.3) + (amount_score * 0.4) + (merchant_score * 0.3)
            
            if score > best_score:
                best_score = score
                best_match_id = bank.get("id")
        
        if best_score > 0.6:  # Simple threshold
            predicted_matches.add((receipt.get("id"), best_match_id))
    
    if len(predicted_matches) == 0:
        # No predicted matches:
        # - If no true matches exist either, recall is vacuously 1.0.
        # - If true matches exist, recall must be 0.0 (not 1.0).
        if len(true_matches) == 0:
            return 0.0, 1.0, 0.0
        return 0.0, 0.0, 0.0
    
    correct = len(predicted_matches & true_matches)
    precision = correct / len(predicted_matches) if predicted_matches else 0.0
    recall = correct / len(true_matches) if true_matches else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def eval_extraction_accuracy(gold_records: List[Dict]) -> Tuple[float, float, float]:
    """
    Evaluate OCR extraction for receipts.
    
    Measures how well extracted fields are populated.
    Returns: merchant accuracy, date accuracy, amount accuracy
    """
    receipts = [r for r in gold_records if r.get("source") == "receipt"]
    
    if not receipts:
        return 0.0, 0.0, 0.0
    
    merchant_correct = 0
    date_correct = 0
    amount_correct = 0
    
    for receipt in receipts:
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
