"""
Comprehensive evaluation for Phase 3 baseline.

Evaluates:
- ML model on test set
- Rules engine on same test set
- Compares side-by-side
- Confusion matrix
- Error analysis
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from difflib import SequenceMatcher
import csv


# Keywords for rules engine (from categorization.py)
KEYWORD_RULES = {
    "HOME DEPOT": "Repairs and Maintenance",
    "LOWES": "Repairs and Maintenance",
    "AMAZON": "Office Supplies",
    "OFFICE DEPOT": "Office Supplies",
    "STAPLES": "Office Supplies",
    "UBER": "Travel",
    "LYFT": "Travel",
    "GAS": "Travel",
    "CHEVRON": "Travel",
    "STARBUCKS": "Meals and Lodging",
    "CAFE": "Meals and Lodging",
    "RESTAURANT": "Meals and Lodging",
    "HOTEL": "Travel",
    "AIRBNB": "Travel",
    "UTILITY": "Utilities",
    "INTERNET": "Utilities",
    "INSURANCE": "Insurance",
    "TAX": "Taxes",
}


def load_merchant_category_map(path: str = "data/mappings/merchant_category.csv") -> Dict[str, str]:
    """Load merchant mapping from CSV."""
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


def apply_rules(text: str, merchant_map: Dict[str, str]) -> Tuple[str, float]:
    """
    Apply rules-based categorization.
    
    Priority:
    1. Exact merchant match
    2. Keyword match
    3. Uncategorized
    """
    if not text:
        return "Uncategorized", 0.0
    
    merchant_norm = text.split()[0].upper() if text else ""
    
    # Exact match
    if merchant_norm in merchant_map:
        return merchant_map[merchant_norm], 0.95
    
    # Keyword match
    for keyword, category in KEYWORD_RULES.items():
        if keyword in text.upper():
            return category, 0.6
    
    return "Uncategorized", 0.3


def load_test_set(data_dir: Path = None) -> pd.DataFrame:
    """Load test split."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "ml"
    
    return pd.read_csv(data_dir / "ml_test.csv")


def load_pipeline(pipeline_path: Path = None) -> object:
    """Load trained pipeline."""
    if pipeline_path is None:
        pipeline_path = Path(__file__).parent.parent.parent.parent / "data" / "ml" / "baseline_pipeline.pkl"
    
    if not pipeline_path.exists():
        return None
    
    with open(pipeline_path, "rb") as f:
        return pickle.load(f)


def evaluate_ml(
    test_df: pd.DataFrame,
    pipeline: object,
) -> Tuple[list, list, dict]:
    """Evaluate ML model on test set."""
    if pipeline is None:
        print("⚠️  No trained pipeline found. Run train_baseline.py first.")
        return None, None, {}
    
    y_true = test_df["category"].values
    y_pred = pipeline.predict(test_df["text"])
    y_proba = pipeline.predict_proba(test_df["text"])
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    classes = pipeline.named_steps["classifier"].classes_
    
    metrics = {
        "accuracy": accuracy,
        "macro_f1": f1.mean(),
        "weighted_f1": np.average(f1, weights=support),
        "classes": classes,
        "precision_per_class": dict(zip(classes, precision)),
        "recall_per_class": dict(zip(classes, recall)),
        "f1_per_class": dict(zip(classes, f1)),
        "support_per_class": dict(zip(classes, support)),
    }
    
    return y_pred, y_proba, metrics


def evaluate_rules(
    test_df: pd.DataFrame,
    merchant_map: Dict[str, str],
) -> Tuple[list, dict]:
    """Evaluate rules engine on test set."""
    y_true = test_df["category"].values
    y_pred = []
    
    for text in test_df["text"]:
        pred, _ = apply_rules(text, merchant_map)
        y_pred.append(pred)
    
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    classes = np.unique(y_true)
    
    metrics = {
        "accuracy": accuracy,
        "macro_f1": f1.mean() if len(f1) > 0 else 0.0,
        "weighted_f1": np.average(f1, weights=support) if len(f1) > 0 else 0.0,
        "classes": classes,
        "precision_per_class": dict(zip(classes, precision)),
        "recall_per_class": dict(zip(classes, recall)),
        "f1_per_class": dict(zip(classes, f1)),
        "support_per_class": dict(zip(classes, support)),
    }
    
    return y_pred, metrics


def analyze_errors(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str = "Model",
) -> List[Dict]:
    """
    Analyze misclassifications.
    
    Returns top 20 errors with context.
    """
    errors = []
    
    for i, (text, true, pred) in enumerate(zip(test_df["text"], y_true, y_pred)):
        if true != pred:
            errors.append({
                "text": text,
                "true": true,
                "pred": pred,
            })
    
    # Sort by count (most common error first)
    error_summary = {}
    for err in errors:
        key = f"{err['true']} → {err['pred']}"
        error_summary[key] = error_summary.get(key, 0) + 1
    
    return errors, error_summary


def print_results(
    test_df: pd.DataFrame,
    y_true_rules: np.ndarray = None,
    y_pred_rules: np.ndarray = None,
    metrics_rules: dict = None,
    y_true_ml: np.ndarray = None,
    y_pred_ml: np.ndarray = None,
    metrics_ml: dict = None,
) -> None:
    """Print evaluation results in human-readable format."""
    print("\n" + "=" * 80)
    print("SOURCETAX PHASE 3 BASELINE EVALUATION")
    print("=" * 80)
    
    print(f"\nTest Set: {len(test_df)} records")
    print(f"Categories: {', '.join(test_df['category'].unique())}")
    
    if metrics_rules:
        print("\n" + "-" * 40)
        print("RULES ENGINE")
        print("-" * 40)
        print(f"Accuracy:    {metrics_rules['accuracy']:.1%}")
        print(f"Macro F1:    {metrics_rules['macro_f1']:.1%}")
        print(f"Weighted F1: {metrics_rules['weighted_f1']:.1%}")
        
        print("\nPer-Category Performance:")
        for cls in metrics_rules.get("classes", []):
            p = metrics_rules["precision_per_class"].get(cls, 0)
            r = metrics_rules["recall_per_class"].get(cls, 0)
            f = metrics_rules["f1_per_class"].get(cls, 0)
            s = metrics_rules["support_per_class"].get(cls, 0)
            print(f"  {cls:30s} P:{p:.1%} R:{r:.1%} F1:{f:.1%} (n={int(s)})")
    
    if metrics_ml:
        print("\n" + "-" * 40)
        print("ML MODEL (TF-IDF + LogisticRegression)")
        print("-" * 40)
        print(f"Accuracy:    {metrics_ml['accuracy']:.1%}")
        print(f"Macro F1:    {metrics_ml['macro_f1']:.1%}")
        print(f"Weighted F1: {metrics_ml['weighted_f1']:.1%}")
        
        print("\nPer-Category Performance:")
        for cls in metrics_ml.get("classes", []):
            p = metrics_ml["precision_per_class"].get(cls, 0)
            r = metrics_ml["recall_per_class"].get(cls, 0)
            f = metrics_ml["f1_per_class"].get(cls, 0)
            s = metrics_ml["support_per_class"].get(cls, 0)
            print(f"  {cls:30s} P:{p:.1%} R:{r:.1%} F1:{f:.1%} (n={int(s)})")
    
    # Comparison
    if metrics_rules and metrics_ml:
        print("\n" + "-" * 40)
        print("COMPARISON")
        print("-" * 40)
        print(f"{'Method':20s} {'Accuracy':>15s} {'Macro F1':>15s} {'Weighted F1':>15s}")
        print("-" * 65)
        print(f"{'Rules':20s} {metrics_rules['accuracy']:>14.1%} {metrics_rules['macro_f1']:>14.1%} {metrics_rules['weighted_f1']:>14.1%}")
        print(f"{'ML':20s} {metrics_ml['accuracy']:>14.1%} {metrics_ml['macro_f1']:>14.1%} {metrics_ml['weighted_f1']:>14.1%}")
        
        ml_advantage = metrics_ml['accuracy'] - metrics_rules['accuracy']
        direction = "📈" if ml_advantage > 0 else "📉"
        print(f"\nML Advantage: {direction} {ml_advantage:+.1%}")
    
    # Error analysis
    if y_pred_ml is not None and y_true_ml is not None:
        errors, error_summary = analyze_errors(test_df, y_true_ml, y_pred_ml)
        
        if errors:
            print("\n" + "-" * 40)
            print("TOP MISCLASSIFICATIONS (ML)")
            print("-" * 40)
            for i, (error_type, count) in enumerate(sorted(error_summary.items(), key=lambda x: -x[1])[:5]):
                print(f"  {error_type:40s} ({count}x)")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Run complete evaluation."""
    print("📖 Loading test set...")
    test_df = load_test_set()
    print(f"   {len(test_df)} records")
    
    print("🔧 Loading merchant mapping...")
    merchant_map = load_merchant_category_map()
    
    print("\n" + "=" * 80)
    print("EVALUATING RULES ENGINE")
    print("=" * 80)
    y_pred_rules, metrics_rules = evaluate_rules(test_df, merchant_map)
    
    print("\n" + "=" * 80)
    print("EVALUATING ML MODEL")
    print("=" * 80)
    y_pred_ml, y_proba_ml, metrics_ml = evaluate_ml(test_df, load_pipeline())
    
    # Print results
    print_results(
        test_df,
        y_true_rules=test_df["category"].values,
        y_pred_rules=y_pred_rules,
        metrics_rules=metrics_rules,
        y_true_ml=test_df["category"].values,
        y_pred_ml=y_pred_ml if y_pred_ml is not None else None,
        metrics_ml=metrics_ml if metrics_ml else None,
    )


if __name__ == "__main__":
    main()
