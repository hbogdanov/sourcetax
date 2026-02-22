#!/usr/bin/env python
"""
Phase 3 ML Baseline: End-to-end orchestration.

Runs:
1. Data preparation (split dataset)
2. Train TF-IDF + LogisticRegression
3. Evaluate on test set, compare with rules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax.models import data_prep, train_baseline, evaluate


def main():
    """Run full pipeline."""
    print("\n" + "=" * 80)
    print("SOURCETAX PHASE 3: ML BASELINE PIPELINE")
    print("=" * 80)
    
    # Step 1: Prepare data
    print("\n📦 STEP 1: DATA PREPARATION")
    print("-" * 80)
    
    gold_records = data_prep.load_gold_set()
    if not gold_records:
        print("❌ No gold dataset found.")
        print("   Create data/gold/gold_transactions.jsonl first.")
        return 1
    
    print(f"✅ Loaded {len(gold_records)} gold records")
    
    ml_df = data_prep.prepare_ml_records(gold_records)
    print(f"✅ Prepared {len(ml_df)} ML records with labels")
    
    if len(ml_df) < 10:
        print("\n⚠️  WARNING: Very small dataset (<10 records)")
        print("   Current baseline will not be reliable.")
        print("   Recommended: Expand gold set to 200+ records.")
        print("   Use app_review.py to label more records interactively.")
        print("\n   Continuing anyway for demonstration...\n")
    
    train_df, val_df, test_df = data_prep.split_dataset(ml_df)
    print(f"✅ Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    data_dir = data_prep.save_splits(train_df, val_df, test_df)
    
    # Step 2: Train baseline
    print("\n" + "=" * 80)
    print("🤖 STEP 2: TRAIN BASELINE MODEL")
    print("-" * 80)
    
    pipeline, metrics = train_baseline.train_baseline(train_df, val_df)
    print(f"✅ Trained pipeline on {len(train_df)} records")
    print(f"✅ Vocabulary size: {metrics['vocabulary_size']:,} features")
    
    train_baseline.save_pipeline(pipeline)
    
    # Step 3: Evaluate
    print("\n" + "=" * 80)
    print("📊 STEP 3: EVALUATION & COMPARISON")
    print("-" * 80)
    
    test_df = evaluate.load_test_set()
    merchant_map = evaluate.load_merchant_category_map()
    
    y_pred_rules, metrics_rules = evaluate.evaluate_rules(test_df, merchant_map)
    print(f"✅ Rules engine: {metrics_rules['accuracy']:.1%} accuracy")
    
    pipeline = evaluate.load_pipeline()
    if pipeline:
        y_pred_ml, y_proba_ml, metrics_ml = evaluate.evaluate_ml(test_df, pipeline)
        print(f"✅ ML model: {metrics_ml['accuracy']:.1%} accuracy")
    else:
        metrics_ml = None
        print("⚠️  ML model not available")
    
    # Print results
    evaluate.print_results(
        test_df,
        y_true_rules=test_df["category"].values,
        y_pred_rules=y_pred_rules,
        metrics_rules=metrics_rules,
        y_true_ml=test_df["category"].values,
        y_pred_ml=y_pred_ml if metrics_ml else None,
        metrics_ml=metrics_ml,
    )
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3 BASELINE COMPLETE")
    print("=" * 80)
    print("""
Next steps:

1. 📝 Expand gold dataset to 200+ records
   - Use: streamlit run app_review.py
   - Mark good matches, override wrong categories
   - Export as data/gold/gold_transactions.jsonl

2. 🔄 Re-run baseline with larger dataset
   - Run: python tools/train_ml_baseline.py
   - Should see improved metrics

3. 📈 Analyze results
   - Check which categories are hard
   - Look at confusion matrix
   - Read error analysis

4. 🚀 Next: Add features, try ensemble, hyperparameter tune
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
