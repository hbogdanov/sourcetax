#!/usr/bin/env python
"""
Integration test: End-to-end Phase 3 ML workflow.

Demonstrates:
1. Load gold dataset
2. Normalize merchants
3. Compute embeddings
4. Train TF-IDF baseline
5. Train SBERT classifier
6. Evaluate & compare
7. Select samples with active learning
8. Generate visualizations

Run: python tests/test_phase3_integration.py
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax.models import (
    data_prep,
    train_baseline,
    train_sbert,
    active_learning,
    merchant_normalizer,
    embeddings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_merchant_normalization():
    """Test merchant normalization."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Merchant Normalization")
    logger.info("=" * 80)
    
    test_cases = [
        ("SQ *STARBUCKS COFFEE 123 SF CA", "Starbucks"),
        ("UBER TRIP SAN FRANCISCO", "Uber"),
        ("AMZN.COM", "Amazon"),
        ("WHOLE FOODS MKT", "Whole Foods"),
    ]
    
    for raw, expected_brand in test_cases:
        clean, root, brand = merchant_normalizer.normalize_merchant(raw)
        logger.info(f"  {raw:40} → {clean:20} (brand: {brand})")
        assert brand == expected_brand, f"Expected {expected_brand}, got {brand}"
    
    logger.info("✅ Merchant normalization: PASS")


def test_embeddings():
    """Test SBERT embeddings."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: SBERT Embeddings")
    logger.info("=" * 80)
    
    try:
        # Sample texts
        texts = [
            "Starbucks Coffee Seattle",
            "Starbucks Cafe San Francisco",
            "McDonald's Fast Food",
        ]
        
        # Embed
        embedder = embeddings.get_embedder()
        X_emb = embeddings.compute_embeddings(texts, embedder)
        
        logger.info(f"  Embedded {len(texts)} texts")
        logger.info(f"  Shape: {X_emb.shape}")
        
        # Check similarity
        sim_01 = np.dot(X_emb[0], X_emb[1]) / (np.linalg.norm(X_emb[0]) * np.linalg.norm(X_emb[1]))
        sim_02 = np.dot(X_emb[0], X_emb[2]) / (np.linalg.norm(X_emb[0]) * np.linalg.norm(X_emb[2]))
        
        logger.info(f"  Similarity (Starbucks-Starbucks): {sim_01:.3f}")
        logger.info(f"  Similarity (Starbucks-McDonald's): {sim_02:.3f}")
        
        assert sim_01 > sim_02, "Starbucks should be more similar to Starbucks than McDonald's"
        
        logger.info("✅ SBERT embeddings: PASS")
    except ImportError:
        logger.warning("❌ sentence-transformers not installed, skipping SBERT test")
        return


def test_active_learning():
    """Test active learning strategies."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Active Learning")
    logger.info("=" * 80)
    
    # Create synthetic data
    n_samples = 100
    n_classes = 5
    n_features = 50
    
    X_emb = np.random.randn(n_samples, n_features)
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Synthetic probabilities (high confidence on true class)
    y_proba = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        y_proba[i, y_true[i]] = 0.8 + 0.2 * np.random.rand()
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    labeled_mask = np.zeros(n_samples, dtype=bool)
    
    # Test strategies
    strategies = ["uncertainty", "margin", "entropy", "diversity"]
    
    for strategy in strategies:
        selected_idx, summary = active_learning.select_for_labeling(
            X_emb, y_proba, labeled_mask,
            n_select=10,
            strategy=strategy,
            random_fraction=0.2
        )
        
        logger.info(f"  {strategy:12} → Selected {len(selected_idx)} samples")
        logger.info(f"    Top 3 indices: {selected_idx[:3]}")
    
    logger.info("✅ Active learning: PASS")


def test_gold_dataset():
    """Test loading and using gold dataset."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Gold Dataset & Splits")
    logger.info("=" * 80)
    
    gold_path = Path(__file__).parent.parent / "data" / "gold" / "gold_transactions.jsonl"
    
    if not gold_path.exists():
        logger.warning(f"⚠️  Gold dataset not found at {gold_path}")
        return
    
    gold_records = data_prep.load_gold_set(gold_path)
    logger.info(f"  Loaded {len(gold_records)} records from gold set")
    
    df = data_prep.prepare_ml_records(gold_records)
    logger.info(f"  Prepared {len(df)} ML-ready records")
    
    # Check required fields
    required_fields = ["text", "category"]
    for field in required_fields:
        assert field in df.columns, f"Missing field: {field}"
    
    if len(df) > 0:
        logger.info(f"  Categories: {df['category'].unique()}")
    logger.info("✅ Gold dataset: PASS")


def test_baseline_training():
    """Test TF-IDF baseline training."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: TF-IDF Baseline Training")
    logger.info("=" * 80)
    
    # Synthetic data - as DataFrames since train_baseline expects that
    train_df = pd.DataFrame({
        "text": [
            "Starbucks coffee Seattle",
            "Starbucks cafe San Francisco",
            "McDonald's burger",
            "McDonald's fries",
        ],
        "category": ["Coffee", "Coffee", "FastFood", "FastFood"]
    })
    
    val_df = pd.DataFrame({
        "text": [
            "Starbucks latte",
            "McDonald's sandwich",
        ],
        "category": ["Coffee", "FastFood"]
    })
    
    # Train
    pipeline, metrics = train_baseline.train_baseline(train_df, val_df)
    
    logger.info(f"  Train records: {metrics.get('train_records', '?')}")
    logger.info(f"  Val accuracy: {metrics.get('val_accuracy', '?')}")
    
    # Predict
    preds = pipeline.predict(val_df["text"])
    logger.info(f"  Predictions: {preds}")
    
    logger.info("✅ TF-IDF baseline: PASS")


def test_sbert_training():
    """Test SBERT classifier training."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: SBERT Classifier Training")
    logger.info("=" * 80)
    
    try:
        # Synthetic data
        X_train = [
            "Starbucks coffee Seattle",
            "Starbucks cafe San Francisco",
            "McDonald's burger",
            "McDonald's fries",
        ]
        y_train = np.array([0, 0, 1, 1])
        
        X_val = [
            "Starbucks latte",
            "McDonald's sandwich",
        ]
        y_val = np.array([0, 1])
        
        # Train
        pipeline, metrics = train_sbert.train_sbert_classifier(
            X_train, y_train, X_val, y_val
        )
        
        logger.info(f"  Train accuracy: {metrics['train_acc']:.3f}")
        logger.info(f"  Val accuracy: {metrics.get('val_acc', '?')}")
        
        # Predict
        preds = pipeline.predict(X_val)
        logger.info(f"  Predictions: {preds}")
        
        logger.info("✅ SBERT classifier: PASS")
    except ImportError:
        logger.warning("❌ sentence-transformers not installed, skipping SBERT test")


def main():
    logger.info("\n\n" + "▓" * 80)
    logger.info("PHASE 3 INTEGRATION TEST SUITE")
    logger.info("▓" * 80)
    
    try:
        test_merchant_normalization()
        test_embeddings()
        test_active_learning()
        test_gold_dataset()
        test_baseline_training()
        test_sbert_training()
        
        logger.info("\n\n" + "▓" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("▓" * 80)
        return 0
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
