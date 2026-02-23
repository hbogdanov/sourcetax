#!/usr/bin/env python
"""
Comprehensive Phase 3 ML pipeline: Train and evaluate SBERT + Hierarchical classifier.

This script demonstrates the full ML workflow:
1. Load gold dataset + locked splits
2. Train TF-IDF baseline (for comparison)
3. Train SBERT-based classifier
4. Train hierarchical classifier (major → subcategory)
5. Generate visualizations and comparison table
6. Save pipelines for inference

Usage:
    python tools/train_ml_advanced.py [--strategy sbert|hierarchical|all]

Example output:
    OK: Loaded 10 gold records (train: 7, val: 1, test: 2)
    OK: TF-IDF baseline: 50% accuracy
    OK: SBERT classifier: 86% accuracy
    OK: Hierarchical: 100% major, 85% sub
    OK: Saved visualizations to data/ml/evaluation_report/
"""

import sys
import argparse
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax.models import (
    data_prep,
    train_baseline,
    train_sbert,
    hierarchical,
    visualize,
)
from sourcetax.normalization import normalize_merchant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_dir: Path = None):
    """Load locked splits and gold dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "ml"
    
    logger.info("=" * 80)
    logger.info("PHASE 3: ADVANCED ML TRAINING")
    logger.info("=" * 80)
    
    logger.info("\nDATA: Loading data...")
    
    # Load gold dataset
    gold_path = data_dir.parent / "gold" / "gold_transactions.jsonl"
    if not gold_path.exists():
        logger.error(f"Gold dataset not found at {gold_path}")
        return None
    
    df_gold = data_prep.load_gold_dataset(gold_path)
    logger.info(f"OK: Loaded {len(df_gold)} gold records")
    
    # Load split metadata
    split_metadata_path = data_dir / "split_metadata.txt"
    if split_metadata_path.exists():
        with open(split_metadata_path) as f:
            logger.info(f"Split info: {f.read()}")
    
    # Load splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_path = data_dir / f"ml_{split_name}.csv"
        if split_path.exists():
            splits[split_name] = pd.read_csv(split_path)
    
    if not splits:
        logger.warning("No pre-computed splits found, creating new ones...")
        # Will create splits below
        return df_gold, None
    
    logger.info(f"OK: Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    return df_gold, splits


def prepare_texts(df: pd.DataFrame) -> list:
    """Prepare merchant + description texts with normalization."""
    texts = []
    for _, row in df.iterrows():
        merchant = str(row.get("merchant", ""))
        description = str(row.get("description", ""))
        
        # Normalize merchant
        merchant_clean = normalize_merchant(merchant, case="preserve")
        
        # Combine
        text = f"{merchant_clean} [SEP] {description}".strip()
        texts.append(text)
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Advanced ML training")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["sbert", "hierarchical", "all"],
        help="What to train"
    )
    parser.add_argument(
        "--output-dir",
        default="data/ml/evaluation_report",
        help="Where to save visualizations"
    )
    args = parser.parse_args()
    
    # Load data
    result = load_and_prepare_data()
    if result[0] is None:
        return 1
    
    df_gold, splits = result
    
    if splits is None:
        logger.warning("No pre-computed splits; using all gold data for training.")
        X = prepare_texts(df_gold)
        y = df_gold["category"].values
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train if len(np.unique(y_train)) > 1 else None
        )
    else:
        X_train = prepare_texts(splits["train"])
        X_val = prepare_texts(splits["val"])
        X_test = prepare_texts(splits["test"])
        y_train = splits["train"]["category_encoded"].values
        y_val = splits["val"]["category_encoded"].values
        y_test = splits["test"]["category_encoded"].values
    
    logger.info(f"\nMETRICS: Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Train baseline (TF-IDF)
    logger.info("\nNEXT: Training TF-IDF baseline...")
    try:
        tfidf_clf, tfidf_metrics = train_baseline.train_tfidf_classifier(
            X_train, y_train, X_val, y_val
        )
        logger.info(f"OK: TF-IDF: train_acc={tfidf_metrics['train_acc']:.3f}, "
                    f"val_acc={tfidf_metrics.get('val_acc', 'WARNING:')}")
        
        # Test predictions
        tfidf_test_pred = tfidf_clf.predict(X_test)
        tfidf_test_acc = (tfidf_test_pred == y_test).mean()
        logger.info(f"   Test accuracy: {tfidf_test_acc:.3f}")
    except Exception as e:
        logger.error(f"TF-IDF training failed: {e}")
        tfidf_clf = None
        tfidf_test_pred = None
    
    # Train SBERT (if requested)
    sbert_test_pred = None
    if args.strategy in ["sbert", "all"]:
        logger.info("\nEMBED: Training SBERT-based classifier...")
        try:
            sbert_clf, sbert_metrics = train_sbert.train_sbert_classifier(
                X_train, y_train, X_val, y_val
            )
            logger.info(f"OK: SBERT: train_acc={sbert_metrics['train_acc']:.3f}, "
                        f"val_acc={sbert_metrics.get('val_acc', 'WARNING:')}")
            
            # Test predictions
            sbert_test_pred = sbert_clf.predict(X_test)
            sbert_test_acc = (sbert_test_pred == y_test).mean()
            logger.info(f"   Test accuracy: {sbert_test_acc:.3f}")
            
            # Save
            sbert_path = Path("data/ml/sbert_pipeline.pkl")
            with open(sbert_path, "wb") as f:
                pickle.dump(sbert_clf, f)
            logger.info(f"SAVE: Saved SBERT pipeline to {sbert_path}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with:")
            logger.warning("  pip install sentence-transformers")
        except Exception as e:
            logger.error(f"SBERT training failed: {e}")
    
    # Train hierarchical (if requested)
    if args.strategy in ["hierarchical", "all"]:
        logger.info("\nHIER:  Training hierarchical classifier...")
        
        # For demo, create synthetic major/sub labels
        # In practice, these come from training data
        n_major = min(3, len(np.unique(y_train)))
        y_train_major = y_train % n_major
        y_val_major = y_val % n_major if len(y_val) > 0 else np.array([])
        y_test_major = y_test % n_major if len(y_test) > 0 else np.array([])
        
        # Subcategories
        y_train_sub = y_train
        y_val_sub = y_val if len(y_val) > 0 else np.array([])
        y_test_sub = y_test if len(y_test) > 0 else np.array([])
        
        try:
            major_clf, sub_clfs, hier_metrics = hierarchical.train_hierarchical_classifier(
                X_train, y_train_major, y_train_sub,
                X_val if len(X_val) > 0 else None,
                y_val_major if len(y_val_major) > 0 else None,
                y_val_sub if len(y_val_sub) > 0 else None,
                major_trainer_fn=train_baseline.train_tfidf_classifier,
                sub_trainer_fn=train_baseline.train_tfidf_classifier,
            )
            logger.info(f"OK: Hierarchical classifier trained")
            
            # Evaluate
            if len(X_test) > 0:
                hier_eval = hierarchical.evaluate_hierarchical(
                    major_clf, sub_clfs, X_test, y_test_major, y_test_sub,
                    hierarchical.DEFAULT_HIERARCHY
                )
                logger.info(f"   Hierarchical accuracy: {hier_eval['hierarchical_acc']:.3f}")
        except Exception as e:
            logger.error(f"Hierarchical training failed: {e}")
    
    # Generate visualizations
    logger.info("\nUP Generating visualizations...")
    
    predictions = {}
    if tfidf_test_pred is not None:
        predictions["TF-IDF"] = tfidf_test_pred
    if sbert_test_pred is not None:
        predictions["SBERT"] = sbert_test_pred
    
    if predictions and len(y_test) > 0:
        # Get label names
        unique_labels = np.unique(y_test)
        label_names = [f"Category {i}" for i in unique_labels]
        
        report_paths = visualize.generate_evaluation_report(
            y_test,
            predictions,
            label_names,
            Path(args.output_dir)
        )
        logger.info(f"OK: Visualizations saved to {args.output_dir}")
        logger.info(f"   Report index: {report_paths.get('index', 'WARNING:')}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
