#!/usr/bin/env python
"""
Active learning selection: Choose best samples to label next.

Usage:
    python tools/select_for_labeling.py --strategy diversity --n 50

Outputs:
    - Ranked list of samples to label
    - Index CSV for use in app_review.py
"""

import sys
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax.models import active_learning, embeddings


def load_unlabeled_pool(data_dir: Path = None) -> pd.DataFrame:
    """Load unlabeled transaction pool (e.g., from a larger dataset)."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "ml"
    
    # For now, return empty if no unlabeled pool exists
    # In practice, you'd have a separate unlabeled_pool.csv
    unlabeled_file = data_dir / "unlabeled_pool.csv"
    if unlabeled_file.exists():
        return pd.read_csv(unlabeled_file)
    else:
        print(f"⚠️  No unlabeled pool found at {unlabeled_file}")
        print("   Create one by exporting transactions without labels.")
        return pd.DataFrame()


def get_predictions(
    pool_df: pd.DataFrame,
    pipeline_path: Path = None,
) -> np.ndarray:
    """Get predictions and probabilities for a pool."""
    if pipeline_path is None:
        pipeline_path = Path(__file__).parent.parent / "data" / "ml" / "baseline_pipeline.pkl"
    
    if not pipeline_path.exists():
        print(f"❌ Pipeline not found at {pipeline_path}")
        print("   Run: python tools/train_ml_baseline.py")
        return None
    
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    
    print(f"Predicting on {len(pool_df)} samples...")
    y_proba = pipeline.predict_proba(pool_df["text"])
    
    return y_proba


def main():
    parser = argparse.ArgumentParser(description="Select samples for active learning")
    parser.add_argument(
        "--strategy",
        default="diversity",
        choices=["uncertainty", "margin", "entropy", "diversity"],
        help="Active learning strategy"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of samples to select"
    )
    parser.add_argument(
        "--random-fraction",
        type=float,
        default=0.2,
        help="Fraction of random samples to mix in (avoid overfitting to edge cases)"
    )
    parser.add_argument(
        "--output",
        default="data/ml/selection_for_labeling.csv",
        help="Output CSV with selected samples"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"ACTIVE LEARNING: SELECT {args.n} SAMPLES FOR LABELING")
    print(f"Strategy: {args.strategy}")
    print("=" * 80)
    
    # Load unlabeled pool
    print("\n📋 Loading unlabeled pool...")
    pool_df = load_unlabeled_pool()
    
    if len(pool_df) == 0:
        print("\n⚠️  No unlabeled pool available.")
        print("\nToDemo mode: Using test set as unlabeled pool")
        test_path = Path(__file__).parent.parent / "data" / "ml" / "ml_test.csv"
        if test_path.exists():
            pool_df = pd.read_csv(test_path)
            print(f"Using test set ({len(pool_df)} samples)")
        else:
            print("Run: python tools/train_ml_baseline.py")
            return 1
    
    print(f"Pool size: {len(pool_df)}")
    
    # Get predictions
    print("\n🔮 Getting predictions...")
    y_proba = get_predictions(pool_df)
    
    if y_proba is None:
        return 1
    
    # Compute embeddings
    print("\n🔢 Computing embeddings...")
    emb, _ = embeddings.embed_dataset(pool_df)
    
    # Apply active learning
    print(f"\n🎯 Selecting {args.n} samples with {args.strategy}...")
    labeled_mask = np.zeros(len(pool_df), dtype=bool)  # All unlabeled in this demo
    
    selected_idx, summary = active_learning.select_for_labeling(
        emb,
        y_proba,
        labeled_mask,
        n_select=args.n,
        strategy=args.strategy,
        random_fraction=args.random_fraction,
    )
    
    # Create output
    print(f"\n✅ Selected {len(selected_idx)} samples")
    
    output_df = pool_df.iloc[selected_idx].copy()
    output_df["max_prob"] = summary["max_prob"].values
    output_df["entropy"] = summary["entropy"].values
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved to {output_path}")
    print(f"\nTop 10 to label (highest uncertainty first):")
    print(output_df[["text", "max_prob", "entropy"]].head(10).to_string())
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Review selected samples in app_review.py")
    print("2. Label them (mark matches, override categories)")
    print("3. Re-run: python tools/train_ml_baseline.py")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
