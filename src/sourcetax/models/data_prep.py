"""
Data preparation for ML baseline.

Loads gold dataset, splits into train/validation/test.
Critical: Lock the split. Never change it.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split


def load_gold_set(gold_path: Path = None) -> List[Dict]:
    """Load gold standard dataset (JSONL)."""
    if gold_path is None:
        gold_path = Path(__file__).parent.parent.parent.parent / "data" / "gold" / "gold_transactions.jsonl"
    
    records = []
    if gold_path.exists():
        with open(gold_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def prepare_ml_records(gold_records: List[Dict]) -> pd.DataFrame:
    """
    Convert gold records to ML-ready format.
    
    Combines merchant_raw + description into single text input.
    Filters to records with ground-truth category_final.
    Returns DataFrame with features and labels.
    """
    data = []
    
    for record in gold_records:
        category = record.get("category_final")
        
        # Only use records with ground-truth labels
        if not category:
            continue
        
        merchant = record.get("merchant_raw", "").strip()
        description = record.get("raw_payload", {}).get("description", "").strip()
        
        # Combine into single text feature
        text = f"{merchant} {description}".strip()
        
        if text:
            data.append({
                "id": record.get("id"),
                "text": text,
                "merchant": merchant,
                "description": description,
                "category": category,
                "source": record.get("source"),
            })
    
    return pd.DataFrame(data)


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test.
    
    For small datasets: uses stratification when possible, falls back to random when not.
    For large datasets: stratified by category to ensure each split has category balance.
    
    Args:
        df: ML-ready DataFrame
        train_size: proportion for training (default 70%)
        val_size: proportion for validation (default 15%)
        test_size: proportion for testing (default 15%)
        random_state: seed for reproducibility (locked at 42)
    
    Returns:
        (train_df, val_df, test_df)
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    stratify = None
    
    # Check if stratification is viable
    # (each class must have >1 record)
    class_counts = df["category"].value_counts()
    if len(df) >= 5 and (class_counts >= 2).all():
        stratify = df["category"]
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify,
        random_state=random_state,
    )
    
    # Second split: val vs test (from temp pool)
    val_ratio = val_size / (val_size + test_size)
    
    # Recheck stratification for second split
    stratify = None
    if len(temp_df) >= 5 and (temp_df["category"].value_counts() >= 2).all():
        stratify = temp_df["category"]
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=stratify,
        random_state=random_state,
    )
    
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = None,
) -> Path:
    """
    Save splits to CSV files for inspection and reproducibility.
    
    Files:
    - ml_train.csv
    - ml_val.csv
    - ml_test.csv
    - split_metadata.txt
    
    Args:
        train_df, val_df, test_df: DataFrames
        output_dir: where to save (default: data/ml/)
    
    Returns:
        output_dir Path
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "ml"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_df.to_csv(output_dir / "ml_train.csv", index=False)
    val_df.to_csv(output_dir / "ml_val.csv", index=False)
    test_df.to_csv(output_dir / "ml_test.csv", index=False)
    
    # Save metadata
    metadata = f"""# ML Dataset Split (Locked)

Timestamp: {pd.Timestamp.now().isoformat()}
Random seed: 42 (reproducible)

## Sizes
Train:      {len(train_df)} records
Validation: {len(val_df)} records
Test:       {len(test_df)} records
Total:      {len(train_df) + len(val_df) + len(test_df)} records

## Categories
Train:
{train_df['category'].value_counts().to_string()}

Validation:
{val_df['category'].value_counts().to_string()}

Test:
{test_df['category'].value_counts().to_string()}

## Important
- Test set is SACRED. Do not tune on it.
- Use train for training.
- Use validation for hyperparameter search.
- Use test only for final evaluation.
- Never peek at test labels during development.
"""
    
    with open(output_dir / "split_metadata.txt", "w") as f:
        f.write(metadata)
    
    print(f"✅ Splits saved to {output_dir}")
    print(metadata)
    
    return output_dir


def main():
    """Load gold set, split, save."""
    print("📖 Loading gold dataset...")
    gold_records = load_gold_set()
    print(f"   Loaded {len(gold_records)} records")
    
    print("🔄 Preparing ML records...")
    df = prepare_ml_records(gold_records)
    print(f"   {len(df)} records with labels")
    
    if len(df) < 5:
        print("⚠️  Warning: Very small dataset (<5 records). ML unreliable at this scale.")
        print("   Next step: expand gold set to 200+ records using app_review.py")
    
    print("📊 Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)
    
    print("💾 Saving splits...")
    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
