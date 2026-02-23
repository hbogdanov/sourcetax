"""
Train baseline ML model: TF-IDF + LogisticRegression.

Simple. Interpretable. Strong baseline.
"""

import pickle
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_train_val(
    data_dir: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "ml"
    
    train_df = pd.read_csv(data_dir / "ml_train.csv")
    val_df = pd.read_csv(data_dir / "ml_val.csv")
    
    return train_df, val_df


def build_pipeline() -> Pipeline:
    """
    Build TF-IDF + LogisticRegression pipeline.
    
    TF-IDF config:
    - Bigrams (1-2): capture merchant phrases like "STARBUCKS COFFEE"
    - min_df=2: ignore words appearing in <2 documents (noise)
    - max_features=5000: limit vocabulary
    - lowercase: normalize case
    
    LogisticRegression config:
    - max_iter=200: enough for convergence
    - class_weight="balanced": handle category imbalance
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        lowercase=True,
        stop_words="english",
    )
    
    classifier = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        random_state=42,
    )
    
    return Pipeline([
        ("tfidf", tfidf),
        ("classifier", classifier),
    ])


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
) -> Tuple[Pipeline, dict]:
    """
    Train baseline model.
    
    Args:
        train_df: Training data
        val_df: Validation data (optional, used for reporting)
    
    Returns:
        (pipeline, metrics_dict)
    """
    print("BUILD: Building pipeline...")
    pipeline = build_pipeline()
    
    print("TRAIN: Training on {} records...".format(len(train_df)))
    pipeline.fit(train_df["text"], train_df["category"])
    
    metrics = {
        "train_records": len(train_df),
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
    }
    
    # Validation metrics (if provided)
    if val_df is not None:
        print("METRICS: Evaluating on validation set...")
        y_pred = pipeline.predict(val_df["text"])
        y_true = val_df["category"]
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        
        metrics.update({
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_records": len(val_df),
        })
        
        print(f"   Accuracy:  {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall:    {recall:.1%}")
        print(f"   F1:        {f1:.1%}")
    
    return pipeline, metrics


def save_pipeline(
    pipeline: Pipeline,
    output_path: Path = None,
) -> Path:
    """Save trained pipeline to disk."""
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent.parent / "data" / "ml" / "baseline_pipeline.pkl"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    print(f"OK: Pipeline saved to {output_path}")
    return output_path


def main():
    """Train and save baseline model."""
    print("LOAD: Loading splits...")
    train_df, val_df = load_train_val()
    
    print(f"   Train: {len(train_df)} records")
    print(f"   Val:   {len(val_df)} records")
    
    print("\n🤖 Training baseline...")
    pipeline, metrics = train_baseline(train_df, val_df)
    
    print("\nSAVE: Saving pipeline...")
    save_pipeline(pipeline)
    
    print("\nUP Baseline Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.1%}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
