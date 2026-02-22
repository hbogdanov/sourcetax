"""
Hierarchical classification: Predict both major category and subcategory.

Structure:
    Major Category (e.g., "Meals & Entertainment")
        -> Subcategories (e.g., "Coffee", "Restaurant", "Bar")
    
    Utilities
        -> Electric, Gas, Water, Internet

This allows more granular expense tracking while keeping broad rollups.

Usage:
    from sourcetax.models import hierarchical
    major_clf, sub_clfs = hierarchical.build_hierarchical_classifier(...)
    major_pred, sub_pred = hierarchical.hierarchical_predict(...)
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger(__name__)


# Example hierarchy for tax categories
DEFAULT_HIERARCHY = {
    "Meals & Entertainment": ["Coffee", "Restaurant", "Bar", "Fast Food", "Groceries"],
    "Travel": ["Flight", "Hotel", "Rental Car", "Gas", "Parking"],
    "Office Supplies": ["Stationery", "Printer Supplies", "Furniture", "Equipment"],
    "Utilities": ["Electric", "Gas", "Water", "Internet", "Phone"],
    "Professional Services": ["Legal", "Accounting", "Consulting"],
    "Software & Subscriptions": ["Cloud Storage", "SaaS", "Productivity", "Development Tools"],
    "Other": ["Miscellaneous", "Uncategorized"],
}


def build_label_hierarchy(
    categories: List[str],
    hierarchy: Dict[str, List[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build mapping from subcategory to major category.
    
    Args:
        categories: List of all known categories (from training data)
        hierarchy: Dict mapping major → [subcategories] (uses DEFAULT if None)
    
    Returns:
        (subcat_to_major, major_to_subs)
    """
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY
    
    subcat_to_major = {}
    major_to_subs = {}
    
    for major, subs in hierarchy.items():
        major_to_subs[major] = subs
        for sub in subs:
            if sub in categories:
                subcat_to_major[sub] = major
    
    logger.info(f"Built hierarchy: {len(major_to_subs)} major categories, "
                f"{len(subcat_to_major)} mapped subcategories")
    
    return subcat_to_major, major_to_subs


def hierarchical_predict(
    major_classifier: Any,  # sklearn-like classifier
    sub_classifiers: Dict[str, Any],  # {major_category: classifier}
    X: List[str],  # texts to classify
    subcat_to_major: Dict[str, str],  # subcategory → major mapping
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage hierarchical prediction.
    
    Stage 1: Predict major category (Meals & Entertainment, Travel, etc.)
    Stage 2: For each sample, predict subcategory from {major_category} classifier
    
    Args:
        major_classifier: Trained classifier for major categories
        sub_classifiers: Dict of classifiers, one per major category
        X: List of texts to classify
        subcat_to_major: Subcategory → major category mapping (for label lookup)
    
    Returns:
        (major_predictions, sub_predictions)
        - major_predictions: Array of major category indices
        - sub_predictions: Array of subcategory indices
    """
    # Stage 1: Predict major categories
    major_preds = major_classifier.predict(X)
    major_labels = major_classifier.classes_
    
    sub_preds = np.zeros(len(X), dtype=int)
    
    # Stage 2: For each major category, predict subcategories
    for major_idx in range(len(major_labels)):
        major_label = major_labels[major_idx]
        
        if major_label not in sub_classifiers:
            logger.warning(f"No subcategory classifier for {major_label}")
            continue
        
        # Find samples assigned to this major category
        mask = major_preds == major_idx
        if not mask.any():
            continue
        
        X_major = [X[i] for i in range(len(X)) if mask[i]]
        
        # Predict subcategories for these samples
        sub_classifier = sub_classifiers[major_label]
        sub_preds_local = sub_classifier.predict(X_major)
        sub_labels = sub_classifier.classes_
        
        # Map local subcategory indices to global indices
        # (This is simplified; in practice you'd have a global label encoder)
        sub_preds[mask] = sub_preds_local
    
    return major_preds, sub_preds


def train_hierarchical_classifier(
    X_train: List[str],
    y_train_major: np.ndarray,  # major category indices
    y_train_sub: np.ndarray,  # subcategory indices
    X_val: List[str] = None,
    y_val_major: np.ndarray = None,
    y_val_sub: np.ndarray = None,
    major_trainer_fn: callable = None,  # function to train major classifier
    sub_trainer_fn: callable = None,  # function to train subcategory classifiers
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Train hierarchical classifier.
    
    Args:
        X_train: Training texts
        y_train_major: Major category labels
        y_train_sub: Subcategory labels
        X_val: Validation texts
        y_val_major: Validation major labels
        y_val_sub: Validation sub labels
        major_trainer_fn: Function(X, y) -> classifier (e.g., TF-IDF or SBERT)
        sub_trainer_fn: Function(X, y) -> classifier per major category
    
    Returns:
        (major_classifier, sub_classifiers, metrics)
    """
    logger.info("Training hierarchical classifier...")
    
    # Default trainers if not provided
    if major_trainer_fn is None:
        from . import train_baseline
        major_trainer_fn = train_baseline.train_tfidf_classifier
    
    if sub_trainer_fn is None:
        sub_trainer_fn = major_trainer_fn  # Use same trainer for subs
    
    # Train major category classifier
    logger.info("Training major category classifier...")
    major_clf, major_metrics = major_trainer_fn(
        X_train, y_train_major,
        X_val, y_val_major,
    )
    logger.info(f"Major classifier metrics: {major_metrics}")
    
    # Train subcategory classifiers (one per major category)
    logger.info("Training subcategory classifiers...")
    sub_clfs = {}
    major_labels = major_clf.classes_
    
    for major_idx, major_label in enumerate(major_labels):
        # Find samples in this major category
        mask = y_train_major == major_idx
        if mask.sum() < 2:
            logger.warning(f"Not enough samples for {major_label} (n={mask.sum()})")
            continue
        
        X_major = [X_train[i] for i in range(len(X_train)) if mask[i]]
        y_major_sub = y_train_sub[mask]
        
        # Corresponding validation set
        X_val_major = None
        y_val_major_sub = None
        
        if X_val is not None:
            val_mask = y_val_major == major_idx
            if val_mask.sum() > 0:
                X_val_major = [X_val[i] for i in range(len(X_val)) if val_mask[i]]
                y_val_major_sub = y_val_sub[val_mask]
        
        # Train
        logger.info(f"  {major_label}: {len(X_major)} training samples")
        sub_clfs[major_label], sub_metrics = sub_trainer_fn(
            X_major, y_major_sub,
            X_val_major, y_val_major_sub,
        )
        logger.info(f"    Metrics: {sub_metrics}")
    
    metrics = {
        "major": major_metrics,
        "sub": {label: {} for label in major_labels},  # Placeholder
    }
    
    logger.info("Hierarchical classifier training complete")
    
    return major_clf, sub_clfs, metrics


def evaluate_hierarchical(
    major_clf: Any,
    sub_clfs: Dict[str, Any],
    X: List[str],
    y_major: np.ndarray,
    y_sub: np.ndarray,
    subcat_to_major: Dict[str, str],
    major_labels: list = None,
) -> Dict[str, float]:
    """
    Evaluate hierarchical classifier.
    
    Returns:
        Dict with:
        - "major_acc": Accuracy on major categories
        - "sub_acc": Accuracy on subcategories
        - "hierarchical_acc": Both major AND sub correct
    """
    from sklearn.metrics import accuracy_score
    
    # Predict
    major_preds, sub_preds = hierarchical_predict(
        major_clf, sub_clfs, X, subcat_to_major
    )
    
    # Metrics
    major_acc = accuracy_score(y_major, major_preds)
    sub_acc = accuracy_score(y_sub, sub_preds)
    
    # Full hierarchy correctness
    both_correct = (major_preds == y_major) & (sub_preds == y_sub)
    hier_acc = both_correct.mean()
    
    return {
        "major_acc": major_acc,
        "sub_acc": sub_acc,
        "hierarchical_acc": hier_acc,
    }
