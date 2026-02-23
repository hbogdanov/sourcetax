"""
Active learning: Select most informative unlabeled samples to label next.

Strategies:
- Uncertainty sampling (pick lowest confidence)
- Margin sampling (smallest gap between top-2 predictions)
- Entropy (highest prediction entropy)
- Diversity (spread across embedding space)

Use this to build your gold dataset efficiently.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


def uncertainty_sampling(
    y_proba: np.ndarray,
    n_select: int = 50,
    random_fraction: float = 0.2,
) -> np.ndarray:
    """
    Select samples with lowest max probability.
    
    Model is most uncertain about these.
    
    Args:
        y_proba: Prediction probabilities (n_samples, n_classes)
        n_select: Number of samples to select
        random_fraction: Mix in some random samples (e.g., 0.2 = 20% random)
    
    Returns:
        Indices of selected samples
    """
    max_probs = np.max(y_proba, axis=1)
    uncertainty = 1.0 - max_probs
    
    n_uncertain = int(n_select * (1 - random_fraction))
    n_random = n_select - n_uncertain
    
    # Select most uncertain
    uncertain_idx = np.argsort(uncertainty)[-n_uncertain:]
    
    # Mix in random (avoid overfitting to weird edge cases)
    all_idx = np.arange(len(y_proba))
    available = np.setdiff1d(all_idx, uncertain_idx)
    random_idx = np.random.choice(available, size=min(n_random, len(available)), replace=False)
    
    selected = np.concatenate([uncertain_idx, random_idx])
    return selected


def margin_sampling(
    y_proba: np.ndarray,
    n_select: int = 50,
    random_fraction: float = 0.2,
) -> np.ndarray:
    """
    Select samples with smallest margin (difference between top-2 predictions).
    
    Args:
        y_proba: Prediction probabilities (n_samples, n_classes)
        n_select: Number of samples to select
        random_fraction: Mix in some random samples
    
    Returns:
        Indices of selected samples
    """
    top2 = np.partition(y_proba, -2, axis=1)[:, -2:]
    top1, top2_val = np.max(y_proba, axis=1), np.sort(top2, axis=1)[:, -2]
    margin = top1 - top2_val
    
    n_uncertain = int(n_select * (1 - random_fraction))
    n_random = n_select - n_uncertain
    
    # Select smallest margin
    uncertain_idx = np.argsort(margin)[:n_uncertain]
    
    # Mix in random
    all_idx = np.arange(len(y_proba))
    available = np.setdiff1d(all_idx, uncertain_idx)
    random_idx = np.random.choice(available, size=min(n_random, len(available)), replace=False)
    
    selected = np.concatenate([uncertain_idx, random_idx])
    return selected


def entropy_sampling(
    y_proba: np.ndarray,
    n_select: int = 50,
    random_fraction: float = 0.2,
) -> np.ndarray:
    """
    Select samples with highest prediction entropy.
    
    Args:
        y_proba: Prediction probabilities (n_samples, n_classes)
        n_select: Number of samples to select
        random_fraction: Mix in some random samples
    
    Returns:
        Indices of selected samples
    """
    # Entropy = -sum(p * log(p))
    entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
    
    n_uncertain = int(n_select * (1 - random_fraction))
    n_random = n_select - n_uncertain
    
    # Select highest entropy
    uncertain_idx = np.argsort(entropy)[-n_uncertain:]
    
    # Mix in random
    all_idx = np.arange(len(y_proba))
    available = np.setdiff1d(all_idx, uncertain_idx)
    random_idx = np.random.choice(available, size=min(n_random, len(available)), replace=False)
    
    selected = np.concatenate([uncertain_idx, random_idx])
    return selected


def diversity_sampling(
    embeddings: np.ndarray,
    y_proba: np.ndarray,
    n_select: int = 50,
    n_clusters: int = 10,
    random_fraction: float = 0.2,
) -> np.ndarray:
    """
    Select uncertain samples spread across embedding space (avoid duplicates).
    
    Approach:
    1. Cluster embeddings with K-means
    2. Pick uncertain samples from different clusters
    3. Mix in random for coverage
    
    Args:
        embeddings: Embedding vectors (n_samples, n_dims)
        y_proba: Prediction probabilities (n_samples, n_classes)
        n_select: Number of samples to select
        n_clusters: Number of clusters
        random_fraction: Mix in some random samples
    
    Returns:
        Indices of selected samples
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("WARNING:  sklearn not available. Falling back to uncertainty sampling.")
        return uncertainty_sampling(y_proba, n_select, random_fraction)
    
    # Cluster
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Uncertainty
    uncertainty = 1.0 - np.max(y_proba, axis=1)
    
    # Pick uncertain samples from each cluster
    selected = []
    n_per_cluster = max(1, n_select // n_clusters)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if not cluster_mask.any():
            continue
        
        cluster_indices = np.where(cluster_mask)[0]
        cluster_uncertainty = uncertainty[cluster_indices]
        
        # Pick most uncertain in this cluster
        top_in_cluster = cluster_indices[np.argsort(cluster_uncertainty)[-n_per_cluster:]]
        selected.extend(top_in_cluster)
    
    selected = np.array(selected)[:n_select]
    
    # Fill with random if needed
    if len(selected) < n_select:
        all_idx = np.arange(len(embeddings))
        available = np.setdiff1d(all_idx, selected)
        n_more = n_select - len(selected)
        more = np.random.choice(available, size=min(n_more, len(available)), replace=False)
        selected = np.concatenate([selected, more])
    
    return selected


def select_for_labeling(
    embeddings: np.ndarray,
    y_proba: np.ndarray,
    labeled_mask: np.ndarray,
    n_select: int = 50,
    strategy: str = "diversity",
    random_fraction: float = 0.2,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Select unlabeled samples to label next.
    
    Args:
        embeddings: Embedding vectors
        y_proba: Prediction probabilities
        labeled_mask: Boolean array (True = labeled, False = unlabeled)
        n_select: Number of samples to select
        strategy: "uncertainty", "margin", "entropy", or "diversity"
        random_fraction: Fraction of random samples to mix in
    
    Returns:
        (selected_indices, summary_df)
    """
    # Only consider unlabeled
    unlabeled_idx = np.where(~labeled_mask)[0]
    
    if len(unlabeled_idx) == 0:
        print("WARNING:  No unlabeled samples remaining!")
        return np.array([]), pd.DataFrame()
    
    # Adjust n_select
    n_select = min(n_select, len(unlabeled_idx))
    
    print(f"Selecting {n_select} from {len(unlabeled_idx)} unlabeled samples using {strategy}...")
    
    if strategy == "uncertainty":
        selected_abs = uncertainty_sampling(y_proba, n_select, random_fraction)
    elif strategy == "margin":
        selected_abs = margin_sampling(y_proba, n_select, random_fraction)
    elif strategy == "entropy":
        selected_abs = entropy_sampling(y_proba, n_select, random_fraction)
    elif strategy == "diversity":
        selected_abs = diversity_sampling(embeddings, y_proba, n_select, random_fraction=random_fraction)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Filter to only unlabeled
    selected = np.intersect1d(selected_abs, unlabeled_idx)
    
    # Summary
    summary = pd.DataFrame({
        "index": selected,
        "max_prob": np.max(y_proba[selected], axis=1),
        "entropy": -np.sum(y_proba[selected] * np.log(y_proba[selected] + 1e-10), axis=1),
    })
    summary = summary.sort_values("max_prob")
    
    return selected, summary


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n_samples, n_classes = 100, 5
    
    # Dummy data
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    embeddings = np.random.randn(n_samples, 384)
    labeled_mask = np.random.rand(n_samples) < 0.3  # 30% labeled
    
    selected, summary = select_for_labeling(
        embeddings, y_proba, labeled_mask, n_select=10, strategy="uncertainty"
    )
    
    print(f"\nOK: Selected {len(selected)} samples")
    print(summary.head())
