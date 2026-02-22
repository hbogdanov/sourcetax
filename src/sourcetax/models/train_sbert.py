"""
Train LogisticRegression on top of SBERT embeddings.

This is more powerful than TF-IDF because:
1. Embeddings capture semantic similarity (not just token overlap)
2. Works well with short/noisy transaction data (merchant + description)
3. Pre-trained model generalizes across domains

Usage:
    from sourcetax.models import train_sbert
    pipeline, metrics = train_sbert.train_sbert_classifier(X_train, y_train, X_val, y_val)
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

from . import embeddings

logger = logging.getLogger(__name__)


def _prepare_embedding_pipeline(
    embedder=None,
    model_name: str = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create embedding → scaling pipeline.
    
    Args:
        embedder: Pre-loaded SentenceTransformer (if None, creates new)
        model_name: Model name if creating new embedder (default: all-MiniLM-L6-v2)
    
    Returns:
        Custom pipeline-like object that embeds then scales
    """
    if embedder is None:
        embedder = embeddings.get_embedder(model_name or "all-MiniLM-L6-v2")
    
    scaler = StandardScaler()
    
    class EmbeddingTransformer:
        """Transform texts to scaled embeddings."""
        
        def __init__(self, emb, scl):
            self.embedder = emb
            self.scaler = scl
            self.fitted = False
        
        def fit(self, X, y=None):
            # X should be list of texts
            emb_vectors = embeddings.compute_embeddings(X, self.embedder)
            self.scaler.fit(emb_vectors)
            self.fitted = True
            return self
        
        def transform(self, X):
            emb_vectors = embeddings.compute_embeddings(X, self.embedder)
            return self.scaler.transform(emb_vectors)
        
        def fit_transform(self, X, y=None):
            self.fit(X, y)
            return self.transform(X)
    
    return EmbeddingTransformer(embedder, scaler), {"embedder": embedder}


def train_sbert_classifier(
    X_train: list,  # list of text (merchant + description)
    y_train: np.ndarray,  # labels
    X_val: list = None,
    y_val: np.ndarray = None,
    model_name: str = None,
    max_iter: int = 1000,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train LogisticRegression on SBERT embeddings.
    
    Args:
        X_train: List of texts to train on
        y_train: Category labels (0-indexed)
        X_val: Validation texts (for reporting)
        y_val: Validation labels
        model_name: SentenceTransformer model (default: all-MiniLM-L6-v2)
        max_iter: LogisticRegression max_iter
    
    Returns:
        (pipeline, metrics_dict)
        - pipeline: sklearn Pipeline compatible
        - metrics_dict: {"train_acc": float, "val_acc": float, ...}
    """
    logger.info(f"Training SBERT classifier on {len(X_train)} samples...")
    
    # Create embeddings
    embedder = embeddings.get_embedder(model_name or "all-MiniLM-L6-v2")
    logger.info("Computing embeddings...")
    
    X_train_emb = embeddings.compute_embeddings(X_train, embedder)
    logger.info(f"Embeddings shape: {X_train_emb.shape}")
    
    # Scale embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    
    # Train classifier
    logger.info("Training LogisticRegression...")
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1,
        solver='lbfgs',  # handles small datasets better
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    logger.info(f"Train accuracy: {train_acc:.3f}")
    
    metrics = {"train_acc": train_acc}
    
    if X_val is not None and y_val is not None:
        X_val_emb = embeddings.compute_embeddings(X_val, embedder)
        X_val_scaled = scaler.transform(X_val_emb)
        val_acc = clf.score(X_val_scaled, y_val)
        metrics["val_acc"] = val_acc
        logger.info(f"Val accuracy: {val_acc:.3f}")
    
    # Create sklearn-compatible pipeline
    pipeline = _SBertPipeline(embedder, scaler, clf, model_name)
    
    return pipeline, metrics


class _SBertPipeline:
    """sklearn-compatible wrapper for SBERT + scaler + classifier."""
    
    def __init__(self, embedder, scaler, classifier, model_name: str = None):
        self.embedder = embedder
        self.scaler = scaler
        self.classifier = classifier
        self.model_name = model_name or "all-MiniLM-L6-v2"
    
    def predict(self, X):
        """Predict categories for texts."""
        X_emb = embeddings.compute_embeddings(X, self.embedder)
        X_scaled = self.scaler.transform(X_emb)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities for categories."""
        X_emb = embeddings.compute_embeddings(X, self.embedder)
        X_scaled = self.scaler.transform(X_emb)
        return self.classifier.predict_proba(X_scaled)
    
    def score(self, X, y):
        """Accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    @property
    def classes_(self):
        """Category labels."""
        return self.classifier.classes_
    
    def __repr__(self):
        return f"_SBertPipeline(model={self.model_name})"
