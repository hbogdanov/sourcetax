"""
Sentence embeddings (SBERT) utilities for merchant + description text.

Supports:
- Lazy optional import of `sentence-transformers`
- Embedding computation from model name or preloaded embedder instance
- Disk caching of computed embeddings
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def try_import_sbert():
    """Try to import sentence-transformers, with a helpful error message."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Install with: pip install sentence-transformers\n"
            "Or: pip install -e .[embeddings]"
        ) from exc


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Construct a SentenceTransformer embedder."""
    SentenceTransformer = try_import_sbert()
    return SentenceTransformer(model_name)


def prepare_texts(df: pd.DataFrame) -> List[str]:
    """Combine merchant + description fields into a single text input."""
    texts: List[str] = []
    for _, row in df.iterrows():
        merchant = str(row.get("merchant", "")).strip()
        description = str(row.get("description", "")).strip()
        texts.append(f"{merchant} [SEP] {description}".strip())
    return texts


def compute_embeddings(
    texts: List[str],
    model_name: Any = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute embeddings for texts.

    `model_name` can be either a model-name string or a preloaded embedder object.
    """
    if isinstance(model_name, str):
        print(f"Loading embedder: {model_name}")
        embedder = get_embedder(model_name)
    else:
        print("Using provided embedder instance")
        embedder = model_name

    print(f"Computing embeddings for {len(texts)} texts...")
    vectors = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    vectors = np.asarray(vectors)
    print(f"Embeddings shape: {vectors.shape}")
    return vectors


def cache_embeddings(
    embeddings: np.ndarray,
    cache_path: Path,
    metadata: Dict | None = None,
) -> Path:
    """Persist embeddings and metadata to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump({"embeddings": embeddings, "metadata": metadata or {}}, f)
    print(f"Embeddings cached to {cache_path}")
    return cache_path


def load_cached_embeddings(cache_path: Path) -> Optional[Tuple[np.ndarray, Dict]]:
    """Load cached embeddings if available."""
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        cache_data = pickle.load(f)
    return cache_data["embeddings"], cache_data.get("metadata", {})


def embed_dataset(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: Path | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Embed a dataframe and cache the vectors."""
    if cache_path is None:
        cache_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "ml"
            / f"embeddings_{model_name}.pkl"
        )

    cached = load_cached_embeddings(cache_path)
    if cached is not None:
        print(f"Loaded cached embeddings from {cache_path}")
        vectors, _ = cached
        return vectors, df

    texts = prepare_texts(df)
    vectors = compute_embeddings(texts, model_name)
    cache_embeddings(
        vectors,
        cache_path,
        metadata={
            "model": model_name,
            "num_samples": len(vectors),
            "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        },
    )
    return vectors, df


if __name__ == "__main__":
    try:
        test_df = pd.DataFrame(
            {
                "merchant": ["Starbucks", "Uber", "Amazon"],
                "description": ["Coffee", "Trip", "Retail"],
            }
        )
        emb, _ = embed_dataset(test_df)
        print(f"Test complete. Embeddings shape: {emb.shape}")
    except ImportError as exc:
        print(f"SBERT unavailable: {exc}")
