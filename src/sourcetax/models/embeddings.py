"""
Sentence embeddings (SBERT) for merchant + description.

Faster and more effective than TF-IDF for short/messy text.

Precomputes embeddings and caches them (don't recompute each run).
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def try_import_sbert():
    """Try to import sentence-transformers, with helpful error message."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Install with: pip install sentence-transformers\n"
            "Or: pip install -e .[embeddings]"
        )


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or cache a SentenceTransformer model.
    
    Options:
    - all-MiniLM-L6-v2: Fast (384 dims), good for speed
    - all-mpnet-base-v2: Quality (768 dims), slower but better
    - all-roberta-large-v1: Largest (1024 dims), best quality
    
    Default: all-MiniLM-L6-v2 (good balance)
    """
    SentenceTransformer = try_import_sbert()
    return SentenceTransformer(model_name)


def prepare_texts(df: pd.DataFrame) -> List[str]:
    """
    Combine merchant + description with [SEP] token.
    
    Transformer models understand [SEP] as field separator.
    """
    texts = []
    for _, row in df.iterrows():
        merchant = str(row.get("merchant", "")).strip()
        description = str(row.get("description", "")).strip()
        
        # Combine with separator token
        text = f"{merchant} [SEP] {description}".strip()
        texts.append(text)
    
    return texts


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute embeddings for texts.
    
    Args:
        texts: List of text strings
        model_name: SentenceTransformer model
        batch_size: Batch size for inference (larger = faster but more memory)
    
    Returns:
        Embeddings array (n_samples, embedding_dim)
    """
    print(f"📥 Loading embedder: {model_name}")
    embedder = get_embedder(model_name)
    
    print(f"🔢 Computing embeddings for {len(texts)} texts...")
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings


def cache_embeddings(
    embeddings: np.ndarray,
    cache_path: Path,
    metadata: Dict = None,
) -> Path:
    """
    Cache embeddings to disk.
    
    Avoids recomputing every run (saves time and electricity).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        "embeddings": embeddings,
        "metadata": metadata or {},
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    
    print(f"💾 Embeddings cached to {cache_path}")
    return cache_path


def load_cached_embeddings(cache_path: Path) -> Optional[Tuple[np.ndarray, Dict]]:
    """Load cached embeddings if they exist."""
    if not cache_path.exists():
        return None
    
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    
    return cache_data["embeddings"], cache_data.get("metadata", {})


def embed_dataset(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: Path = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Embed a dataset.
    
    Args:
        df: DataFrame with merchant, description columns
        model_name: SentenceTransformer model
        cache_path: Where to cache embeddings
    
    Returns:
        (embeddings array, updated DataFrame with embedding path)
    """
    if cache_path is None:
        cache_path = Path(__file__).parent.parent.parent.parent / "data" / "ml" / f"embeddings_{model_name}.pkl"
    
    # Check cache first
    cached = load_cached_embeddings(cache_path)
    if cached is not None:
        print(f"✅ Loaded cached embeddings from {cache_path}")
        embeddings, _ = cached
        return embeddings, df
    
    # Compute embeddings
    texts = prepare_texts(df)
    embeddings = compute_embeddings(texts, model_name)
    
    # Cache them
    cache_embeddings(
        embeddings,
        cache_path,
        metadata={
            "model": model_name,
            "num_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
        },
    )
    
    return embeddings, df


if __name__ == "__main__":
    # Test
    try:
        import pandas as pd
        
        # Create dummy data
        test_df = pd.DataFrame({
            "merchant": ["Starbucks", "Uber", "Amazon"],
            "description": ["Coffee", "Trip", "Retail"],
        })
        
        embeddings, _ = embed_dataset(test_df)
        print(f"\n✅ Test complete. Embeddings shape: {embeddings.shape}")
    except ImportError as e:
        print(f"⚠️  {e}")
