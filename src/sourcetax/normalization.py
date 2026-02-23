"""Shared normalization helpers used across ingest, matching, and categorization."""

from __future__ import annotations

from typing import Optional, Tuple


def normalize_merchant_components(value: str | None) -> Tuple[str, str, Optional[str]]:
    """Return (clean, root, brand) using the central merchant normalizer when available."""
    if not value:
        return "", "", None

    text = str(value).strip()
    if not text:
        return "", "", None

    try:
        from sourcetax.models import merchant_normalizer

        return merchant_normalizer.normalize_merchant(text)
    except Exception:
        return text, text, None


def normalize_merchant(value: str | None, case: str = "lower") -> str:
    """Canonical merchant normalization used by matching and ingest."""
    clean, _, brand = normalize_merchant_components(value)
    normalized = (brand or clean).strip()

    if case == "upper":
        return normalized.upper()
    if case == "preserve":
        return normalized
    return normalized.lower()


def normalize_merchant_name(value: str | None, case: str = "lower") -> str:
    """Backward-compatible alias for the shared merchant normalizer."""
    return normalize_merchant(value, case=case)
