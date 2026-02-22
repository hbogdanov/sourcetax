"""Shared normalization helpers used across ingest, matching, and categorization."""

from __future__ import annotations


def normalize_merchant_name(value: str | None, case: str = "lower") -> str:
    """Return a normalized merchant string using the central normalizer when available."""
    if not value:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        from sourcetax.models import merchant_normalizer

        clean, _, brand = merchant_normalizer.normalize_merchant(text)
        normalized = (brand or clean or text).strip()
    except Exception:
        normalized = text

    if case == "upper":
        return normalized.upper()
    if case == "preserve":
        return normalized

    return normalized.lower()

