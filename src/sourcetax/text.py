"""Shared text preprocessing for ML training and inference."""

from __future__ import annotations

import re
from typing import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_CURRENCY_TOKEN_RE = re.compile(r"\b(?:usd|eur|gbp|cad|aud|inr|jpy)\b", re.IGNORECASE)
_COUNTRY_TOKEN_RE = re.compile(
    r"\b(?:us|usa|uk|gb|in|india|ca|canada|au|australia|sg|singapore|ae|uae|de|germany|fr|france)\b",
    re.IGNORECASE,
)


def normalize_text(
    text: str,
    remove_currency_tokens: bool = False,
    remove_country_tokens: bool = False,
    lowercase: bool = True,
) -> str:
    value = str(text or "").strip()
    if lowercase and value:
        value = value.lower()
    if remove_currency_tokens and value:
        value = _CURRENCY_TOKEN_RE.sub(" ", value)
    if remove_country_tokens and value:
        value = _COUNTRY_TOKEN_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def combine_text_fields(
    parts: Iterable[str],
    *,
    remove_currency_tokens: bool = False,
    remove_country_tokens: bool = False,
    lowercase: bool = True,
) -> str:
    cleaned = [
        normalize_text(
            p,
            remove_currency_tokens=remove_currency_tokens,
            remove_country_tokens=remove_country_tokens,
            lowercase=lowercase,
        )
        for p in parts
    ]
    cleaned = [p for p in cleaned if p]
    return " ".join(cleaned)
