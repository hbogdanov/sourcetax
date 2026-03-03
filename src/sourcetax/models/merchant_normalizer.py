"""
Rule-based merchant normalization.

Cleans raw merchant strings into canonical form.
Handles:
- Whitespace, case, punctuation
- POS/intermediary prefixes (SQ *, TST*, PAYPAL *, AMZN MKTP)
- Phone/store/terminal ids
- Data-driven alias mapping from file
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


# Common single-token prefixes to strip
JUNK_PREFIXES = {
    "SQ",
    "POS",
    "SQUARE",
    "PAYPAL",
    "STRIPE",
    "HELP",
    "SUPPORT",
    "TST",
}

# Common suffixes to strip
JUNK_SUFFIXES = {
    "CA",
    "NY",
    "TX",
    "FL",  # States
    "COM",
    "ORG",
    "NET",  # TLDs
    "HELP",
    "SUPPORT",
    "INFO",
    "PHONE",
    "BILLING",
}

BASE = Path(__file__).resolve().parents[3]
MERCHANT_ALIASES_PATH = BASE / "data" / "mappings" / "merchant_aliases.json"

INTERMEDIARY_PREFIX_RE = re.compile(
    r"^\s*(SQ|TST|PAYPAL|PP|POS|SQUARE)\s*\*?\s+",
    re.IGNORECASE,
)
AMZN_PREFIX_RE = re.compile(
    r"^\s*AMZN\s+(MKTP|MKTPLACE)\b",
    re.IGNORECASE,
)
CARD_AMOUNT_PATTERN = re.compile(r"\b\d{2,4}[-\s]\d{2,4}[-\s]\d{2,4}\b")
STORE_ID_PATTERN = re.compile(r"\b#\d{2,}\b|\b\d{4,}[A-Z]?\b")
STORE_LABEL_PATTERN = re.compile(
    r"\b(STORE|STR|LOCATION|LOC|TERMINAL|TERM|ID)\s*#?\s*\d+\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_merchant_aliases(path: Optional[str] = None) -> Dict[str, str]:
    """Load aliases from data file so alias maintenance does not require code edits."""
    p = Path(path) if path else MERCHANT_ALIASES_PATH
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    aliases: Dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip().upper()
        val = str(v).strip()
        if key and val:
            aliases[key] = val
    return aliases


def clean_merchant_text(merchant_raw: str) -> str:
    """
    Clean raw merchant string.

    Steps:
    1. Normalize known intermediary prefixes
    2. Uppercase
    3. Remove store/terminal/ID patterns
    4. Strip punctuation
    5. Remove junk tokens
    6. Collapse whitespace
    """
    if not merchant_raw:
        return ""

    text = merchant_raw.strip()

    # Normalize common payment intermediaries/prefixes.
    text = INTERMEDIARY_PREFIX_RE.sub("", text)
    text = AMZN_PREFIX_RE.sub("AMZN ", text)

    text = text.upper()

    # Remove store/terminal labels and ids before punctuation clean.
    text = re.sub(STORE_LABEL_PATTERN, " ", text)
    text = re.sub(CARD_AMOUNT_PATTERN, " ", text)
    text = re.sub(STORE_ID_PATTERN, " ", text)

    # Strip punctuation and collapse spacing.
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove known junk
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in JUNK_PREFIXES and t not in JUNK_SUFFIXES]

    # Remove common location suffixes
    while tokens and tokens[-1] in {"CA", "NY", "TX", "FL", "IL", "PA", "OH", "WA"}:
        tokens = tokens[:-1]

    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def apply_aliases(clean_text: str) -> str:
    """Apply merchant aliases to standardize brand names."""
    aliases = load_merchant_aliases()
    for alias, canonical in sorted(aliases.items(), key=lambda x: -len(x[0])):
        if alias in clean_text:
            return canonical
    return clean_text


def extract_root_merchant(clean_text: str, top_n: int = 2) -> str:
    """Extract first N tokens as a root merchant signature."""
    tokens = clean_text.split()[:top_n]
    return " ".join(tokens) if tokens else "UNKNOWN"


def normalize_merchant(merchant_raw: str) -> Tuple[str, str, Optional[str]]:
    """
    Normalize a raw merchant string.

    Returns:
      (merchant_clean, merchant_root, merchant_brand)
    """
    if not merchant_raw:
        return "", "", None

    clean = clean_merchant_text(merchant_raw)
    aliased = apply_aliases(clean)
    root = extract_root_merchant(clean)

    brand = aliased if aliased != clean else None
    return clean, root, brand

