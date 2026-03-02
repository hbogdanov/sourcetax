"""Merchant noise modeling utilities for synthetic and matching datasets."""

from __future__ import annotations

import random
import re
from typing import List, Optional


ACQUIRER_TOKENS = [
    "SQ *",
    "TST*",
    "PAYPAL *",
    "POS ",
    "AMZN MKTP ",
]

LOCATION_TOKENS = [
    "NY NY",
    "SF CA",
    "AUSTIN TX",
    "SEATTLE WA",
    "LOS ANGELES CA",
]


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, rng: random.Random) -> str:
    if len(text) < 8:
        return text
    cut = rng.randint(max(4, len(text) // 2), len(text) - 1)
    return text[:cut].rstrip()


def generate_noisy_merchant_variant(
    canonical_merchant: str,
    *,
    rng: Optional[random.Random] = None,
    acquirer_prob: float = 0.45,
    store_location_prob: float = 0.65,
    id_suffix_prob: float = 0.35,
    truncation_prob: float = 0.15,
) -> str:
    """Generate one noisy merchant_raw variant from a canonical merchant."""
    if not canonical_merchant:
        return ""

    r = rng or random.Random()
    merchant = canonical_merchant.strip().upper()

    if r.random() < acquirer_prob:
        merchant = f"{r.choice(ACQUIRER_TOKENS)}{merchant}"

    if r.random() < store_location_prob:
        location = r.choice(LOCATION_TOKENS)
        merchant = f"{merchant} {location}"

    if r.random() < id_suffix_prob:
        suffix = str(r.randint(100, 99999))
        if r.random() < 0.5:
            merchant = f"{merchant} #{suffix}"
        else:
            merchant = f"{merchant} {suffix}"

    if r.random() < truncation_prob:
        merchant = _truncate(merchant, r)

    return _clean_spaces(merchant)


def generate_noisy_merchant_variants(
    canonical_merchant: str,
    n: int = 5,
    *,
    seed: Optional[int] = None,
    acquirer_prob: float = 0.45,
    store_location_prob: float = 0.65,
    id_suffix_prob: float = 0.35,
    truncation_prob: float = 0.15,
) -> List[str]:
    """Generate up to n unique noisy variants for one canonical merchant."""
    if n <= 0:
        return []
    rng = random.Random(seed)
    seen = set()
    out: List[str] = []
    attempts = 0
    max_attempts = max(20, n * 8)

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        candidate = generate_noisy_merchant_variant(
            canonical_merchant,
            rng=rng,
            acquirer_prob=acquirer_prob,
            store_location_prob=store_location_prob,
            id_suffix_prob=id_suffix_prob,
            truncation_prob=truncation_prob,
        )
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)

    if not out:
        return [canonical_merchant.strip().upper()]
    return out

