"""Deterministic mapping helpers for SourceTax v1 category assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

BASE = Path(__file__).resolve().parents[2] / "data"
EXTERNAL_MAP_PATH = BASE / "mappings" / "external_category_to_sourcetax_v1.json"
MCC_MAP_PATH = BASE / "mappings" / "mcc_to_sourcetax_v1.json"


DEFAULT_KEYWORD_RULES: Dict[str, str] = {
    "PAYROLL": "Payroll & Contractors",
    "GUSTO": "Payroll & Contractors",
    "ADP": "Payroll & Contractors",
    "IRS": "Taxes & Licenses",
    "STATE TAX": "Taxes & Licenses",
    "LICENSE": "Taxes & Licenses",
    "INSURANCE": "Insurance",
    "COMCAST": "Rent & Utilities",
    "ELECTRIC": "Rent & Utilities",
    "WATER": "Rent & Utilities",
    "AWS": "Equipment & Software",
    "MICROSOFT": "Equipment & Software",
    "ADOBE": "Equipment & Software",
    "ZOOM": "Equipment & Software",
}


def load_json_mapping(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): str(v) for k, v in data.items()}


def map_external_category(external_category: Optional[str]) -> Optional[str]:
    if not external_category:
        return None
    mapping = load_json_mapping(EXTERNAL_MAP_PATH)
    return mapping.get(external_category.strip())


def map_mcc(mcc: Optional[str]) -> Optional[str]:
    if not mcc:
        return None
    mapping = load_json_mapping(MCC_MAP_PATH)
    return mapping.get(str(mcc).strip())


def _keyword_match(text: str, keyword_rules: Dict[str, str]) -> Optional[str]:
    if not text:
        return None
    t = text.upper()
    for keyword, category in keyword_rules.items():
        if keyword in t:
            return category
    return None


def resolve_category_with_precedence(
    *,
    merchant_raw: Optional[str] = None,
    description: Optional[str] = None,
    mcc: Optional[str] = None,
    external_category: Optional[str] = None,
    keyword_rules: Optional[Dict[str, str]] = None,
    fallback: str = "Other Expense",
) -> str:
    """Resolve SourceTax category using precedence:
    explicit keyword rules > MCC > external label mapping > fallback.
    """
    rules = keyword_rules or DEFAULT_KEYWORD_RULES
    text_candidates: Iterable[str] = (
        merchant_raw or "",
        description or "",
    )
    for txt in text_candidates:
        category = _keyword_match(txt, rules)
        if category:
            return category

    mcc_category = map_mcc(mcc)
    if mcc_category:
        return mcc_category

    ext_category = map_external_category(external_category)
    if ext_category:
        return ext_category

    return fallback

