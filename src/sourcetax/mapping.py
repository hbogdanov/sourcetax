"""Deterministic mapping helpers for SourceTax v1 category assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BASE = Path(__file__).resolve().parents[2] / "data"
EXTERNAL_MAP_PATH = BASE / "mappings" / "external_category_to_sourcetax_v1.json"
MCC_MAP_PATH = BASE / "mappings" / "mcc_to_sourcetax_v1.json"


DEFAULT_KEYWORD_RULES: Dict[str, str] = {
    # Vehicle
    "SHELL": "Vehicle Expenses",
    "CHEVRON": "Vehicle Expenses",
    "EXXON": "Vehicle Expenses",
    "BP": "Vehicle Expenses",
    "MARATHON": "Vehicle Expenses",
    "QT": "Vehicle Expenses",
    "RACETRAC": "Vehicle Expenses",
    "GAS": "Vehicle Expenses",
    # Travel
    "DELTA": "Travel",
    "UNITED": "Travel",
    "AMERICAN": "Travel",
    "SOUTHWEST": "Travel",
    "AIRBNB": "Travel",
    "MARRIOTT": "Travel",
    "HILTON": "Travel",
    "HYATT": "Travel",
    "HOTEL": "Travel",
    # Meals
    "STARBUCKS": "Meals & Entertainment",
    "MCDONALD": "Meals & Entertainment",
    "DOORDASH": "Meals & Entertainment",
    "UBER EATS": "Meals & Entertainment",
    "RESTAURANT": "Meals & Entertainment",
    # Financial fees
    "ATM FEE": "Financial Fees",
    "OVERDRAFT": "Financial Fees",
    "INTEREST": "Financial Fees",
    "WIRE FEE": "Financial Fees",
    "SERVICE FEE": "Financial Fees",
    # Existing
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


def map_mcc_description(mcc_description: Optional[str]) -> Optional[str]:
    if not mcc_description:
        return None
    mapping = load_json_mapping(MCC_MAP_PATH)
    key = str(mcc_description).strip()
    if not key:
        return None
    direct = mapping.get(key)
    if direct:
        return direct
    return mapping.get(key.upper())


def _keyword_match(text: str, keyword_rules: Dict[str, str]) -> Optional[str]:
    if not text:
        return None
    t = text.upper()
    for keyword, category in keyword_rules.items():
        if keyword in t:
            return category
    return None


def _keyword_match_with_reason(
    text: str, keyword_rules: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    t = text.upper()
    for keyword, category in keyword_rules.items():
        if keyword in t:
            slug = keyword.lower().replace(" ", "_")
            return category, f"keyword:{slug}"
    return None, None


def resolve_category_with_reason(
    *,
    merchant_raw: Optional[str] = None,
    description: Optional[str] = None,
    mcc: Optional[str] = None,
    mcc_description: Optional[str] = None,
    external_category: Optional[str] = None,
    keyword_rules: Optional[Dict[str, str]] = None,
    fallback: str = "Other Expense",
) -> Tuple[str, List[str]]:
    """Resolve SourceTax category and include mapping reasons.

    Precedence:
    1) keyword match in merchant/description
    2) MCC code mapping
    3) MCC description mapping
    4) external category mapping
    5) fallback
    """
    rules = keyword_rules or DEFAULT_KEYWORD_RULES
    text_candidates: Iterable[str] = (
        merchant_raw or "",
        description or "",
    )
    for txt in text_candidates:
        category, reason = _keyword_match_with_reason(txt, rules)
        if category:
            return category, [reason] if reason else ["keyword"]

    mcc_category = map_mcc(mcc)
    if mcc_category:
        return mcc_category, [f"mcc:{str(mcc).strip()}"]

    mcc_desc_category = map_mcc_description(mcc_description)
    if mcc_desc_category:
        desc_slug = str(mcc_description).strip().lower().replace(" ", "_")
        return mcc_desc_category, [f"mcc_description:{desc_slug}"]

    ext_category = map_external_category(external_category)
    if ext_category:
        return ext_category, [f"external:{str(external_category).strip()}"]

    return fallback, [f"fallback:{fallback}"]


def resolve_category_with_precedence(
    *,
    merchant_raw: Optional[str] = None,
    description: Optional[str] = None,
    mcc: Optional[str] = None,
    mcc_description: Optional[str] = None,
    external_category: Optional[str] = None,
    keyword_rules: Optional[Dict[str, str]] = None,
    fallback: str = "Other Expense",
) -> str:
    """Resolve SourceTax category using precedence:
    explicit keyword rules > MCC > external label mapping > fallback.
    """
    category, _ = resolve_category_with_reason(
        merchant_raw=merchant_raw,
        description=description,
        mcc=mcc,
        mcc_description=mcc_description,
        external_category=external_category,
        keyword_rules=keyword_rules,
        fallback=fallback,
    )
    return category
