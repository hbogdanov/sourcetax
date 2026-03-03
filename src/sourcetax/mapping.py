"""Deterministic mapping helpers for SourceTax v1 category assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BASE = Path(__file__).resolve().parents[2] / "data"
EXTERNAL_MAP_PATH = BASE / "mappings" / "external_category_to_sourcetax_v1.json"
MCC_MAP_PATH = BASE / "mappings" / "mcc_to_sourcetax_v1.json"


DEFAULT_KEYWORD_RULES: Dict[str, str] = {
    # Office / shipping / supplies
    "STAPLES": "Office Supplies",
    "USPS": "Office Supplies",
    "FEDEX": "Office Supplies",
    "KINKO": "Office Supplies",
    # Facilities / maintenance
    "HOME DEPOT": "Repairs & Maintenance",
    "GRAINGER": "Repairs & Maintenance",
    "FRAGERS": "Repairs & Maintenance",
    # Gov / tax / license signals
    "IRS": "Taxes & Licenses",
    "NPDB": "Taxes & Licenses",
    "HIPDB": "Taxes & Licenses",
    "GSA": "Taxes & Licenses",
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
    "UBER": "Travel",
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
    "BANK FEE": "Financial Fees",
    "MONTHLY FEE": "Financial Fees",
    "NSF": "Financial Fees",
    "OVERDRAFT": "Financial Fees",
    "INTEREST": "Financial Fees",
    "WIRE FEE": "Financial Fees",
    "SERVICE FEE": "Financial Fees",
    "TOLL": "Financial Fees",
    # Existing
    "PAYROLL": "Payroll & Contractors",
    "PAYCHEX": "Payroll & Contractors",
    "CONTRACTOR PAY": "Payroll & Contractors",
    "CONTRACTOR PAYMENT": "Payroll & Contractors",
    "1099": "Payroll & Contractors",
    "GUSTO": "Payroll & Contractors",
    "ADP": "Payroll & Contractors",
    "IRS": "Taxes & Licenses",
    "STATE TAX": "Taxes & Licenses",
    "LICENSE": "Taxes & Licenses",
    "INSURANCE": "Insurance",
    "GEICO": "Insurance",
    "PROGRESSIVE": "Insurance",
    "STATE FARM": "Insurance",
    "TRAVELERS": "Insurance",
    "HISCOX": "Insurance",
    "COMCAST": "Rent & Utilities",
    "ELECTRIC BILL": "Rent & Utilities",
    "UTILITY BILL": "Rent & Utilities",
    "WATER": "Rent & Utilities",
    "AWS": "Equipment & Software",
    "MICROSOFT": "Equipment & Software",
    "ADOBE": "Equipment & Software",
    "ZOOM": "Equipment & Software",
}


FINANCIAL_FEES_HIGH_SIGNAL = (
    "INTEREST",
    "OVERDRAFT",
    "NSF",
    "FINANCE CHARGE",
    "LATE FEE",
    "WIRE FEE",
    "ACH RETURN FEE",
    "RETURNED PAYMENT",
    "CHARGEBACK",
)

FINANCIAL_FEES_MEDIUM_SIGNAL = (
    "FEE",
    "SERVICE CHARGE",
    "MONTHLY MAINTENANCE",
    "PROCESSING FEE",
    "MERCHANT FEE",
    "CREDIT CARD FEE",
    "BANK FEE",
    "ACH FEE",
    "SERVICE CHARGES",
    "PROCESSOR SETTLEMENT FEE",
    "PAYMENT PROCESSOR FEES",
    "TOLL",
)

PAYMENT_PROCESSOR_HINTS = ("STRIPE", "SQUARE", "PAYPAL", "CLOVER")

RPM_HIGH_SIGNAL = (
    "ELECTRIC SERVICE",
    "ELECTRICAL SUP",
    "ELECTRIC SUPP",
    "ELECTRIC SUPLY",
    "PLUMBING",
    "HVAC",
    "JANITORIAL",
    "OVERHEAD DOOR",
    "WELDING",
    "HYDRAULIC",
    "APPLIANCE REPAIR",
    "EQUIPMENT REPAIR",
    "SERVICE CORP",
    "ALARM SERVICE",
    "TRULAND",
    "FERGUSON",
    "RE MICHEL",
    "ADCOCK",
    "HANDYMAN",
    "PEST CONTROL",
    "LANDSCAPING",
)

RPM_MEDIUM_SIGNAL = (
    "REPAIR",
    "MAINTENANCE",
    "SERVICE CALL",
    "DOOR CO",
    "TECHNICAL SOLUTION",
    "CANONBUS",
    "DMS ENTERPRISES",
    "CONTRACTORS",
)

RPM_MCC_SIGNAL = (
    "ELECTRICAL CONTRACTORS",
    "GENL CONTRACTORS-RESIDENTIAL, AND COMMERCIAL",
    "SPECIAL TRADE CONTRACTORS,NOT ELSEWHERE CLASSIFIED",
    "COMPUTER MAINT&REPAIR SERVICE,NOT ELSEWHERE CLASS.",
    "REPAIR SHOPS AND RELATED SERVICES - MISCELLANEOUS",
    "ELECTRICAL AND SMALL APPLIANCE REPAIR SHOPS",
    "ELECTRONICS REPAIR SHOPS",
    "WELDING REPAIR",
    "CLEANING & MAINTENANCE, JANITORIAL SERVICES",
    "PLUMBING AND HEATING EQUIPMENT AND SUPPLIES",
    "ELECTRICAL PARTS AND EQUIPMENT",
)


BUSINESS_HIGH_SIGNAL: Dict[str, Tuple[str, str]] = {
    # Equipment & Software
    "ADOBE": ("Equipment & Software", "eqs"),
    "MICROSOFT": ("Equipment & Software", "eqs"),
    "GOOGLE WORKSPACE": ("Equipment & Software", "eqs"),
    "AWS": ("Equipment & Software", "eqs"),
    "GITHUB": ("Equipment & Software", "eqs"),
    "ATLASSIAN": ("Equipment & Software", "eqs"),
    "SLACK": ("Equipment & Software", "eqs"),
    "NOTION": ("Equipment & Software", "eqs"),
    "ZOOM": ("Equipment & Software", "eqs"),
    "DROPBOX": ("Equipment & Software", "eqs"),
    "OPENAI": ("Equipment & Software", "eqs"),
    "NVIDIA": ("Equipment & Software", "eqs"),
    "AUTODESK": ("Equipment & Software", "eqs"),
    "APPLEONLINESTORE": ("Equipment & Software", "eqs"),
    "BEST BUY": ("Equipment & Software", "eqs"),
    "DUPONT COMPUTER": ("Equipment & Software", "eqs"),
    "COMPUTER INC": ("Equipment & Software", "eqs"),
    # Office Supplies
    "OFFICE DEPOT": ("Office Supplies", "off"),
    "OFFICEMAX": ("Office Supplies", "off"),
    "ULINE": ("Office Supplies", "off"),
    "BRANCH SUPPLY": ("Office Supplies", "off"),
    "METROPOLITAN OFFICE": ("Office Supplies", "off"),
    "LABSAFE": ("Office Supplies", "off"),
    # Advertising & Marketing
    "GOOGLE ADS": ("Advertising & Marketing", "mkt"),
    "META ADS": ("Advertising & Marketing", "mkt"),
    "FACEBOOK ADS": ("Advertising & Marketing", "mkt"),
    "LINKEDIN ADS": ("Advertising & Marketing", "mkt"),
    "TIKTOK ADS": ("Advertising & Marketing", "mkt"),
    "X ADS": ("Advertising & Marketing", "mkt"),
    "MAILCHIMP": ("Advertising & Marketing", "mkt"),
    "HUBSPOT": ("Advertising & Marketing", "mkt"),
    "HOOTSUITE": ("Advertising & Marketing", "mkt"),
    "PRESSTEK": ("Advertising & Marketing", "mkt"),
    "BRENTWORKS": ("Advertising & Marketing", "mkt"),
    # Professional Services
    "CONSULTING": ("Professional Services", "pro"),
    "LEGAL": ("Professional Services", "pro"),
    "ACCOUNTING": ("Professional Services", "pro"),
    "CPA": ("Professional Services", "pro"),
    "ATTORNEY": ("Professional Services", "pro"),
    "SHRM": ("Professional Services", "pro"),
    "ABLE REPORTING": ("Professional Services", "pro"),
    "DCBIA": ("Professional Services", "pro"),
    # COGS
    "WHOLESALE": ("COGS", "cogs"),
    "RAW MATERIAL": ("COGS", "cogs"),
    "INVENTORY": ("COGS", "cogs"),
    "INDUSTRIAL SUPPLIES": ("COGS", "cogs"),
}

BUSINESS_MEDIUM_SIGNAL: Dict[str, Tuple[str, str]] = {
    # Equipment & Software
    "SUBSCRIPTION": ("Equipment & Software", "eqs"),
    "LICENSE": ("Equipment & Software", "eqs"),
    "SAAS": ("Equipment & Software", "eqs"),
    "CLOUD": ("Equipment & Software", "eqs"),
    "HOSTING": ("Equipment & Software", "eqs"),
    "DOMAIN": ("Equipment & Software", "eqs"),
    "SOFTWARE": ("Equipment & Software", "eqs"),
    "COMPUTER HARDWARE": ("Equipment & Software", "eqs"),
    # Advertising & Marketing
    "PRINTING": ("Advertising & Marketing", "mkt"),
    "SIGN": ("Advertising & Marketing", "mkt"),
    "PROMO": ("Advertising & Marketing", "mkt"),
    "MARKETING": ("Advertising & Marketing", "mkt"),
    "ADVERTISING": ("Advertising & Marketing", "mkt"),
    "SEO": ("Advertising & Marketing", "mkt"),
    # Professional Services
    "BOOKKEEPING": ("Professional Services", "pro"),
    "AUDIT": ("Professional Services", "pro"),
    # COGS
    "MATERIALS": ("COGS", "cogs"),
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


def _financial_fee_match(
    *,
    merchant_raw: Optional[str],
    description: Optional[str],
    mcc_description: Optional[str],
    amount: Optional[float],
) -> Tuple[Optional[str], Optional[str]]:
    parts = [
        str(merchant_raw or ""),
        str(description or ""),
        str(mcc_description or ""),
    ]
    text = " ".join(parts).upper()
    if not text.strip():
        return None, None

    for token in FINANCIAL_FEES_HIGH_SIGNAL:
        if token in text:
            return "Financial Fees", f"financial_high:{token.lower().replace(' ', '_')}"

    for token in FINANCIAL_FEES_MEDIUM_SIGNAL:
        if token in text:
            return "Financial Fees", f"financial_medium:{token.lower().replace(' ', '_')}"

    if amount is not None:
        try:
            amt = abs(float(amount))
        except Exception:
            amt = None
        if amt is not None and amt < 50:
            for proc in PAYMENT_PROCESSOR_HINTS:
                if proc in text:
                    return "Financial Fees", f"financial_medium:{proc.lower()}_small_amount"

    return None, None


def _business_category_match(
    *,
    merchant_raw: Optional[str],
    description: Optional[str],
    mcc_description: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join(
        [
            str(merchant_raw or ""),
            str(description or ""),
            str(mcc_description or ""),
        ]
    ).upper()
    if not text.strip():
        return None, None

    # Keep Amazon/Walmart ambiguous by default unless strong lexical signal exists.
    if ("AMAZON" in text or "WALMART" in text) and not any(
        token in text for token in ("SUBSCRIPTION", "LICENSE", "SOFTWARE", "CLOUD", "HOSTING", "DOMAIN")
    ):
        return None, None

    for token, (category, family) in BUSINESS_HIGH_SIGNAL.items():
        if token in text:
            return category, f"{family}_high:{token.lower().replace(' ', '_')}"

    for token, (category, family) in BUSINESS_MEDIUM_SIGNAL.items():
        if token in text:
            return category, f"{family}_medium:{token.lower().replace(' ', '_')}"

    return None, None


def _repairs_maintenance_match(
    *,
    merchant_raw: Optional[str],
    description: Optional[str],
    mcc_description: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join(
        [
            str(merchant_raw or ""),
            str(description or ""),
            str(mcc_description or ""),
        ]
    ).upper()
    if not text.strip():
        return None, None

    for token in RPM_MCC_SIGNAL:
        if token in text:
            return "Repairs & Maintenance", f"rpm_mcc:{token.lower().replace(' ', '_')}"

    if "ELECTRIC" in text and any(t in text for t in ("SERVICE", "SUPP", "SUPLY", "LLC", "CONTRACT", "ALARM")):
        return "Repairs & Maintenance", "rpm_high:electric_service_context"

    for token in RPM_HIGH_SIGNAL:
        if token in text:
            return "Repairs & Maintenance", f"rpm_high:{token.lower().replace(' ', '_')}"

    for token in RPM_MEDIUM_SIGNAL:
        if token in text:
            return "Repairs & Maintenance", f"rpm_medium:{token.lower().replace(' ', '_')}"

    return None, None


def resolve_category_with_reason(
    *,
    merchant_raw: Optional[str] = None,
    description: Optional[str] = None,
    mcc: Optional[str] = None,
    mcc_description: Optional[str] = None,
    external_category: Optional[str] = None,
    amount: Optional[float] = None,
    keyword_rules: Optional[Dict[str, str]] = None,
    fallback: str = "Other Expense",
) -> Tuple[str, List[str]]:
    """Resolve SourceTax category and include mapping reasons.

    Precedence:
    1) keyword match in merchant/description
    2) explicit financial-fee lexical matching
    3) business lexical disambiguation (EQS/OFF/MKT/PRO/COGS)
    4) MCC code mapping
    5) MCC description mapping
    6) external category mapping
    7) fallback
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

    rpm_category, rpm_reason = _repairs_maintenance_match(
        merchant_raw=merchant_raw,
        description=description,
        mcc_description=mcc_description,
    )
    if rpm_category:
        return rpm_category, [rpm_reason] if rpm_reason else ["rpm_lexical"]

    financial_category, financial_reason = _financial_fee_match(
        merchant_raw=merchant_raw,
        description=description,
        mcc_description=mcc_description,
        amount=amount,
    )
    if financial_category:
        return financial_category, [financial_reason] if financial_reason else ["financial_lexical"]

    business_category, business_reason = _business_category_match(
        merchant_raw=merchant_raw,
        description=description,
        mcc_description=mcc_description,
    )
    if business_category:
        return business_category, [business_reason] if business_reason else ["business_lexical"]

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
    amount: Optional[float] = None,
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
        amount=amount,
        keyword_rules=keyword_rules,
        fallback=fallback,
    )
    return category
