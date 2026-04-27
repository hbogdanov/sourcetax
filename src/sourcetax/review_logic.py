"""UI-facing review helpers for confidence, exceptions, and ambiguity detection."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .taxonomy import normalize_category_name


LOW_CONFIDENCE_THRESHOLD = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.85

AMBIGUOUS_MERCHANT_TOKENS = {
    "AMAZON": "broad marketplace merchant",
    "WALMART": "broad retail merchant",
    "WHOLE FOODS": "retail/grocery merchant can map multiple ways",
    "SQUARE": "processor or deposit descriptor",
    "STRIPE": "processor or transfer descriptor",
    "PAYPAL": "processor or transfer descriptor",
    "INTUIT PAYMENTS": "processor or transfer descriptor",
    "OPENING BALANCE": "bookkeeping-style placeholder merchant",
    "TRANSFER": "transfer descriptor, not a vendor",
    "DEPOSIT": "deposit descriptor, not a vendor",
}


def confidence_value(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def has_missing_required_fields(record: Dict[str, Any]) -> bool:
    for key in ("transaction_date", "merchant_raw", "amount"):
        value = record.get(key)
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
    return False


def merchant_ambiguity_reason(record: Dict[str, Any]) -> str:
    merchant_raw = str(record.get("merchant_raw") or "").strip()
    merchant_norm = str(record.get("merchant_norm") or "").strip()
    raw_payload = record.get("raw_payload") if isinstance(record.get("raw_payload"), dict) else {}
    merchant_text = f"{merchant_raw} {merchant_norm}".upper()

    if not merchant_raw or not merchant_norm:
        return "missing merchant normalization"

    for token, reason in AMBIGUOUS_MERCHANT_TOKENS.items():
        if token in merchant_text:
            return reason

    rule_category = normalize_category_name(raw_payload.get("rule_category"))
    ml_prediction = normalize_category_name(raw_payload.get("ml_prediction") or raw_payload.get("model_pred"))
    predicted = normalize_category_name(record.get("category_pred"))
    rule_conf = confidence_value(raw_payload.get("rule_confidence"))
    ml_conf = confidence_value(raw_payload.get("ml_confidence"))

    if (
        rule_category
        and ml_prediction
        and rule_category != ml_prediction
        and (rule_conf < HIGH_CONFIDENCE_THRESHOLD or predicted in {"Other Expense", "Uncategorized"})
    ):
        return "rules and ML disagree"

    if predicted in {"Other Expense", "Uncategorized"} and rule_conf < LOW_CONFIDENCE_THRESHOLD:
        return "fell back to generic category"

    if rule_conf < LOW_CONFIDENCE_THRESHOLD and ml_conf < 0.35:
        return "weak signals across rule and ML paths"

    return ""


def build_issue_flags(
    record: Dict[str, Any],
    *,
    has_receipts: bool,
    is_conflict: bool,
    low_conf_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> Dict[str, str]:
    flags: Dict[str, str] = {}
    final_category = str(record.get("category_final") or "").strip()
    confidence = confidence_value(record.get("confidence"))
    source = str(record.get("source") or "").strip().lower()
    matched_transaction_id = str(record.get("matched_transaction_id") or "").strip()

    if not final_category and confidence < low_conf_threshold:
        flags["low_confidence"] = f"confidence {confidence:.0%} is below {low_conf_threshold:.0%}"
    if is_conflict:
        flags["conflict"] = "rules and ML disagree"
    if source == "receipt" and not matched_transaction_id:
        flags["unmatched_receipt"] = "receipt not matched to a transaction"
    if has_receipts and source in {"bank", "toast", "quickbooks"} and not matched_transaction_id:
        flags["unmatched_bank_txn"] = "bank/POS transaction not linked to a receipt"
    ambiguity_reason = merchant_ambiguity_reason(record)
    if ambiguity_reason:
        flags["ambiguous_merchant"] = ambiguity_reason
    if has_missing_required_fields(record):
        flags["missing_fields"] = "one or more required fields are blank"
    return flags


def primary_issue_type(flags: Dict[str, str]) -> Optional[str]:
    for key in (
        "low_confidence",
        "conflict",
        "ambiguous_merchant",
        "missing_fields",
        "unmatched_receipt",
        "unmatched_bank_txn",
    ):
        if key in flags:
            return key
    return None
