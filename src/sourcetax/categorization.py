"""
Phase 2.3: Rules-based categorization + learned overrides.

Start with rules, no ML. Learn from user overrides.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import csv
import json
from datetime import datetime, timezone
from functools import lru_cache
from .normalization import normalize_merchant_name
from .taxonomy import normalize_category_name, require_valid_category
from .gold import normalize_label_confidence, normalize_label_notes
from . import mapping
from . import shadow_ml


# Keyword rules: merchant substring → (category, confidence)
KEYWORD_RULES = {
    "HOME DEPOT": ("Office Supplies", 0.6),
    "LOWES": ("Office Supplies", 0.6),
    "AMAZON": ("Office Supplies", 0.6),
    "OFFICE DEPOT": ("Office Supplies", 0.65),
    "STAPLES": ("Office Supplies", 0.65),
    "UBER": ("Travel", 0.7),
    "LYFT": ("Travel", 0.7),
    "GAS STATION": ("Travel", 0.6),
    "SHELL": ("Travel", 0.6),
    "CHEVRON": ("Travel", 0.6),
    "STARBUCKS": ("Meals & Entertainment", 0.6),
    "CAFE": ("Meals & Entertainment", 0.55),
    "RESTAURANT": ("Meals & Entertainment", 0.65),
    "PIZZA": ("Meals & Entertainment", 0.7),
    "HOTEL": ("Travel", 0.75),
    "MOTEL": ("Travel", 0.75),
    "AIRBNB": ("Travel", 0.75),
    "UTILITY": ("Rent & Utilities", 0.8),
    "POWER": ("Rent & Utilities", 0.7),
    "WATER": ("Rent & Utilities", 0.7),
    "INTERNET": ("Rent & Utilities", 0.75),
    "PHONE": ("Rent & Utilities", 0.7),
    "RENT": ("Rent & Utilities", 0.9),
    "APARTMENT": ("Rent & Utilities", 0.6),
    "INSURANCE": ("Insurance", 0.85),
    "TAX": ("Taxes & Licenses", 0.85),
    "PERMIT": ("Taxes & Licenses", 0.8),
    "LICENSE": ("Taxes & Licenses", 0.8),
}

HYBRID_THRESHOLD = 0.85
PROD_RULE_CONF_THRESHOLD = 0.95
PROD_ML_CONF_THRESHOLD = 0.30


def load_merchant_category_map(path: str = "data/mappings/merchant_category.csv") -> Dict[str, str]:
    """Load exact merchant → category mapping from CSV.
    
    CSV columns: merchant, category_code, category_name, notes
    Returns dict mapping merchant (uppercase) → category_name
    """
    mapping = {}
    p = Path(path)
    if not p.exists():
        return mapping
    
    try:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merchant = normalize_merchant_name(row.get("merchant", ""), case="upper")
                # Read category_name (human-readable) for exports
                category = normalize_category_name(row.get("category_name", "").strip())
                if merchant and category:
                    mapping[merchant] = category
    except Exception as e:
        print(f"Error loading merchant map: {e}")
    
    return mapping


def categorize_by_merchant_exact(merchant: str, merchant_map: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Try exact merchant match (highest confidence)."""
    if not merchant:
        return None
    
    merchant_norm = normalize_merchant_name(merchant, case="upper")
    category = merchant_map.get(merchant_norm)
    
    if category:
        return (category, 0.95)
    
    return None


def categorize_by_merchant_fuzzy(merchant: str, merchant_map: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Try fuzzy merchant match (medium confidence)."""
    if not merchant:
        return None
    
    merchant_norm = normalize_merchant_name(merchant, case="upper")
    
    # Check if merchant is substring of known merchant
    for known_merchant, category in merchant_map.items():
        if merchant_norm in known_merchant or known_merchant in merchant_norm:
            return (category, 0.75)
    
    return None


def categorize_by_keywords(merchant: str) -> Optional[Tuple[str, float]]:
    """Try keyword match (lower confidence)."""
    if not merchant:
        return None
    
    merchant_upper = normalize_merchant_name(merchant, case="upper") or merchant.upper()
    
    for keyword, (category, confidence) in KEYWORD_RULES.items():
        if keyword in merchant_upper:
            return (category, confidence)
    
    return None


def get_learned_override(merchant: str, db_path: str = "data/store.db") -> Optional[Tuple[str, float]]:
    """Check if user has overridden category for this merchant (highest priority)."""
    if not merchant:
        return None
    
    merchant_raw_key = normalize_merchant_name(merchant, case="upper")
    merchant_norm_key = normalize_merchant_name(merchant, case="lower")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT category_final FROM canonical_records
            WHERE category_final IS NOT NULL
              AND (
                    merchant_norm = ?
                    OR UPPER(TRIM(merchant_raw)) = ?
                  )
            LIMIT 1
            """,
            (merchant_norm_key, merchant_raw_key),
        )
        result = cur.fetchone()
    finally:
        conn.close()

    if result:
        category = normalize_category_name(result[0])
        if category:
            return (category, 0.99)  # Learned overrides have highest confidence

    return None


def categorize_record(record_id: str, db_path: str = "data/store.db") -> Tuple[Optional[str], float]:
    """
    Auto-categorize a record using rules priority:
    1. Learned override (user has set this merchant before)
    2. Exact merchant match
    3. Fuzzy merchant match
    4. Keyword match
    5. Other Expense
    
    Returns: (category, confidence)
    """
    # Fetch record
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT merchant_raw, category_pred FROM canonical_records WHERE id = ?",
        (record_id,),
    )
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return "Other Expense", 0.0
    
    merchant = row[0]
    
    # Load merchant map
    merchant_map = load_merchant_category_map()
    
    # Try rules in order
    result = get_learned_override(merchant, db_path)
    if result:
        return result
    
    result = categorize_by_merchant_exact(merchant, merchant_map)
    if result:
        category = normalize_category_name(result[0]) or "Other Expense"
        return category, result[1]
    
    result = categorize_by_merchant_fuzzy(merchant, merchant_map)
    if result:
        category = normalize_category_name(result[0]) or "Other Expense"
        return category, result[1]
    
    result = categorize_by_keywords(merchant)
    if result:
        category = normalize_category_name(result[0]) or "Other Expense"
        return category, result[1]

    return "Other Expense", 0.3


@lru_cache(maxsize=1)
def _load_ml_pipeline_cached():
    pipeline, metadata, status = shadow_ml.load_artifacts()
    return {"pipeline": pipeline, "metadata": metadata, "status": status}


def _safe_json_loads(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _extract_context_fields(raw_payload: dict) -> dict:
    return {
        "description": str(raw_payload.get("description") or raw_payload.get("ocr_text") or "").strip(),
        "mcc": str(raw_payload.get("mcc") or "").strip(),
        "mcc_description": str(raw_payload.get("mcc_description") or "").strip(),
        "category_external": str(raw_payload.get("category_external") or "").strip(),
    }


def _predict_ml_category(
    merchant_raw: str,
    raw_payload: dict,
    *,
    amount: Optional[float] = None,
) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
    loaded = _load_ml_pipeline_cached()
    pipeline = loaded.get("pipeline")
    status = loaded.get("status", "MODEL_UNKNOWN")
    metadata = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else {}
    if pipeline is None:
        return None, None, {"ml_model_status": status, "ml_feature_text_len": 0, "ml_feature_nnz": 0}
    try:
        ctx = _extract_context_fields(raw_payload)
        text, reasons = shadow_ml.build_enriched_text(
            merchant_raw=merchant_raw or "",
            description=ctx.get("description") or "",
            mcc=ctx.get("mcc") or "",
            mcc_description=ctx.get("mcc_description") or "",
            category_external=ctx.get("category_external") or "",
            amount=amount,
        )
        if not text:
            return None, None, {
                "ml_model_status": status,
                "ml_feature_text_len": 0,
                "ml_feature_nnz": 0,
                "ml_feature_reason_count": len(reasons),
            }
        tfidf = pipeline.named_steps.get("tfidf") if hasattr(pipeline, "named_steps") else None
        X = tfidf.transform([text]) if tfidf is not None else None
        nnz = int(X.nnz) if X is not None else 0
        pred = pipeline.predict([text])[0]
        pred_norm = normalize_category_name(pred)
        if not pred_norm:
            return None, None, {
                "ml_model_status": "MODEL_PRED_INVALID_LABEL",
                "ml_feature_text_len": len(text),
                "ml_feature_nnz": nnz,
                "ml_feature_reason_count": len(reasons),
            }
        conf = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([text])[0]
            conf = float(max(proba)) if len(proba) else None
        return pred_norm, conf, {
            "ml_model_status": status,
            "ml_feature_text_len": len(text),
            "ml_feature_nnz": nnz,
            "ml_feature_reason_count": len(reasons),
            "ml_feature_preview": text[:180],
            "ml_feature_builder_version": shadow_ml.FEATURE_BUILDER_VERSION,
            "ml_model_taxonomy_hash": metadata.get("taxonomy_hash"),
            "ml_model_trained_at_utc": metadata.get("trained_at_utc"),
            "ml_model_git_commit": metadata.get("git_commit"),
        }
    except Exception as exc:
        return None, None, {"ml_model_status": f"MODEL_PREDICT_ERROR:{exc}", "ml_feature_text_len": 0, "ml_feature_nnz": 0}


def _mapping_reason_confidence(reasons: list[str]) -> float:
    if not reasons:
        return 0.3
    first = reasons[0]
    if first.startswith("financial_high:") or "_high:" in first or first.startswith("keyword:"):
        return 0.9
    if first.startswith("financial_medium:") or "_medium:" in first:
        return 0.7
    if first.startswith("mcc:") or first.startswith("mcc_description:"):
        return 0.85
    if first.startswith("external:"):
        return 0.7
    return 0.3


def build_shadow_decisions(
    *,
    merchant_raw: str,
    amount: Optional[float],
    description: str = "",
    mcc: str = "",
    mcc_description: str = "",
    category_external: str = "",
    raw_payload: Optional[dict] = None,
    threshold_primary: float = 0.85,
    threshold_alt: float = 0.70,
) -> dict:
    """Return rule/ml/hybrid predictions for shadow mode logging."""
    raw_payload = raw_payload if isinstance(raw_payload, dict) else {}
    rule_category, reasons = mapping.resolve_category_with_reason(
        merchant_raw=merchant_raw or None,
        description=description or None,
        mcc=mcc or None,
        mcc_description=mcc_description or None,
        external_category=category_external or None,
        amount=amount,
        fallback="Other Expense",
    )
    rule_category = normalize_category_name(rule_category) or "Other Expense"
    rule_confidence = _mapping_reason_confidence(reasons)

    payload_for_ml = dict(raw_payload)
    if description:
        payload_for_ml.setdefault("description", description)
    if mcc:
        payload_for_ml.setdefault("mcc", mcc)
    if mcc_description:
        payload_for_ml.setdefault("mcc_description", mcc_description)
    if category_external:
        payload_for_ml.setdefault("category_external", category_external)

    ml_prediction, ml_confidence, ml_diag = _predict_ml_category(
        merchant_raw or "",
        payload_for_ml,
        amount=amount,
    )

    hybrid_t85 = rule_category if rule_confidence >= threshold_primary else (ml_prediction or rule_category)
    hybrid_t70 = rule_category if rule_confidence >= threshold_alt else (ml_prediction or rule_category)
    prod_candidate_use_ml = bool(
        ml_prediction
        and ml_prediction != rule_category
        and rule_confidence < PROD_RULE_CONF_THRESHOLD
        and float(ml_confidence or 0.0) >= PROD_ML_CONF_THRESHOLD
    )
    hybrid_prod_candidate = ml_prediction if prod_candidate_use_ml else rule_category

    return {
        "rule_category": rule_category,
        "rule_confidence": float(rule_confidence),
        "rule_reason": reasons,
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,
        "hybrid_prediction": hybrid_t85,
        "hybrid_prediction_t85": hybrid_t85,
        "hybrid_prediction_t70": hybrid_t70,
        "hybrid_prediction_prod_candidate": hybrid_prod_candidate,
        "hybrid_prod_rule_conf_threshold": float(PROD_RULE_CONF_THRESHOLD),
        "hybrid_prod_ml_conf_threshold": float(PROD_ML_CONF_THRESHOLD),
        "hybrid_prod_candidate_eligible_flip": bool(prod_candidate_use_ml),
        "hybrid_threshold": float(threshold_primary),
        "final_category": rule_category,
        "shadow_mode": True,
        "shadow_logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **(ml_diag if isinstance(ml_diag, dict) else {}),
    }


def categorize_record_shadow(record_id: str, db_path: str = "data/store.db") -> Optional[dict]:
    """Compute rule/ml/hybrid outputs and return shadow logging payload.

    Production final remains rules-only for now.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT merchant_raw, amount, raw_payload FROM canonical_records WHERE id = ?",
        (record_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None

    merchant_raw = row[0] or ""
    amount = row[1]
    raw_payload = _safe_json_loads(row[2], {})
    if not isinstance(raw_payload, dict):
        raw_payload = {}
    ctx = _extract_context_fields(raw_payload)
    return build_shadow_decisions(
        merchant_raw=merchant_raw,
        amount=amount if amount is not None else raw_payload.get("amount"),
        description=ctx.get("description") or "",
        mcc=ctx.get("mcc") or "",
        mcc_description=ctx.get("mcc_description") or "",
        category_external=ctx.get("category_external") or "",
        raw_payload=raw_payload,
        threshold_primary=HYBRID_THRESHOLD,
        threshold_alt=0.70,
    )


def log_shadow_inference(
    record_id: str,
    db_path: str = "data/store.db",
    *,
    overwrite_category_pred: bool = False,
) -> bool:
    """Write shadow inference fields into raw_payload for one record.

    If overwrite_category_pred=True, category_pred/confidence are updated to rule output.
    Otherwise, existing category_pred/confidence remain unchanged.
    """
    shadow = categorize_record_shadow(record_id, db_path=db_path)
    if not shadow:
        return False

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT raw_payload FROM canonical_records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    raw_payload = _safe_json_loads(row[0], {}) if row else {}
    if not isinstance(raw_payload, dict):
        raw_payload = {}
    raw_payload.update(
        {
            "rule_category": shadow.get("rule_category"),
            "rule_confidence": shadow.get("rule_confidence"),
            "rule_reason": shadow.get("rule_reason"),
            "ml_prediction": shadow.get("ml_prediction"),
            "ml_confidence": shadow.get("ml_confidence"),
            "hybrid_prediction": shadow.get("hybrid_prediction"),
            "hybrid_prediction_t85": shadow.get("hybrid_prediction_t85"),
            "hybrid_prediction_t70": shadow.get("hybrid_prediction_t70"),
            "hybrid_threshold": shadow.get("hybrid_threshold"),
            "final_category": shadow.get("final_category"),
            "shadow_mode": True,
            "shadow_logged_at_utc": shadow.get("shadow_logged_at_utc"),
            "ml_model_status": shadow.get("ml_model_status"),
            "ml_feature_text_len": shadow.get("ml_feature_text_len"),
            "ml_feature_nnz": shadow.get("ml_feature_nnz"),
            "ml_feature_reason_count": shadow.get("ml_feature_reason_count"),
            "ml_feature_builder_version": shadow.get("ml_feature_builder_version"),
            "ml_model_taxonomy_hash": shadow.get("ml_model_taxonomy_hash"),
            "ml_model_trained_at_utc": shadow.get("ml_model_trained_at_utc"),
            "ml_model_git_commit": shadow.get("ml_model_git_commit"),
        }
    )

    if overwrite_category_pred:
        cur.execute(
            "UPDATE canonical_records SET category_pred = ?, confidence = ?, raw_payload = ? WHERE id = ?",
            (
                shadow.get("final_category"),
                float(shadow.get("rule_confidence") or 0.0),
                json.dumps(raw_payload, ensure_ascii=False),
                record_id,
            ),
        )
    else:
        cur.execute(
            "UPDATE canonical_records SET raw_payload = ? WHERE id = ?",
            (json.dumps(raw_payload, ensure_ascii=False), record_id),
        )
    conn.commit()
    conn.close()
    return True


def categorize_all_records(db_path: str = "data/store.db") -> int:
    """
    Auto-categorize all records without a category_pred.
    Updates category_pred field.
    
    Returns: count of categorized records.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Fetch all records without a category
    cur.execute(
        "SELECT id FROM canonical_records WHERE category_pred IS NULL OR category_pred = 'Uncategorized'"
    )
    records = [row[0] for row in cur.fetchall()]
    
    conn.close()
    
    categorized_count = 0
    for record_id in records:
        ok = log_shadow_inference(record_id, db_path=db_path, overwrite_category_pred=True)
        if not ok:
            continue
        
        categorized_count += 1
    
    return categorized_count


def save_category_override(
    record_id: str,
    category: str,
    db_path: str = "data/store.db",
    label_confidence: str = "medium",
    label_notes: str = "",
):
    """Save user's category override. This creates a learned rule for future matching."""
    category = require_valid_category(category, field_name="category_final")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT raw_payload FROM canonical_records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    raw_payload = {}
    if row and row[0]:
        try:
            parsed = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            if isinstance(parsed, dict):
                raw_payload = parsed
        except Exception:
            raw_payload = {}
    raw_payload["label_source"] = "human"
    raw_payload["labeled_at_utc"] = datetime.now(timezone.utc).isoformat()
    raw_payload["label_confidence"] = normalize_label_confidence(label_confidence)
    raw_payload["label_notes"] = normalize_label_notes(label_notes)

    cur.execute(
        "UPDATE canonical_records SET category_final = ?, raw_payload = ? WHERE id = ?",
        (category, json.dumps(raw_payload, ensure_ascii=False), record_id),
    )

    conn.commit()
    conn.close()
