"""Shared enriched ML feature builder + artifact IO for shadow inference."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sourcetax import mapping, taxonomy
from sourcetax.normalization import normalize_merchant_name

ARTIFACT_DIR = Path("artifacts")
PIPELINE_PATH = ARTIFACT_DIR / "ml_enriched_tfidf_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "ml_enriched_metadata.json"
FEATURE_BUILDER_VERSION = "enriched_v1"


def _safe_json_loads(value: Any, default: Any) -> Any:
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


def taxonomy_hash() -> str:
    p = Path("data/taxonomy/sourcetax_v1.json")
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return (result.stdout or "").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def extract_context(rec: Dict[str, Any]) -> Dict[str, Any]:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "merchant_raw": str(rec.get("merchant_raw") or "").strip(),
        "description": str(rec.get("description") or raw.get("description") or raw.get("ocr_text") or "").strip(),
        "mcc": str(rec.get("mcc") or raw.get("mcc") or "").strip(),
        "mcc_description": str(rec.get("mcc_description") or raw.get("mcc_description") or "").strip(),
        "category_external": str(rec.get("category_external") or raw.get("category_external") or "").strip(),
        "amount": rec.get("amount"),
    }


def build_enriched_text(
    *,
    merchant_raw: str,
    description: str = "",
    mcc: str = "",
    mcc_description: str = "",
    category_external: str = "",
    amount: Optional[float] = None,
) -> Tuple[str, list[str]]:
    merchant_norm = normalize_merchant_name(merchant_raw or "", case="lower") if merchant_raw else ""
    _, reasons = mapping.resolve_category_with_reason(
        merchant_raw=merchant_raw or None,
        description=description or None,
        mcc=mcc or None,
        mcc_description=mcc_description or None,
        external_category=category_external or None,
        amount=amount,
        fallback="Other Expense",
    )
    reason_tokens = " ".join(str(x).replace(":", " ").replace("_", " ") for x in (reasons or []))
    missing_mcc = "missing_mcc" if not mcc_description else "has_mcc"
    amount_token = ""
    try:
        if amount is not None:
            a = abs(float(amount))
            if a < 25:
                amount_token = "amt_small"
            elif a < 250:
                amount_token = "amt_medium"
            else:
                amount_token = "amt_large"
            if float(amount) > 0:
                amount_token += " is_refund"
    except Exception:
        pass
    parts = [merchant_raw, merchant_norm, description, mcc_description, reason_tokens, missing_mcc, amount_token]
    text = " ".join(p for p in parts if p).strip()
    return text, reasons or []


def expected_categories() -> list[str]:
    return taxonomy.load_sourcetax_categories(include_uncategorized=False)


def validate_pipeline(pipeline: Any) -> Tuple[bool, str]:
    try:
        from sklearn.pipeline import Pipeline  # type: ignore
    except Exception:
        Pipeline = object  # type: ignore

    if pipeline is None:
        return False, "MODEL_MISSING"
    if not isinstance(pipeline, Pipeline):
        return False, "MODEL_NOT_PIPELINE"
    tfidf = pipeline.named_steps.get("tfidf") if hasattr(pipeline, "named_steps") else None
    clf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("classifier") if hasattr(pipeline, "named_steps") else None
    if tfidf is None or clf is None:
        return False, "MODEL_MISSING_STEPS"
    vocab = getattr(tfidf, "vocabulary_", None)
    if not isinstance(vocab, dict) or len(vocab) < 100:
        return False, "MODEL_WEAK_VOCAB"
    raw_classes = getattr(clf, "classes_", None)
    if raw_classes is None:
        classes = []
    else:
        classes = list(raw_classes)
    if not classes:
        return False, "MODEL_NO_CLASSES"
    expected = set(expected_categories())
    got = set(str(c) for c in classes)
    if not expected.issubset(got):
        return False, "MODEL_CLASS_SPACE_MISMATCH"
    return True, "OK"


def load_artifacts() -> Tuple[Optional[Any], Dict[str, Any], str]:
    metadata: Dict[str, Any] = {}
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    if not PIPELINE_PATH.exists():
        return None, metadata, "MODEL_MISSING"
    try:
        import joblib  # type: ignore

        pipeline = joblib.load(PIPELINE_PATH)
    except Exception as exc:
        return None, metadata, f"MODEL_LOAD_ERROR:{exc}"
    ok, status = validate_pipeline(pipeline)
    if not ok:
        return None, metadata, status
    return pipeline, metadata, "OK"


def artifact_metadata(
    *,
    train_rows: int,
    natural_rows: int,
    synthetic_rows: int,
    class_counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "taxonomy_hash": taxonomy_hash(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "train_rows": int(train_rows),
        "natural_rows": int(natural_rows),
        "synthetic_rows": int(synthetic_rows),
        "class_counts": class_counts,
        "pipeline_path": str(PIPELINE_PATH),
    }
