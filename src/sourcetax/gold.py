"""Gold dataset enforcement helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return default
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def get_label_source(record: Dict[str, Any]) -> str:
    raw_payload = _safe_json_loads(record.get("raw_payload"), {})
    if not isinstance(raw_payload, dict):
        raw_payload = {}
    source = (
        str(record.get("label_source") or raw_payload.get("label_source") or "")
        .strip()
        .lower()
    )
    return source


def is_human_labeled_gold_record(record: Dict[str, Any]) -> bool:
    category = str(record.get("category_final") or "").strip()
    if not category:
        return False
    return get_label_source(record) == "human"


def filter_human_labeled_gold(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    skipped = 0
    for rec in records:
        if is_human_labeled_gold_record(rec):
            out.append(rec)
        else:
            skipped += 1
    return out, skipped

