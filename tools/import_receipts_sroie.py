#!/usr/bin/env python
"""Import SROIE (HF) receipt corpus into staging_receipts.

Target mapping:
- source='sroie'
- ocr_text reconstructed from words/entities
- total from entities.total (parsed)
- merchant_raw from entities.company (fallback to first OCR line)
- full original annotation retained in raw_payload_json (JSON-safe normalized)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sourcetax import staging


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value]
    # PIL images and other objects -> metadata-ish string
    return str(value)


def _extract_words(example: Dict[str, Any]) -> List[str]:
    words = example.get("words")
    if isinstance(words, list):
        return [str(w).strip() for w in words if str(w).strip()]
    return []


def _extract_entities(example: Dict[str, Any]) -> Dict[str, Any]:
    entities = example.get("entities")
    if isinstance(entities, dict):
        return entities
    return {}


def _extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = str(text)
    matches = re.findall(r"[-+]?\d[\d,]*\.?\d*", s)
    if not matches:
        return None
    # Use the last numeric token, typically the final total value.
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except Exception:
        return None


def _reconstruct_ocr_text(words: Iterable[str], entities: Dict[str, Any]) -> str:
    parts = []
    words_list = [w for w in words if w]
    if words_list:
        parts.append(" ".join(words_list))
    if entities:
        for key in ["company", "date", "address", "total"]:
            v = entities.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(f"{key.upper()}: {v.strip()}")
    return "\n".join(parts).strip()


def _to_staging_row(example: Dict[str, Any], idx: int, currency: str) -> Dict[str, Any]:
    entities = _extract_entities(example)
    words = _extract_words(example)
    ocr_text = _reconstruct_ocr_text(words, entities)

    merchant = entities.get("company") if isinstance(entities.get("company"), str) else None
    if not merchant and words:
        merchant = words[0]
    merchant = str(merchant).strip() if merchant else None

    total = _extract_number(entities.get("total"))
    receipt_ts = entities.get("date") if isinstance(entities.get("date"), str) else None

    source_record_id = str(
        example.get("id")
        or example.get("receipt_id")
        or example.get("file_name")
        or example.get("image_id")
        or f"sroie_{idx}"
    )

    return {
        "source": "sroie",
        "source_record_id": source_record_id,
        "receipt_ts": receipt_ts,
        "merchant_raw": merchant,
        "total": total,
        "tax": None,
        "currency": currency,
        "ocr_text": ocr_text or None,
        "structured_fields_json": {
            "entities": {
                "company": entities.get("company"),
                "date": entities.get("date"),
                "address": entities.get("address"),
                "total": entities.get("total"),
            },
            "words_count": len(words),
        },
        "raw_payload_json": {
            "dataset": "jsdnrs/ICDAR2019-SROIE",
            "annotation": _json_safe(example),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--currency", default="MYR")
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use HF streaming API to avoid full local download (default: true).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming and load split in-memory.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "datasets package is required. Install with: pip install datasets\n"
            f"Underlying import error: {exc}"
        )

    db_path = Path(args.staging_db)
    staging.ensure_staging_db(db_path)

    print("Loading dataset jsdnrs/ICDAR2019-SROIE...")
    ds = load_dataset(
        "jsdnrs/ICDAR2019-SROIE",
        split=args.split,
        streaming=bool(args.streaming),
    )

    buffer: List[Dict[str, Any]] = []
    inserted_total = 0
    for idx, ex in enumerate(ds):
        if idx >= args.max_rows:
            break
        buffer.append(_to_staging_row(ex, idx=idx, currency=args.currency))
        if len(buffer) >= args.batch_size:
            inserted_total += staging.insert_staging_receipts(
                buffer, path=db_path, batch_size=args.batch_size
            )
            buffer = []
            if inserted_total % (args.batch_size * 5) == 0:
                print(f"Inserted {inserted_total} rows...")

    if buffer:
        inserted_total += staging.insert_staging_receipts(
            buffer, path=db_path, batch_size=args.batch_size
        )

    counts = staging.get_staging_counts(path=db_path)
    print("Import complete.")
    print(f"- inserted_rows: {inserted_total}")
    print(f"- staging_transactions_total: {counts['staging_transactions']}")
    print(f"- staging_receipts_total: {counts['staging_receipts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


