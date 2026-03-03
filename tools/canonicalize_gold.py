#!/usr/bin/env python
"""Canonicalize gold JSONL records to schema contract.

Enforces:
- amount is non-negative
- direction is present and in {expense,income}
- currency present (default USD)
- source_record_id present (stable hash fallback)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _to_float(value: Any):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _infer_direction(direction_value: Any, amount_value: Any, category_value: Any) -> str:
    d = str(direction_value or "").strip().lower()
    if d in {"expense", "income"}:
        return d
    amt = _to_float(amount_value)
    if amt is not None:
        if amt < 0:
            return "expense"
        if amt > 0:
            return "income"
    cat = str(category_value or "").strip().lower()
    if cat == "income":
        return "income"
    return "expense"


def _stable_source_record_id(rec: Dict[str, Any]) -> str:
    source = str(rec.get("source") or "").strip().lower()
    merchant = str(rec.get("merchant_raw") or "").strip().lower()
    date = str(rec.get("transaction_date") or "").strip()
    amount = _to_float(rec.get("amount"))
    amount_txt = "" if amount is None else f"{abs(amount):.2f}"
    direction = str(rec.get("direction") or "").strip().lower()
    base = f"{source}|{merchant}|{date}|{amount_txt}|{direction}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"hash_{digest}"


def canonicalize_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out["currency"] = str(out.get("currency") or "USD").strip() or "USD"
    out["direction"] = _infer_direction(
        out.get("direction"),
        out.get("amount"),
        out.get("sourcetax_category_v1") or out.get("category_final"),
    )
    amt = _to_float(out.get("amount"))
    out["amount"] = abs(amt) if amt is not None else None
    if not str(out.get("source_record_id") or "").strip():
        out["source_record_id"] = _stable_source_record_id(out)
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    miss_direction = sum(1 for r in rows if str(r.get("direction") or "").strip().lower() not in {"expense", "income"})
    missing_source_record_id = sum(1 for r in rows if not str(r.get("source_record_id") or "").strip())
    missing_currency = sum(1 for r in rows if not str(r.get("currency") or "").strip())
    neg = 0
    pos = 0
    zero = 0
    direction_counts = Counter()
    for r in rows:
        amt = _to_float(r.get("amount"))
        if amt is None:
            continue
        if amt < 0:
            neg += 1
        elif amt > 0:
            pos += 1
        else:
            zero += 1
        d = str(r.get("direction") or "").strip().lower()
        if d:
            direction_counts[d] += 1
    return {
        "rows": len(rows),
        "missing_direction": miss_direction,
        "negative_amounts": neg,
        "positive_amounts": pos,
        "zero_amounts": zero,
        "missing_source_record_id": missing_source_record_id,
        "missing_currency": missing_currency,
        "direction_counts": dict(direction_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--in-place", action="store_true", default=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    path = Path(args.gold)
    rows = _load_jsonl(path)
    if not rows:
        print(f"No gold rows found at {path}")
        return 1

    before = _stats(rows)
    canonical = [canonicalize_row(r) for r in rows]
    after = _stats(canonical)

    out_path = path if args.in_place or not args.out else Path(args.out)
    _write_jsonl(out_path, canonical)

    print("Gold canonicalization complete.")
    print(f"- input: {path}")
    print(f"- output: {out_path}")
    print(f"- before: {json.dumps(before)}")
    print(f"- after: {json.dumps(after)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

