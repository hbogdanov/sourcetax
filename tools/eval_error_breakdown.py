#!/usr/bin/env python
"""Detailed categorization error breakdown on the gold set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from sourcetax import mapping
from sourcetax.gold import filter_human_labeled_gold
from sourcetax.taxonomy import normalize_category_name


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


def _load_gold(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    filtered, _ = filter_human_labeled_gold(rows)
    return filtered


def _predict(rec: Dict) -> Tuple[str, str]:
    raw_payload = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw_payload, dict):
        raw_payload = {}
    merchant = rec.get("merchant_raw")
    description = rec.get("description") or raw_payload.get("description") or raw_payload.get("ocr_text")
    mcc = rec.get("mcc") or raw_payload.get("mcc")
    mcc_description = rec.get("mcc_description") or raw_payload.get("mcc_description")
    external_category = rec.get("category_external") or raw_payload.get("category_external")
    predicted, reasons = mapping.resolve_category_with_reason(
        merchant_raw=merchant,
        description=description,
        mcc=mcc,
        mcc_description=mcc_description,
        external_category=external_category,
        amount=rec.get("amount"),
        fallback="Other Expense",
    )
    pred = normalize_category_name(predicted) or "Other Expense"
    reason = ";".join(reasons or [])
    return pred, reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--top-wrong", type=int, default=50)
    parser.add_argument("--out-dir", default="outputs/reports")
    args = parser.parse_args()

    gold = _load_gold(Path(args.gold))
    if not gold:
        print("No gold rows found.")
        return 1

    truth_counts = Counter()
    pred_counts = Counter()
    tp_counts = Counter()
    confusions = Counter()
    wrong_rows: List[Dict] = []

    for rec in gold:
        truth = normalize_category_name(rec.get("sourcetax_category_v1") or rec.get("category_final"))
        if not truth:
            continue
        pred, reason = _predict(rec)
        truth_counts[truth] += 1
        pred_counts[pred] += 1
        if pred == truth:
            tp_counts[truth] += 1
        else:
            confusions[(truth, pred)] += 1
            wrong_rows.append(
                {
                    "id": rec.get("id"),
                    "merchant_raw": rec.get("merchant_raw"),
                    "mcc_description": rec.get("mcc_description")
                    or (_safe_json_loads(rec.get("raw_payload"), {}).get("mcc_description")),
                    "gold_label": truth,
                    "predicted_label": pred,
                    "mapping_reason": reason,
                }
            )

    categories = sorted(set(truth_counts.keys()) | set(pred_counts.keys()))
    per_cat_rows: List[Dict] = []
    for c in categories:
        tp = tp_counts[c]
        fp = max(pred_counts[c] - tp, 0)
        fn = max(truth_counts[c] - tp, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_cat_rows.append(
            {
                "category": c,
                "support": truth_counts[c],
                "predicted": pred_counts[c],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )

    top_confusions = confusions.most_common(20)
    wrong_top = wrong_rows[: args.top_wrong]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cat_path = out_dir / "eval_per_category_breakdown.csv"
    conf_path = out_dir / "eval_top_confusions.csv"
    wrong_path = out_dir / "eval_top_wrong_rows.csv"

    with per_cat_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_cat_rows[0].keys()) if per_cat_rows else [])
        if per_cat_rows:
            writer.writeheader()
            writer.writerows(per_cat_rows)

    with conf_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gold_label", "predicted_label", "count"])
        writer.writeheader()
        for (g, p), n in top_confusions:
            writer.writerow({"gold_label": g, "predicted_label": p, "count": n})

    with wrong_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(wrong_top[0].keys()) if wrong_top else [])
        if wrong_top:
            writer.writeheader()
            writer.writerows(wrong_top)

    print("Eval error breakdown complete.")
    print(f"- gold_rows: {len(gold)}")
    print(f"- per_category: {per_cat_path}")
    print(f"- top_confusions: {conf_path}")
    print(f"- top_wrong_rows: {wrong_path}")

    print("\nTop confusions:")
    for (g, p), n in top_confusions[:10]:
        print(f"- {g} -> {p}: {n}")

    print("\nPer-category precision/recall:")
    for row in sorted(per_cat_rows, key=lambda r: r["support"], reverse=True):
        print(
            f"- {row['category']}: precision={row['precision']:.2f}, "
            f"recall={row['recall']:.2f}, support={row['support']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
