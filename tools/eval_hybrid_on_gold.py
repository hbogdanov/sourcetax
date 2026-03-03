#!/usr/bin/env python
"""Evaluate rules/ML/hybrid predictions against gold truth and produce rollout reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from sourcetax import categorization
from sourcetax.gold import filter_human_labeled_gold
from sourcetax.taxonomy import normalize_category_name


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


def _read_gold(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    filtered, _ = filter_human_labeled_gold(rows)
    return filtered


def _context_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "merchant_raw": str(rec.get("merchant_raw") or "").strip(),
        "amount": rec.get("amount"),
        "description": str(rec.get("description") or raw.get("description") or raw.get("ocr_text") or "").strip(),
        "mcc": str(rec.get("mcc") or raw.get("mcc") or "").strip(),
        "mcc_description": str(rec.get("mcc_description") or raw.get("mcc_description") or "").strip(),
        "category_external": str(rec.get("category_external") or raw.get("category_external") or "").strip(),
    }


def _rule_bin(conf: float) -> str:
    if conf >= 0.90:
        return ">=0.90"
    if conf >= 0.80:
        return "0.80-0.89"
    if conf >= 0.70:
        return "0.70-0.79"
    return "<0.70"


def _acc(rows: List[Dict[str, Any]], pred_key: str) -> float:
    if not rows:
        return 0.0
    correct = 0
    total = 0
    for r in rows:
        truth = r.get("gold_label")
        pred = r.get(pred_key)
        if not truth:
            continue
        total += 1
        if truth == pred:
            correct += 1
    return (correct / total) if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--out-dir", default="artifacts/reports")
    parser.add_argument("--candidate-threshold", type=float, default=0.85)
    parser.add_argument("--candidate-ml-min", type=float, default=0.0)
    args = parser.parse_args()

    gold = _read_gold(Path(args.gold))
    if not gold:
        print(f"No gold rows found at {args.gold}")
        return 1

    evaluated: List[Dict[str, Any]] = []
    for rec in gold:
        truth = normalize_category_name(rec.get("sourcetax_category_v1") or rec.get("category_final"))
        if not truth:
            continue
        ctx = _context_from_record(rec)
        shadow = categorization.build_shadow_decisions(
            merchant_raw=ctx["merchant_raw"],
            amount=ctx["amount"],
            description=ctx["description"],
            mcc=ctx["mcc"],
            mcc_description=ctx["mcc_description"],
            category_external=ctx["category_external"],
            raw_payload=rec.get("raw_payload") if isinstance(rec.get("raw_payload"), dict) else {},
            threshold_primary=0.85,
            threshold_alt=0.70,
        )
        evaluated.append(
            {
                "id": rec.get("id"),
                "source": rec.get("source"),
                "merchant_raw": ctx["merchant_raw"],
                "description": ctx["description"],
                "mcc_description": ctx["mcc_description"],
                "amount": ctx["amount"],
                "gold_label": truth,
                "rule_category": normalize_category_name(shadow.get("rule_category")) or "Other Expense",
                "rule_confidence": float(shadow.get("rule_confidence") or 0.0),
                "ml_prediction": normalize_category_name(shadow.get("ml_prediction")) if shadow.get("ml_prediction") else None,
                "ml_confidence": shadow.get("ml_confidence"),
                "hybrid_t85": normalize_category_name(shadow.get("hybrid_prediction_t85")) or "Other Expense",
                "hybrid_t70": normalize_category_name(shadow.get("hybrid_prediction_t70")) or "Other Expense",
                "rule_reason": "|".join(shadow.get("rule_reason") or []),
                "ml_model_status": shadow.get("ml_model_status"),
            }
        )

    overall = {
        "rules_accuracy": _acc(evaluated, "rule_category"),
        "ml_accuracy": _acc(evaluated, "ml_prediction"),
        "hybrid_t70_accuracy": _acc(evaluated, "hybrid_t70"),
        "hybrid_t85_accuracy": _acc(evaluated, "hybrid_t85"),
        "n": len(evaluated),
    }

    by_bin: Dict[str, Dict[str, float]] = {}
    bins = [">=0.90", "0.80-0.89", "0.70-0.79", "<0.70"]
    for b in bins:
        subset = [r for r in evaluated if _rule_bin(float(r.get("rule_confidence") or 0.0)) == b]
        by_bin[b] = {
            "n": len(subset),
            "rules_accuracy": _acc(subset, "rule_category"),
            "ml_accuracy": _acc(subset, "ml_prediction"),
            "hybrid_t70_accuracy": _acc(subset, "hybrid_t70"),
            "hybrid_t85_accuracy": _acc(subset, "hybrid_t85"),
        }

    # Threshold sweep on gold.
    sweep = []
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds = []
        for r in evaluated:
            rule = r["rule_category"]
            ml = r["ml_prediction"] or rule
            rc = float(r["rule_confidence"] or 0.0)
            preds.append(ml if rc < t else rule)
        correct = sum(1 for r, p in zip(evaluated, preds) if r["gold_label"] == p)
        sweep.append({"threshold": t, "hybrid_accuracy": correct / len(evaluated) if evaluated else 0.0})

    # Candidate-change report for rollout review.
    candidates = []
    for r in evaluated:
        rule = r["rule_category"]
        ml = r["ml_prediction"]
        rc = float(r["rule_confidence"] or 0.0)
        mc = float(r["ml_confidence"] or 0.0) if r.get("ml_confidence") is not None else 0.0
        if ml and ml != rule and rc < args.candidate_threshold and mc >= args.candidate_ml_min:
            rr = dict(r)
            rr["hybrid_candidate"] = True
            candidates.append(rr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "hybrid_gold_eval_summary.json"
    bin_path = out_dir / "hybrid_gold_eval_by_conf_bin.csv"
    sweep_path = out_dir / "hybrid_gold_threshold_sweep.csv"
    cand_path = out_dir / "hybrid_candidate_flag_gold.csv"
    row_path = out_dir / "hybrid_gold_row_level.csv"

    summary = {
        "overall": overall,
        "by_conf_bin": by_bin,
        "threshold_sweep": sweep,
        "candidate_threshold": args.candidate_threshold,
        "candidate_ml_min": args.candidate_ml_min,
        "candidate_rows": len(candidates),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with bin_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["bin", "n", "rules_accuracy", "ml_accuracy", "hybrid_t70_accuracy", "hybrid_t85_accuracy"],
        )
        writer.writeheader()
        for b in bins:
            row = dict(by_bin[b])
            row["bin"] = b
            writer.writerow(row)

    with sweep_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["threshold", "hybrid_accuracy"])
        writer.writeheader()
        writer.writerows(sweep)

    if evaluated:
        with row_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(evaluated[0].keys()))
            writer.writeheader()
            writer.writerows(evaluated)
    else:
        row_path.write_text("", encoding="utf-8")

    if candidates:
        with cand_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(candidates[0].keys()))
            writer.writeheader()
            writer.writerows(candidates)
    else:
        cand_path.write_text("", encoding="utf-8")

    print("Hybrid gold evaluation complete.")
    print(f"- n: {overall['n']}")
    print(f"- rules_accuracy: {overall['rules_accuracy']:.3f}")
    print(f"- ml_accuracy: {overall['ml_accuracy']:.3f}")
    print(f"- hybrid_t70_accuracy: {overall['hybrid_t70_accuracy']:.3f}")
    print(f"- hybrid_t85_accuracy: {overall['hybrid_t85_accuracy']:.3f}")
    print(f"- candidate_rows: {len(candidates)}")
    print(f"- summary: {summary_path}")
    print(f"- by_conf_bin: {bin_path}")
    print(f"- threshold_sweep: {sweep_path}")
    print(f"- candidates: {cand_path}")
    print(f"- row_level: {row_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


