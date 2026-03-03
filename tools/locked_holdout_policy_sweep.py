#!/usr/bin/env python
"""Sweep hybrid gating policies on locked-holdout row-level predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _acc(rows: List[Dict], pred_key: str) -> float:
    if not rows:
        return 0.0
    correct = 0
    total = 0
    for r in rows:
        y = str(r.get("gold_label") or "").strip()
        p = str(r.get(pred_key) or "").strip()
        if not y:
            continue
        total += 1
        if y == p:
            correct += 1
    return (correct / total) if total else 0.0


def _policy_rule_conf(rows: List[Dict], t_rule: float) -> float:
    correct = 0
    for r in rows:
        y = str(r["gold_label"])
        rule = str(r["rule_category"])
        ml = str(r["ml_prediction"])
        rc = _f(r.get("rule_confidence"), 0.0)
        pred = ml if rc < t_rule else rule
        if pred == y:
            correct += 1
    return correct / len(rows) if rows else 0.0


def _policy_rule_conf_ml_conf(rows: List[Dict], t_rule: float, t_ml: float) -> float:
    correct = 0
    for r in rows:
        y = str(r["gold_label"])
        rule = str(r["rule_category"])
        ml = str(r["ml_prediction"])
        rc = _f(r.get("rule_confidence"), 0.0)
        mc = _f(r.get("ml_confidence"), 0.0)
        pred = ml if (rc < t_rule and mc >= t_ml) else rule
        if pred == y:
            correct += 1
    return correct / len(rows) if rows else 0.0


def _policy_disagreement_only(rows: List[Dict], t_rule: float, t_ml: float) -> float:
    correct = 0
    for r in rows:
        y = str(r["gold_label"])
        rule = str(r["rule_category"])
        ml = str(r["ml_prediction"])
        rc = _f(r.get("rule_confidence"), 0.0)
        mc = _f(r.get("ml_confidence"), 0.0)
        use_ml = (rule != ml) and (rc < t_rule) and (mc >= t_ml)
        pred = ml if use_ml else rule
        if pred == y:
            correct += 1
    return correct / len(rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row-level-csv",
        default="artifacts/reports/holdout_locked_post_batch456/locked_holdout_row_level.csv",
    )
    parser.add_argument("--out-json", default="artifacts/reports/holdout_locked_post_batch456/policy_sweep_summary.json")
    parser.add_argument("--out-csv", default="artifacts/reports/holdout_locked_post_batch456/policy_sweep_results.csv")
    args = parser.parse_args()

    p = Path(args.row_level_csv)
    if not p.exists():
        print(f"Missing row-level file: {p}")
        return 1

    with p.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"No rows in {p}")
        return 1

    baseline = _acc(rows, "rule_category")
    ml_baseline = _acc(rows, "ml_prediction")

    t_rules = [round(x, 2) for x in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
    t_mls = [round(x, 2) for x in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]

    results: List[Dict] = []
    for tr in t_rules:
        results.append(
            {
                "policy": "hybrid_rule_conf",
                "rule_conf_threshold": tr,
                "ml_conf_threshold": "",
                "accuracy": _policy_rule_conf(rows, tr),
            }
        )
    for tr in t_rules:
        for tm in t_mls:
            results.append(
                {
                    "policy": "hybrid_rule_conf_ml_conf",
                    "rule_conf_threshold": tr,
                    "ml_conf_threshold": tm,
                    "accuracy": _policy_rule_conf_ml_conf(rows, tr, tm),
                }
            )
            results.append(
                {
                    "policy": "hybrid_disagreement_only",
                    "rule_conf_threshold": tr,
                    "ml_conf_threshold": tm,
                    "accuracy": _policy_disagreement_only(rows, tr, tm),
                }
            )

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    best_by_policy: Dict[str, Dict] = {}
    for r in results:
        pol = str(r["policy"])
        if pol not in best_by_policy:
            best_by_policy[pol] = r

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["policy", "rule_conf_threshold", "ml_conf_threshold", "accuracy", "delta_vs_rules_pp"],
        )
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["delta_vs_rules_pp"] = (float(r["accuracy"]) - baseline) * 100.0
            writer.writerow(row)

    summary = {
        "row_level_csv": str(p),
        "n_holdout": len(rows),
        "rules_accuracy": baseline,
        "ml_accuracy": ml_baseline,
        "best_by_policy": best_by_policy,
        "best_overall": results[0] if results else None,
        "beats_rules_by_1pp_or_more": [r for r in results if (float(r["accuracy"]) - baseline) >= 0.01][:20],
        "beats_rules_by_2pp_or_more": [r for r in results if (float(r["accuracy"]) - baseline) >= 0.02][:20],
        "top10": results[:10],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Locked holdout policy sweep complete.")
    print(f"- n_holdout: {len(rows)}")
    print(f"- rules_accuracy: {baseline:.3f}")
    print(f"- ml_accuracy: {ml_baseline:.3f}")
    bo = summary.get("best_overall") or {}
    print(
        f"- best_overall: policy={bo.get('policy')} rule_t={bo.get('rule_conf_threshold')} ml_t={bo.get('ml_conf_threshold')} acc={_f(bo.get('accuracy')):.3f}"
    )
    print(f"- out_json: {out_json}")
    print(f"- out_csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


