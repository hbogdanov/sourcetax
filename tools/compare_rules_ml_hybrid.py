#!/usr/bin/env python
"""Compare Rules vs TF-IDF vs Hybrid on held-out natural gold data.

Runs two training regimes:
1) natural_only: train on non-synthetic gold labels
2) natural_plus_synthetic: train on all gold labels

Evaluation is always on held-out non-synthetic records to avoid synthetic-only inflation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from sourcetax.gold import filter_human_labeled_gold
from sourcetax import mapping
from sourcetax.normalization import normalize_merchant_name
from sourcetax.taxonomy import normalize_category_name

# Reuse current rules logic used by tools/eval.py
from tools import eval as eval_tool


def _load_gold(path: Path) -> List[Dict]:
    rows: List[Dict] = []
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


def _description(rec: Dict) -> str:
    raw = rec.get("raw_payload")
    if isinstance(raw, dict):
        return str(rec.get("description") or raw.get("description") or raw.get("ocr_text") or "").strip()
    return str(rec.get("description") or "").strip()


def _label(rec: Dict) -> str:
    return normalize_category_name(rec.get("sourcetax_category_v1") or rec.get("category_final")) or ""


def _text_feature(rec: Dict) -> str:
    merchant = str(rec.get("merchant_raw") or "").strip()
    desc = _description(rec)
    return f"{merchant} {desc}".strip()


def _mcc_description(rec: Dict) -> str:
    raw = rec.get("raw_payload")
    if isinstance(raw, dict):
        return str(rec.get("mcc_description") or raw.get("mcc_description") or "").strip()
    return str(rec.get("mcc_description") or "").strip()


def _enriched_text_feature(rec: Dict) -> str:
    merchant_raw = str(rec.get("merchant_raw") or "").strip()
    merchant_norm = normalize_merchant_name(merchant_raw, case="lower") if merchant_raw else ""
    desc = _description(rec)
    mcc_desc = _mcc_description(rec)
    _, reasons = mapping.resolve_category_with_reason(
        merchant_raw=merchant_raw or None,
        description=desc or None,
        mcc=rec.get("mcc"),
        mcc_description=mcc_desc or None,
        external_category=rec.get("category_external"),
        amount=rec.get("amount"),
        fallback="Other Expense",
    )
    reason_tokens = " ".join(str(x).replace(":", " ").replace("_", " ") for x in (reasons or []))
    missing_mcc = "missing_mcc" if not mcc_desc else "has_mcc"
    parts = [merchant_raw, merchant_norm, desc, mcc_desc, reason_tokens, missing_mcc]
    return " ".join(p for p in parts if p).strip()


def _is_synthetic(rec: Dict) -> bool:
    return str(rec.get("source") or "").strip().lower() == "synthetic_gapfill"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=10000,
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _prepare(records: List[Dict], feature_mode: str) -> List[Dict]:
    out = []
    for r in records:
        y = _label(r)
        x = _enriched_text_feature(r) if feature_mode == "enriched" else _text_feature(r)
        if not y or not x:
            continue
        rr = dict(r)
        rr["_label"] = y
        rr["_text"] = x
        out.append(rr)
    return out


def _rules_pred_and_conf(rec: Dict, merchant_map: Dict[str, str]) -> Tuple[str, float]:
    pred, conf = eval_tool.categorize_by_rules(
        rec.get("merchant_raw"),
        merchant_map,
        description=_description(rec),
        mcc=rec.get("mcc"),
        mcc_description=rec.get("mcc_description"),
        external_category=rec.get("category_external"),
        amount=rec.get("amount"),
    )
    pred_n = normalize_category_name(pred) or "Other Expense"
    return pred_n, float(conf)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _fmt(m: Dict[str, float]) -> str:
    return f"acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} weighted_f1={m['weighted_f1']:.3f}"


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hybrid-threshold", type=float, default=0.85)
    parser.add_argument(
        "--feature-mode",
        default="enriched",
        choices=["baseline", "enriched"],
        help="baseline=merchant+description, enriched=merchant_norm+mcc_description+mapping_reason",
    )
    parser.add_argument(
        "--lowconf-threshold",
        type=float,
        default=0.8,
        help="Threshold for ambiguous-only slice (rule confidence < threshold).",
    )
    args = parser.parse_args()

    rows = _prepare(_load_gold(Path(args.gold)), feature_mode=args.feature_mode)
    if len(rows) < 30:
        print("Not enough gold rows for robust comparison.")
        return 1

    natural = [r for r in rows if not _is_synthetic(r)]
    synthetic = [r for r in rows if _is_synthetic(r)]
    if len(natural) < 20:
        print("Not enough non-synthetic rows for held-out evaluation.")
        return 1

    y_nat = [r["_label"] for r in natural]
    label_counts = Counter(y_nat)
    can_stratify = len(label_counts) > 1 and min(label_counts.values()) >= 2
    train_nat, test_nat = train_test_split(
        natural,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_nat if can_stratify else None,
    )

    print("Dataset summary")
    print(f"- total_rows: {len(rows)}")
    print(f"- natural_rows: {len(natural)}")
    print(f"- synthetic_rows: {len(synthetic)}")
    print(f"- feature_mode: {args.feature_mode}")
    print(f"- test_natural_rows: {len(test_nat)}")
    print(f"- train_natural_rows: {len(train_nat)}")

    merchant_map = eval_tool.load_merchant_category_map()
    y_true = np.asarray([r["_label"] for r in test_nat])

    # Rules baseline
    rule_preds = []
    rule_conf = []
    for r in test_nat:
        p, c = _rules_pred_and_conf(r, merchant_map)
        rule_preds.append(p)
        rule_conf.append(c)
    y_rules = np.asarray(rule_preds)
    c_rules = np.asarray(rule_conf, dtype=float)
    m_rules = _metrics(y_true, y_rules)

    # ML: natural only
    pipe_nat = _build_pipeline()
    pipe_nat.fit([r["_text"] for r in train_nat], [r["_label"] for r in train_nat])
    y_ml_nat = pipe_nat.predict([r["_text"] for r in test_nat])
    m_ml_nat = _metrics(y_true, y_ml_nat)

    # ML: natural + synthetic
    train_plus = train_nat + synthetic
    pipe_plus = _build_pipeline()
    pipe_plus.fit([r["_text"] for r in train_plus], [r["_label"] for r in train_plus])
    y_ml_plus = pipe_plus.predict([r["_text"] for r in test_nat])
    m_ml_plus = _metrics(y_true, y_ml_plus)

    # Hybrid on natural-only model
    y_hybrid_nat = np.where(c_rules >= args.hybrid_threshold, y_rules, y_ml_nat)
    m_hybrid_nat = _metrics(y_true, y_hybrid_nat)

    # Hybrid on plus-synthetic model
    y_hybrid_plus = np.where(c_rules >= args.hybrid_threshold, y_rules, y_ml_plus)
    m_hybrid_plus = _metrics(y_true, y_hybrid_plus)

    print("\nScoreboard (held-out natural test)")
    print(f"- Rules baseline:            {_fmt(m_rules)}")
    print(f"- TF-IDF (natural only):     {_fmt(m_ml_nat)}")
    print(f"- TF-IDF (natural+synth):    {_fmt(m_ml_plus)}")
    print(f"- Hybrid (rules+nat ML):     {_fmt(m_hybrid_nat)} threshold={args.hybrid_threshold}")
    print(f"- Hybrid (rules+plus ML):    {_fmt(m_hybrid_plus)} threshold={args.hybrid_threshold}")

    # Ambiguous-only slice where ML should theoretically help.
    low_mask = c_rules < float(args.lowconf_threshold)
    low_n = int(np.sum(low_mask))
    print(f"\nAmbiguous slice (rule_conf < {args.lowconf_threshold}): n={low_n}")
    if low_n > 0:
        y_true_low = y_true[low_mask]
        y_rules_low = y_rules[low_mask]
        y_ml_nat_low = y_ml_nat[low_mask]
        y_ml_plus_low = y_ml_plus[low_mask]
        print(f"- Rules baseline (lowconf):      {_fmt(_metrics(y_true_low, y_rules_low))}")
        print(f"- TF-IDF natural (lowconf):      {_fmt(_metrics(y_true_low, y_ml_nat_low))}")
        print(f"- TF-IDF plus-synth (lowconf):   {_fmt(_metrics(y_true_low, y_ml_plus_low))}")
    else:
        print("- No low-confidence rows in held-out set at this threshold.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
