#!/usr/bin/env python
"""5-fold CV stability check for rules vs enriched ML vs hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sourcetax import mapping
from sourcetax.gold import filter_human_labeled_gold
from sourcetax.normalization import normalize_merchant_name
from sourcetax.taxonomy import normalize_category_name

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import eval as eval_tool


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
    rows = []
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


def _label(rec: Dict) -> str:
    return normalize_category_name(rec.get("sourcetax_category_v1") or rec.get("category_final")) or ""


def _description(rec: Dict) -> str:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return str(rec.get("description") or raw.get("description") or raw.get("ocr_text") or "").strip()


def _mcc_description(rec: Dict) -> str:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return str(rec.get("mcc_description") or raw.get("mcc_description") or "").strip()


def _enriched_text(rec: Dict) -> str:
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
    return " ".join(p for p in [merchant_raw, merchant_norm, desc, mcc_desc, reason_tokens, missing_mcc] if p).strip()


def _is_synthetic(rec: Dict) -> bool:
    return str(rec.get("source") or "").strip().lower() == "synthetic_gapfill"


def _rules_pred_conf(rec: Dict, merchant_map: Dict[str, str]) -> Tuple[str, float]:
    pred, conf = eval_tool.categorize_by_rules(
        rec.get("merchant_raw"),
        merchant_map,
        description=_description(rec),
        mcc=rec.get("mcc"),
        mcc_description=_mcc_description(rec),
        external_category=rec.get("category_external"),
        amount=rec.get("amount"),
    )
    return normalize_category_name(pred) or "Other Expense", float(conf)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hybrid-threshold", type=float, default=0.85)
    parser.add_argument(
        "--drop-rare-classes",
        action="store_true",
        default=True,
        help="Exclude classes with support < folds for stratified CV stability (default: true).",
    )
    parser.add_argument(
        "--keep-rare-classes",
        action="store_false",
        dest="drop_rare_classes",
        help="Fail instead of excluding rare classes.",
    )
    args = parser.parse_args()

    rows = _load_gold(Path(args.gold))
    natural = [r for r in rows if not _is_synthetic(r)]
    synthetic = [r for r in rows if _is_synthetic(r)]
    prepared = []
    for r in natural:
        y = _label(r)
        x = _enriched_text(r)
        if y and x:
            rr = dict(r)
            rr["_y"] = y
            rr["_x"] = x
            prepared.append(rr)

    if len(prepared) < args.folds * 10:
        print(f"Not enough natural rows for {args.folds}-fold CV: {len(prepared)}")
        return 1

    y_all = np.asarray([r["_y"] for r in prepared])
    counts = dict(zip(*np.unique(y_all, return_counts=True)))
    dropped = {k: v for k, v in counts.items() if v < args.folds}
    if dropped and args.drop_rare_classes:
        prepared = [r for r in prepared if counts.get(r["_y"], 0) >= args.folds]
        y_all = np.asarray([r["_y"] for r in prepared])
        print("Dropped rare classes for CV (support < folds):")
        for k, v in sorted(dropped.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"- {k}: {v}")

    min_class = min(np.unique(y_all, return_counts=True)[1])
    if min_class < args.folds:
        print(f"Cannot run {args.folds}-fold stratified CV; min class count is {min_class}.")
        return 1

    merchant_map = eval_tool.load_merchant_category_map()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    rules_scores: List[float] = []
    ml_scores: List[float] = []
    hybrid_scores: List[float] = []

    synthetic_train = []
    for r in synthetic:
        y = _label(r)
        x = _enriched_text(r)
        if y and x:
            synthetic_train.append((x, y))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(prepared)), y_all), start=1):
        train_rows = [prepared[i] for i in train_idx]
        test_rows = [prepared[i] for i in test_idx]

        X_train = [r["_x"] for r in train_rows] + [x for x, _ in synthetic_train]
        y_train = [r["_y"] for r in train_rows] + [y for _, y in synthetic_train]
        X_test = [r["_x"] for r in test_rows]
        y_test = np.asarray([r["_y"] for r in test_rows])

        pipe = _build_pipeline()
        pipe.fit(X_train, y_train)
        y_ml = pipe.predict(X_test)

        y_rules = []
        c_rules = []
        for r in test_rows:
            p, c = _rules_pred_conf(r, merchant_map)
            y_rules.append(p)
            c_rules.append(c)
        y_rules = np.asarray(y_rules)
        c_rules = np.asarray(c_rules, dtype=float)

        y_hybrid = np.where(c_rules >= args.hybrid_threshold, y_rules, y_ml)

        rules_acc = float(accuracy_score(y_test, y_rules))
        ml_acc = float(accuracy_score(y_test, y_ml))
        hybrid_acc = float(accuracy_score(y_test, y_hybrid))

        rules_scores.append(rules_acc)
        ml_scores.append(ml_acc)
        hybrid_scores.append(hybrid_acc)

        print(
            f"fold={fold_idx} rules={rules_acc:.3f} ml={ml_acc:.3f} "
            f"hybrid={hybrid_acc:.3f} threshold={args.hybrid_threshold}"
        )

    def _summary(name: str, vals: List[float]) -> str:
        arr = np.asarray(vals, dtype=float)
        return f"{name}: mean={arr.mean():.3f} std={arr.std(ddof=0):.3f} min={arr.min():.3f} max={arr.max():.3f}"

    print("\nCV summary")
    print(f"- natural_rows={len(prepared)} synthetic_rows={len(synthetic_train)} folds={args.folds}")
    print(f"- {_summary('rules', rules_scores)}")
    print(f"- {_summary('ml_enriched', ml_scores)}")
    print(f"- {_summary('hybrid', hybrid_scores)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
