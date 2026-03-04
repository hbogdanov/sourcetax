#!/usr/bin/env python
"""Train SourceTax gold classifier with optional Mitul warm-start vocabulary."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sourcetax.models import data_prep, evaluate, train_baseline  # noqa: E402
from sourcetax.normalization import normalize_merchant_name  # noqa: E402
from sourcetax.text import combine_text_fields, normalize_text  # noqa: E402

KEY_CATEGORY_NAMES = [
    "Repairs & Maintenance",
    "Rent & Utilities",
    "Financial Fees",
    "Income",
    "Meals & Entertainment",
]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dataset_file_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _git_commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


def _write_run_index(run_id: str, config: dict, inputs: dict, outputs: dict) -> Path:
    run_dir = Path("artifacts/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "run.json"
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "config": _json_safe(config),
        "inputs": _json_safe(inputs),
        "outputs": _json_safe(outputs),
    }
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_vocab_from_pipeline(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Pretrained model not found: {path}")
    try:
        import joblib

        pipe = joblib.load(path)
    except Exception:
        import pickle

        with path.open("rb") as fh:
            pipe = pickle.load(fh)
    if not hasattr(pipe, "named_steps") or "tfidf" not in pipe.named_steps:
        raise SystemExit(f"Pretrained model at {path} is not a TF-IDF pipeline.")
    tfidf = pipe.named_steps["tfidf"]
    vocab = getattr(tfidf, "vocabulary_", None)
    if not vocab:
        raise SystemExit(f"Pretrained model at {path} has empty TF-IDF vocabulary.")
    return dict(vocab)


def _extract_vocab_from_parquet(path: Path, max_features: int) -> dict:
    if not path.exists():
        raise SystemExit(f"Warm-start parquet not found: {path}")
    import pandas as pd

    df = pd.read_parquet(path, columns=["text"])
    if "text" not in df.columns:
        raise SystemExit(f"Parquet {path} missing required 'text' column.")
    texts = df["text"].astype(str).map(normalize_text)
    texts = texts[texts.str.len() > 0]
    if texts.empty:
        raise SystemExit(f"Warm-start parquet {path} has no non-empty text rows.")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        max_features=max_features,
        lowercase=True,
        stop_words="english",
    )
    vec.fit(texts.tolist())
    vocab = getattr(vec, "vocabulary_", None) or {}
    if not vocab:
        raise SystemExit(f"Failed to build vocabulary from {path}.")
    return dict(vocab)


def _build_pipeline_with_vocab(vocab: dict, seed: int) -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        lowercase=True,
        stop_words="english",
        vocabulary=vocab,
    )
    classifier = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([("tfidf", tfidf), ("classifier", classifier)])


def _per_class_and_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
    return {"per_class_metrics": per_class, "confusion_matrix": {"labels": labels, "matrix": cm.tolist()}}


def _serialize_split_ids(df) -> List[str]:
    out: List[str] = []
    for idx, row in df.iterrows():
        rid = row.get("id")
        if rid is None or str(rid).strip() == "":
            out.append(f"row_index:{idx}")
        else:
            out.append(str(rid))
    return out


def _apply_split_ids(df, ids: List[str]):
    id_index = {}
    for idx, row in df.iterrows():
        rid = row.get("id")
        key = str(rid) if rid is not None and str(rid).strip() else f"row_index:{idx}"
        id_index[key] = idx
    rows = [id_index[x] for x in ids if x in id_index]
    return df.loc[rows].copy()


def _evaluate_and_collect(test_df, pipeline):
    merchant_map = evaluate.load_merchant_category_map()
    y_true = test_df["category"].values
    y_pred_rules, metrics_rules = evaluate.evaluate_rules(test_df, merchant_map)
    y_pred_ml, _y_proba_ml, metrics_ml = evaluate.evaluate_ml(test_df, pipeline)
    y_pred_rules = np.asarray(y_pred_rules)
    y_pred_ml = np.asarray(y_pred_ml)
    rules_extra = _per_class_and_confusion(y_true, y_pred_rules)
    ml_extra = _per_class_and_confusion(y_true, y_pred_ml)
    return y_true, y_pred_rules, y_pred_ml, metrics_rules, metrics_ml, rules_extra, ml_extra


def _split_with_key_test_support(
    ml_df,
    *,
    seed: int,
    key_categories: List[str],
    min_support: int,
):
    if min_support <= 0:
        train_df, val_df, test_df = data_prep.split_dataset(ml_df, random_state=int(seed))
        return train_df, val_df, test_df, []

    working = ml_df.copy()
    forced_test_chunks = []
    needs_labeling = []
    rng = np.random.default_rng(int(seed))

    for category in key_categories:
        cat_df = working[working["category"] == category]
        total = int(len(cat_df))
        if total < int(min_support):
            needs_labeling.append(
                {
                    "category": category,
                    "required_min_test_support": int(min_support),
                    "available_total_rows": total,
                    "additional_rows_needed_for_target": int(min_support) - total,
                }
            )
            continue
        picked = cat_df.sample(n=int(min_support), random_state=int(seed))
        forced_test_chunks.append(picked)
        working = working.drop(index=picked.index)

    train_df, val_df, test_df = data_prep.split_dataset(working, random_state=int(seed))
    if forced_test_chunks:
        forced_df = pd.concat(forced_test_chunks, ignore_index=False)
        test_df = pd.concat([test_df, forced_df], ignore_index=False)
    return train_df, val_df, test_df, needs_labeling


def _build_text_features(
    df,
    *,
    text_case: str,
    strip_currency_tokens: bool,
    strip_country_tokens: bool,
    merchant_normalization: str,
):
    lowercase = text_case == "lower"
    out = df.copy()

    def _norm_merchant(value: str) -> str:
        text = str(value or "")
        if merchant_normalization == "on":
            case = "lower" if lowercase else "preserve"
            return normalize_merchant_name(text, case=case)
        return normalize_text(text, lowercase=lowercase)

    out["text"] = out.apply(
        lambda row: combine_text_fields(
            [
                _norm_merchant(row.get("merchant", "")),
                str(row.get("description", "") or ""),
            ],
            remove_currency_tokens=strip_currency_tokens,
            remove_country_tokens=strip_country_tokens,
            lowercase=lowercase,
        ),
        axis=1,
    )
    out = out[out["text"].str.len() > 0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="", help="Optional run identifier.")
    parser.add_argument("--output-pipeline", default="")
    parser.add_argument(
        "--vectorizer-vocab-from",
        default="",
        help="Optional parquet path with a text column for warm-start TF-IDF vocabulary.",
    )
    parser.add_argument(
        "--pretrained-model",
        default="",
        help="Optional pretrained TF-IDF pipeline path to reuse vocabulary from.",
    )
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional cap for training rows (deterministic sample) for fast debug runs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick end-to-end debug run (sets max-train-rows=200 and max-features=20000 unless provided).",
    )
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--split-ids-out", default="")
    parser.add_argument(
        "--split-ids-in",
        default="",
        help="Optional previously saved split IDs JSON to force identical train/val/test splits.",
    )
    parser.add_argument(
        "--require-split-ids-in",
        action="store_true",
        help="Fail unless --split-ids-in is provided (for strict apples-to-apples comparisons).",
    )
    parser.add_argument(
        "--compat-copy",
        default="",
        help="Optional compatibility copy path (disabled by default).",
    )
    parser.add_argument(
        "--text-case",
        choices=["lower", "raw"],
        default="lower",
        help="Text casing mode for feature construction.",
    )
    parser.add_argument(
        "--strip-currency-tokens",
        action="store_true",
        help="Remove known currency tokens during text normalization.",
    )
    parser.add_argument(
        "--strip-country-tokens",
        action="store_true",
        help="Remove common country tokens during text normalization.",
    )
    parser.add_argument(
        "--merchant-normalization",
        choices=["on", "off"],
        default="on",
        help="Apply merchant normalization before feature construction.",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.85,
        help="Rules confidence threshold for rules-first hybrid fallback to ML.",
    )
    parser.add_argument(
        "--key-test-min-support",
        type=int,
        default=0,
        help="If >0 and generating splits, force at least this many test rows for key categories when available.",
    )
    parser.add_argument(
        "--key-categories",
        default=",".join(KEY_CATEGORY_NAMES),
        help="Comma-separated key categories for minimum test-support enforcement.",
    )
    args = parser.parse_args()
    if args.smoke:
        if not args.max_train_rows:
            args.max_train_rows = 200
        if args.max_features == 100000:
            args.max_features = 20000

    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    if args.require_split_ids_in and not args.split_ids_in:
        raise SystemExit("Strict split mode enabled: --split-ids-in is required.")

    run_id = args.run_id or _now_run_id()
    output_pipeline = Path(args.output_pipeline) if args.output_pipeline else Path(
        f"artifacts/models/gold_ml_baseline_pipeline_{run_id}.pkl"
    )
    metrics_out = Path(args.metrics_out) if args.metrics_out else Path(
        f"artifacts/metrics/gold_ml_baseline_metrics_{run_id}.json"
    )
    split_ids_out = Path(args.split_ids_out) if args.split_ids_out else Path(
        f"artifacts/reports/gold_ml_baseline_split_ids_{run_id}.json"
    )

    output_pipeline.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    split_ids_out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SOURCETAX: ML BASELINE TRAINING")
    print("=" * 80)

    gold_path = Path(args.gold)
    gold_records = data_prep.load_gold_set(gold_path)
    if not gold_records:
        print(f"No gold dataset found at {args.gold}.")
        return 1
    print(f"Loaded {len(gold_records)} gold records")

    ml_df = data_prep.prepare_ml_records(gold_records)
    ml_df = _build_text_features(
        ml_df,
        text_case=str(args.text_case),
        strip_currency_tokens=bool(args.strip_currency_tokens),
        strip_country_tokens=bool(args.strip_country_tokens),
        merchant_normalization=str(args.merchant_normalization),
    )
    if len(ml_df) < 10:
        print("Very small dataset (<10 records). Results are not reliable.")
    print(f"Prepared {len(ml_df)} ML rows")

    key_categories = [x.strip() for x in str(args.key_categories).split(",") if x.strip()]
    split_source = "generated"
    split_ids_in_payload = None
    needs_labeling_categories = []
    if args.split_ids_in:
        split_ids_path = Path(args.split_ids_in)
        if not split_ids_path.exists():
            raise SystemExit(f"split IDs file not found: {split_ids_path}")
        split_ids_in_payload = json.loads(split_ids_path.read_text(encoding="utf-8"))
        train_df = _apply_split_ids(ml_df, split_ids_in_payload.get("train_ids", []))
        val_df = _apply_split_ids(ml_df, split_ids_in_payload.get("val_ids", []))
        test_df = _apply_split_ids(ml_df, split_ids_in_payload.get("test_ids", []))
        split_source = f"from_file:{split_ids_path}"
    else:
        train_df, val_df, test_df, needs_labeling_categories = _split_with_key_test_support(
            ml_df,
            seed=int(args.seed),
            key_categories=key_categories,
            min_support=int(args.key_test_min_support),
        )
        if int(args.key_test_min_support) > 0:
            split_source = "generated:key_min_support"
    print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    if args.max_train_rows and args.max_train_rows > 0 and len(train_df) > int(args.max_train_rows):
        train_df = train_df.sample(n=int(args.max_train_rows), random_state=int(args.seed))
        print(f"Applied max-train-rows={int(args.max_train_rows)} -> train={len(train_df)}")

    warm_start_mode = "none"
    warm_start_meta: Dict[str, str] = {}
    pipeline = None
    if args.pretrained_model:
        pretrained_path = Path(args.pretrained_model)
        vocab = _extract_vocab_from_pipeline(pretrained_path)
        pipeline = _build_pipeline_with_vocab(vocab, seed=int(args.seed))
        warm_start_mode = "pretrained_model_vocab"
        warm_start_meta = _dataset_file_info(pretrained_path)
    elif args.vectorizer_vocab_from:
        vocab_path = Path(args.vectorizer_vocab_from)
        if not vocab_path.exists():
            raise SystemExit(
                f"--vectorizer-vocab-from path does not exist: {vocab_path}. "
                "Run without this flag for gold-only flow."
            )
        vocab = _extract_vocab_from_parquet(vocab_path, max_features=int(args.max_features))
        pipeline = _build_pipeline_with_vocab(vocab, seed=int(args.seed))
        warm_start_mode = "parquet_vocab"
        warm_start_meta = _dataset_file_info(vocab_path)

    if pipeline is None:
        tfidf_params = {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "max_features": int(args.max_features),
            "lowercase": True,
            "stop_words": "english",
        }
        pipeline, train_metrics = train_baseline.train_baseline(
            train_df, val_df, random_state=int(args.seed), tfidf_params=tfidf_params
        )
    else:
        print(f"Training with warm-start vocabulary ({warm_start_mode})...")
        pipeline.fit(train_df["text"], train_df["category"])
        y_val = pipeline.predict(val_df["text"])
        from sklearn.metrics import precision_recall_fscore_support

        acc = float(accuracy_score(val_df["category"], y_val))
        p, r, f1, _ = precision_recall_fscore_support(
            val_df["category"], y_val, average="weighted", zero_division=0
        )
        train_metrics = {
            "train_records": int(len(train_df)),
            "val_records": int(len(val_df)),
            "vocabulary_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
            "val_accuracy": acc,
            "val_macro_f1": float(f1_score(val_df["category"], y_val, average="macro", zero_division=0)),
            "val_weighted_f1": float(f1_score(val_df["category"], y_val, average="weighted", zero_division=0)),
            "val_precision": float(p),
            "val_recall": float(r),
            "val_f1": float(f1),
            "random_state": int(args.seed),
            "tfidf_params": _json_safe(pipeline.named_steps["tfidf"].get_params()),
        }
        print(f"Validation accuracy: {acc:.1%}")

    train_baseline.save_pipeline(pipeline, output_pipeline)
    compat_path = ""
    if args.compat_copy:
        compat_path_obj = Path(args.compat_copy)
        train_baseline.save_pipeline(pipeline, compat_path_obj)
        compat_path = str(compat_path_obj)

    y_true, y_pred_rules, y_pred_ml, metrics_rules, metrics_ml, rules_extra, ml_extra = _evaluate_and_collect(
        test_df, pipeline
    )
    merchant_map = evaluate.load_merchant_category_map()
    rule_conf = np.asarray([float(evaluate.apply_rules(text, merchant_map)[1]) for text in test_df["text"]], dtype=float)
    hybrid_mask = rule_conf >= float(args.hybrid_threshold)
    y_pred_hybrid = np.where(hybrid_mask, y_pred_rules, y_pred_ml)
    hybrid_extra = _per_class_and_confusion(y_true, y_pred_hybrid)
    hybrid_accuracy = float(accuracy_score(y_true, y_pred_hybrid))
    hybrid_macro_f1 = float(f1_score(y_true, y_pred_hybrid, average="macro", zero_division=0))
    hybrid_weighted_f1 = float(f1_score(y_true, y_pred_hybrid, average="weighted", zero_division=0))
    rules_coverage = float(np.mean(hybrid_mask)) if len(hybrid_mask) else 0.0
    ml_coverage = float(1.0 - rules_coverage)
    rules_other_rate = float(np.mean(y_pred_rules == "Other Expense")) if len(y_pred_rules) else 0.0
    ml_other_rate = float(np.mean(y_pred_ml == "Other Expense")) if len(y_pred_ml) else 0.0
    hybrid_other_rate = float(np.mean(y_pred_hybrid == "Other Expense")) if len(y_pred_hybrid) else 0.0
    try:
        evaluate.print_results(
            test_df,
            y_true_rules=y_true,
            y_pred_rules=y_pred_rules,
            metrics_rules=metrics_rules,
            y_true_ml=y_true,
            y_pred_ml=y_pred_ml,
            metrics_ml=metrics_ml,
        )
    except UnicodeEncodeError:
        print("Skipping verbose table output due to terminal encoding limitations.")

    split_payload = {
        "run_id": run_id,
        "seed": int(args.seed),
        "split_source": split_source,
        "train_ids": _serialize_split_ids(train_df),
        "val_ids": _serialize_split_ids(val_df),
        "test_ids": _serialize_split_ids(test_df),
    }
    split_ids_out.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    label_counts = ml_df["category"].value_counts().to_dict()
    metrics_payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "smoke_mode": bool(args.smoke),
        "gold_dataset": _dataset_file_info(gold_path),
        "warm_start_mode": warm_start_mode,
        "warm_start_input": warm_start_meta,
        "split_source": split_source,
        "split_ids_in": args.split_ids_in or "",
        "rows_total_used": int(len(ml_df)),
        "train_records": int(len(train_df)),
        "val_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "label_count": int(len(label_counts)),
        "top_labels": sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "baseline_train_metrics": _json_safe(train_metrics),
        "tfidf_params": _json_safe(pipeline.named_steps["tfidf"].get_params()),
        "vocab_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "rules_test_accuracy": float(metrics_rules["accuracy"]),
        "rules_test_macro_f1": float(metrics_rules["macro_f1"]),
        "rules_test_weighted_f1": float(metrics_rules["weighted_f1"]),
        "ml_test_accuracy": float(metrics_ml["accuracy"]),
        "ml_test_macro_f1": float(metrics_ml["macro_f1"]),
        "ml_test_weighted_f1": float(metrics_ml["weighted_f1"]),
        "rules_breakdown": rules_extra,
        "ml_breakdown": ml_extra,
        "hybrid_threshold": float(args.hybrid_threshold),
        "hybrid_rules_coverage": rules_coverage,
        "hybrid_ml_coverage": ml_coverage,
        "rules_other_rate": rules_other_rate,
        "ml_other_rate": ml_other_rate,
        "hybrid_other_rate": hybrid_other_rate,
        "hybrid_test_accuracy": hybrid_accuracy,
        "hybrid_test_macro_f1": hybrid_macro_f1,
        "hybrid_test_weighted_f1": hybrid_weighted_f1,
        "hybrid_breakdown": hybrid_extra,
        "key_categories": key_categories,
        "key_test_min_support": int(args.key_test_min_support),
        "test_support_key_categories": {
            cat: int((test_df["category"] == cat).sum()) for cat in key_categories
        },
        "needs_labeling_categories": needs_labeling_categories,
        "split_ids_out": str(split_ids_out),
        "output_pipeline": str(output_pipeline),
        "compat_pipeline_out": compat_path,
        "max_train_rows": int(args.max_train_rows),
        "text_case": str(args.text_case),
        "strip_currency_tokens": bool(args.strip_currency_tokens),
        "strip_country_tokens": bool(args.strip_country_tokens),
        "merchant_normalization": str(args.merchant_normalization),
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    split_ids_in_info = {}
    if args.split_ids_in:
        split_ids_in_path = Path(args.split_ids_in)
        if split_ids_in_path.exists():
            split_ids_in_info = _dataset_file_info(split_ids_in_path)

    run_index_path = _write_run_index(
        run_id=run_id,
        config={
            "seed": int(args.seed),
            "gold": str(gold_path),
            "smoke_mode": bool(args.smoke),
            "vectorizer_vocab_from": args.vectorizer_vocab_from,
            "pretrained_model": args.pretrained_model,
            "max_features": int(args.max_features),
            "max_train_rows": int(args.max_train_rows),
            "split_ids_in": args.split_ids_in,
            "require_split_ids_in": bool(args.require_split_ids_in),
            "text_case": str(args.text_case),
            "strip_currency_tokens": bool(args.strip_currency_tokens),
            "strip_country_tokens": bool(args.strip_country_tokens),
            "merchant_normalization": str(args.merchant_normalization),
            "hybrid_threshold": float(args.hybrid_threshold),
            "key_test_min_support": int(args.key_test_min_support),
            "key_categories": key_categories,
        },
        inputs={
            "gold_dataset": _dataset_file_info(gold_path),
            "warm_start_input": warm_start_meta,
            "split_ids_in": split_ids_in_info,
        },
        outputs={
            "pipeline": str(output_pipeline),
            "metrics": str(metrics_out),
            "split_ids": str(split_ids_out),
            "compat_pipeline_out": compat_path,
        },
    )
    print(f"Run ID: {run_id}")
    print(f"Metrics artifact: {metrics_out}")
    print(f"Split IDs artifact: {split_ids_out}")
    print(f"Run index artifact: {run_index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
