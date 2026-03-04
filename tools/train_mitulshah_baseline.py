#!/usr/bin/env python
"""Train/evaluate TF-IDF baseline on Mitul Shah labels (internal sanity check only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sourcetax.text import normalize_text  # noqa: E402


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


def _resolve_output_path(value: str, default_path: Path) -> Path:
    return Path(value) if value else default_path


def _dataset_file_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _write_run_index(run_id: str, config: dict, inputs: dict, outputs: dict) -> Path:
    run_dir = Path("artifacts/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "run.json"
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "config": config,
        "inputs": inputs,
        "outputs": outputs,
    }
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_path


def _load_corpus(path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Corpus parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Corpus missing required columns: {sorted(missing)}")
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    if "country" not in df.columns:
        df["country"] = ""
    if "currency" not in df.columns:
        df["currency"] = ""
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    df["currency"] = df["currency"].fillna("").astype(str).str.strip()
    df = df[df["label"].str.len() > 0]
    if sample_size and sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df


def _prepare_text(
    df: pd.DataFrame,
    *,
    text_case: str,
    strip_currency_tokens: bool,
    strip_country_tokens: bool,
    append_country_currency: bool,
) -> pd.DataFrame:
    lowercase = text_case == "lower"
    out = df.copy()

    def _build(row: pd.Series) -> str:
        parts = [str(row.get("text", "") or "")]
        if append_country_currency:
            parts.extend([str(row.get("country", "") or ""), str(row.get("currency", "") or "")])
        merged = " ".join(p for p in parts if str(p).strip())
        return normalize_text(
            merged,
            remove_currency_tokens=strip_currency_tokens,
            remove_country_tokens=strip_country_tokens,
            lowercase=lowercase,
        )

    out["text"] = out.apply(_build, axis=1)
    out = out[out["text"].str.len() > 0]
    return out


def _split_train_val_test(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    if val_size <= 0 or test_size <= 0:
        raise SystemExit("val-size and test-size must both be > 0.")
    if val_size + test_size >= 1.0:
        raise SystemExit("val-size + test-size must be < 1.0.")

    label_counts = Counter(df["label"].tolist())
    can_stratify_first = min(label_counts.values()) >= 2
    train_df, temp_df = train_test_split(
        df,
        test_size=float(val_size + test_size),
        random_state=int(seed),
        stratify=df["label"] if can_stratify_first else None,
    )

    temp_counts = Counter(temp_df["label"].tolist())
    can_stratify_second = min(temp_counts.values()) >= 2 if temp_counts else False
    test_ratio_in_temp = float(test_size) / float(val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_in_temp,
        random_state=int(seed),
        stratify=temp_df["label"] if can_stratify_second else None,
    )
    return train_df, val_df, test_df


def _build_pipeline(seed: int, tfidf_params: dict) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _calc_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default="data/external/mitulshah_corpus_train.parquet",
        help="Path produced by tools/build_mitulshah_corpus.py",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500000,
        help="Deterministic sampling size for faster iteration. Use 0 for full corpus.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--run-id", default="", help="Optional run identifier.")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--metrics-out", default="")
    parser.add_argument(
        "--text-case",
        choices=["lower", "raw"],
        default="lower",
        help="Text preprocessing case mode.",
    )
    parser.add_argument(
        "--strip-currency-tokens",
        action="store_true",
        help="Remove common currency tokens during text normalization.",
    )
    parser.add_argument(
        "--strip-country-tokens",
        action="store_true",
        help="Remove common country tokens during text normalization.",
    )
    parser.add_argument(
        "--append-country-currency",
        action="store_true",
        help="Append country/currency columns into text before vectorization.",
    )
    parser.add_argument(
        "--warn-macro-f1-threshold",
        type=float,
        default=0.8,
        help="Warn if Mitul test macro F1 falls below this threshold.",
    )
    args = parser.parse_args()

    run_id = args.run_id or _now_run_id()
    model_out = _resolve_output_path(
        args.model_out, Path(f"artifacts/models/mitulshah_baseline_pipeline_{run_id}.joblib")
    )
    metrics_out = _resolve_output_path(
        args.metrics_out, Path(f"artifacts/metrics/mitulshah_baseline_metrics_{run_id}.json")
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    corpus_path = Path(args.corpus)
    df = _load_corpus(corpus_path, sample_size=int(args.sample_size), seed=int(args.seed))
    df = _prepare_text(
        df,
        text_case=str(args.text_case),
        strip_currency_tokens=bool(args.strip_currency_tokens),
        strip_country_tokens=bool(args.strip_country_tokens),
        append_country_currency=bool(args.append_country_currency),
    )
    if len(df) < 1000:
        raise SystemExit(f"Not enough rows for baseline training: {len(df)}")

    train_df, val_df, test_df = _split_train_val_test(
        df,
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )

    tfidf_params = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 1.0,
        "max_features": int(args.max_features),
        "lowercase": False,
        "stop_words": "english",
    }
    pipe = _build_pipeline(seed=int(args.seed), tfidf_params=tfidf_params)
    pipe.fit(train_df["text"], train_df["label"])

    y_val = pipe.predict(val_df["text"])
    y_test = pipe.predict(test_df["text"])
    m_val = _calc_metrics(val_df["label"], y_val)
    m_test = _calc_metrics(test_df["label"], y_test)

    labels_sorted = sorted(df["label"].unique().tolist())
    report = classification_report(test_df["label"], y_test, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_df["label"], y_test, labels=labels_sorted)
    test_label_counts = Counter(test_df["label"].tolist())
    top10_test_labels = [label for label, _count in test_label_counts.most_common(10)]

    per_class_all = {}
    per_class_top10 = {}
    for label in labels_sorted:
        if label not in report:
            continue
        item = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        per_class_all[label] = item
        if label in top10_test_labels:
            per_class_top10[label] = item

    metrics = {
        "run_id": run_id,
        "dataset": "mitulshah/transaction-categorization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "dataset_info": _dataset_file_info(corpus_path),
        "rows_total_used": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "sample_size_requested": int(args.sample_size),
        "labels_unique": int(df["label"].nunique()),
        "top10_labels_all_rows": Counter(df["label"].tolist()).most_common(10),
        "top10_labels_test_rows": test_label_counts.most_common(10),
        "split_policy": {
            "val_size": float(args.val_size),
            "test_size": float(args.test_size),
        },
        "preprocessing": {
            "text_case": str(args.text_case),
            "strip_currency_tokens": bool(args.strip_currency_tokens),
            "strip_country_tokens": bool(args.strip_country_tokens),
            "append_country_currency": bool(args.append_country_currency),
        },
        "tfidf_params": tfidf_params,
        "logreg_params": {"max_iter": 400, "class_weight": "balanced", "random_state": int(args.seed)},
        "vocab_size": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        "val_accuracy": m_val["accuracy"],
        "val_macro_f1": m_val["macro_f1"],
        "val_weighted_f1": m_val["weighted_f1"],
        "test_accuracy": m_test["accuracy"],
        "test_macro_f1": m_test["macro_f1"],
        "test_weighted_f1": m_test["weighted_f1"],
        "accuracy": m_test["accuracy"],
        "macro_f1": m_test["macro_f1"],
        "weighted_f1": m_test["weighted_f1"],
        "per_class_metrics": per_class_all,
        "per_class_top10_test": per_class_top10,
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm.tolist(),
        },
        "model_out": str(model_out),
    }

    import joblib

    joblib.dump(pipe, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_index_path = _write_run_index(
        run_id=run_id,
        config={
            "seed": int(args.seed),
            "corpus": str(corpus_path),
            "sample_size": int(args.sample_size),
            "val_size": float(args.val_size),
            "test_size": float(args.test_size),
            "max_features": int(args.max_features),
            "text_case": str(args.text_case),
            "strip_currency_tokens": bool(args.strip_currency_tokens),
            "strip_country_tokens": bool(args.strip_country_tokens),
            "append_country_currency": bool(args.append_country_currency),
        },
        inputs={"corpus": _dataset_file_info(corpus_path)},
        outputs={"model": str(model_out), "metrics": str(metrics_out)},
    )

    print("Mitul baseline training complete.")
    print(f"- run_id: {run_id}")
    print(f"- model_out: {model_out}")
    print(f"- metrics_out: {metrics_out}")
    print(f"- run_index: {run_index_path}")
    print(f"- val_macro_f1: {metrics['val_macro_f1']:.4f}")
    print(f"- test_macro_f1: {metrics['test_macro_f1']:.4f}")
    if metrics["test_macro_f1"] < float(args.warn_macro_f1_threshold):
        print(
            f"WARNING: test macro F1 below threshold ({metrics['test_macro_f1']:.4f} < {args.warn_macro_f1_threshold:.2f})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
