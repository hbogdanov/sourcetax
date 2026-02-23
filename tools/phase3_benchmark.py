#!/usr/bin/env python
"""
Phase 3 benchmark runner: rules vs TF-IDF vs ensemble (and optional SBERT).

Outputs:
- reports/phase3_eval.md
- reports/phase3_eval_assets/* (CSV + HTML charts)

This script is designed to run on a fresh machine with core deps only.
SBERT is optional and skipped if sentence-transformers is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from sourcetax.models import data_prep, train_baseline, visualize
from sourcetax.models import evaluate as model_eval


def _ensure_text_and_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "category" not in df.columns:
        for candidate in ("category_final", "label"):
            if candidate in df.columns:
                df["category"] = df[candidate]
                break
    if "text" not in df.columns:
        merchant = df["merchant"].fillna("").astype(str) if "merchant" in df.columns else ""
        desc = df["description"].fillna("").astype(str) if "description" in df.columns else ""
        if isinstance(merchant, str):
            merchant = pd.Series([""] * len(df))
        if isinstance(desc, str):
            desc = pd.Series([""] * len(df))
        df["text"] = (merchant + " " + desc).str.strip()
    return df


def load_or_build_splits(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_paths = {
        name: data_dir / f"ml_{name}.csv"
        for name in ("train", "val", "test")
    }
    if all(p.exists() for p in split_paths.values()):
        return tuple(_ensure_text_and_category(pd.read_csv(split_paths[n])) for n in ("train", "val", "test"))

    gold_records = data_prep.load_gold_set()
    ml_df = data_prep.prepare_ml_records(gold_records)
    if ml_df.empty:
        raise RuntimeError("No gold records available to build splits")
    train_df, val_df, test_df = data_prep.split_dataset(ml_df)
    data_prep.save_splits(train_df, val_df, test_df, output_dir=data_dir)
    return train_df, val_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, float], pd.DataFrame]:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "weighted_f1": float(np.average(f1, weights=support)) if len(f1) else 0.0,
    }
    per_class = pd.DataFrame(
        {
            "category": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).sort_values(["support", "category"], ascending=[False, True])
    return metrics, per_class


def write_confusion_csv(y_true: np.ndarray, y_pred: np.ndarray, out_csv: Path) -> None:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)


def rules_predict(test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    merchant_map = model_eval.load_merchant_category_map()
    preds = []
    confs = []
    for text in test_df["text"].astype(str):
        pred, conf = model_eval.apply_rules(text, merchant_map)
        preds.append(pred)
        confs.append(conf)
    return np.asarray(preds), np.asarray(confs, dtype=float)


def ensemble_predict(
    rules_pred: np.ndarray,
    rules_conf: np.ndarray,
    ml_pred: np.ndarray,
    threshold: float,
) -> np.ndarray:
    choose_rules = (rules_conf >= threshold) & (rules_pred != "Uncategorized")
    return np.where(choose_rules, rules_pred, ml_pred)


def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "| Model | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ml")
    parser.add_argument("--report", default="reports/phase3_eval.md")
    parser.add_argument("--assets-dir", default="reports/phase3_eval_assets")
    parser.add_argument("--rules-threshold", type=float, default=0.80)
    parser.add_argument("--allow-small", action="store_true", help="Allow n<30 without failing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_path = Path(args.report)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_or_build_splits(data_dir)
    train_df = _ensure_text_and_category(train_df)
    val_df = _ensure_text_and_category(val_df)
    test_df = _ensure_text_and_category(test_df)

    if test_df.empty:
        raise RuntimeError("Test split is empty")
    if len(test_df) < 30 and not args.allow_small:
        raise RuntimeError(
            f"Test split too small ({len(test_df)} rows). Use --allow-small for demo runs."
        )

    # Train TF-IDF baseline
    baseline_pipe, _ = train_baseline.train_baseline(train_df, val_df)
    y_true = test_df["category"].astype(str).to_numpy()
    y_pred_tfidf = baseline_pipe.predict(test_df["text"])

    # Rules and ensemble
    y_pred_rules, rules_conf = rules_predict(test_df)
    y_pred_ensemble = ensemble_predict(y_pred_rules, rules_conf, y_pred_tfidf, args.rules_threshold)

    predictions: Dict[str, np.ndarray] = {
        "Rules": y_pred_rules,
        "TF-IDF": y_pred_tfidf,
        "Ensemble": y_pred_ensemble,
    }

    # Optional SBERT
    sbert_error = None
    try:
        from sourcetax.models import train_sbert

        sbert_pipe, _ = train_sbert.train_sbert_classifier(
            train_df["text"].astype(str).tolist(),
            train_df["category"].astype(str).to_numpy(),
            val_df["text"].astype(str).tolist() if len(val_df) else None,
            val_df["category"].astype(str).to_numpy() if len(val_df) else None,
        )
        predictions["SBERT"] = sbert_pipe.predict(test_df["text"].astype(str).tolist())
    except ImportError as exc:
        sbert_error = f"SBERT skipped (optional dependency missing): {exc}"
    except Exception as exc:
        sbert_error = f"SBERT failed: {exc}"

    # Metrics and artifacts
    summary_results: Dict[str, Dict[str, float]] = {}
    per_class_files = []
    cm_files = []
    for name, pred in predictions.items():
        metrics, per_class = compute_metrics(y_true, pred)
        summary_results[name] = metrics

        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        per_class_path = assets_dir / f"per_category_{safe_name}.csv"
        cm_path = assets_dir / f"confusion_matrix_{safe_name}.csv"
        per_class.to_csv(per_class_path, index=False)
        write_confusion_csv(y_true, pred, cm_path)
        per_class_files.append((name, per_class_path))
        cm_files.append((name, cm_path))

    # HTML visual report
    label_names = sorted(np.unique(np.concatenate([y_true] + [p.astype(str) for p in predictions.values()])))
    html_paths = visualize.generate_evaluation_report(
        y_true.astype(str),
        {k: v.astype(str) for k, v in predictions.items()},
        label_names=label_names,
        output_dir=assets_dir,
    )

    # Markdown summary report
    notes = []
    if len(test_df) < 30:
        notes.append(
            f"- WARNING: test split has only {len(test_df)} rows; metrics are noisy until the gold set is expanded."
        )
    if sbert_error:
        notes.append(f"- {sbert_error}")

    lines = [
        "# Phase 3 Evaluation",
        "",
        "Rules vs TF-IDF vs Ensemble on the locked test split (SBERT included when available).",
        "",
        f"- Train rows: {len(train_df)}",
        f"- Val rows: {len(val_df)}",
        f"- Test rows: {len(test_df)}",
        f"- Ensemble policy: use rules when confidence >= {args.rules_threshold:.2f}, else TF-IDF",
        "",
    ]
    if notes:
        lines.extend(notes)
        lines.append("")

    lines.extend(
        [
            "## Summary Metrics",
            "",
            format_metrics_table(summary_results),
            "",
            "## Artifacts",
            "",
            f"- Markdown report: `{report_path}`",
            f"- HTML visual index: `{html_paths.get('index', '')}`",
            f"- HTML model comparison: `{html_paths.get('comparison', '')}`",
        ]
    )
    for name, path in cm_files:
        lines.append(f"- Confusion matrix CSV ({name}): `{path}`")
    for name, path in per_class_files:
        lines.append(f"- Per-category metrics CSV ({name}): `{path}`")

    if {"TF-IDF", "SBERT", "Ensemble"}.issubset(summary_results.keys()):
        lines.extend(
            [
                "",
                "## TF-IDF vs SBERT vs Ensemble",
                "",
                f"- TF-IDF macro F1: {summary_results['TF-IDF']['macro_f1']:.3f}",
                f"- SBERT macro F1: {summary_results['SBERT']['macro_f1']:.3f}",
                f"- Ensemble macro F1: {summary_results['Ensemble']['macro_f1']:.3f}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(f"Artifacts in {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
