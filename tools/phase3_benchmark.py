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
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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
warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'.*",
    category=UserWarning,
)

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


def ensemble_mask(
    rules_pred: np.ndarray,
    rules_conf: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (rules_conf >= threshold) & (rules_pred != "Uncategorized")


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


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, limit: int = 10) -> List[Tuple[str, str, int]]:
    counts: Dict[Tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        key = (str(t), str(p))
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [(a, b, c) for (a, b), c in ranked[:limit]]


def top_error_examples(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    rows = []
    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        if str(truth) == str(pred):
            continue
        row = test_df.iloc[idx]
        rows.append(
            {
                "transaction_id": row.get("id", ""),
                "merchant": row.get("merchant", ""),
                "description": row.get("description", ""),
                "text": row.get("text", ""),
                "true_category": truth,
                "predicted_category": pred,
                "confidence": float(confidences[idx]) if confidences is not None else None,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "confidence" in out.columns:
        out = out.sort_values(by=["confidence"], ascending=[True], na_position="last")
    return out.head(limit)


def export_needs_review_csv(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_rules: np.ndarray,
    rules_conf: np.ndarray,
    y_pred_tfidf: np.ndarray,
    y_proba_tfidf: np.ndarray | None,
    out_path: Path,
    low_conf_threshold: float = 0.70,
    top_k: int = 3,
) -> Path:
    """Export a simple human-in-the-loop queue for ambiguous benchmark rows."""
    rows = []
    classes = None
    if y_proba_tfidf is not None:
        # `classes_` order comes from the trained pipeline's classifier; caller aligns this.
        # Stored on dataframe attrs by caller if available.
        classes = test_df.attrs.get("tfidf_classes")
    for i in range(len(test_df)):
        reasons = []
        if float(rules_conf[i]) < low_conf_threshold:
            reasons.append("low_rules_confidence")
        if str(y_pred_rules[i]) != str(y_pred_tfidf[i]):
            reasons.append("rules_ml_conflict")
        top_preds = []
        if y_proba_tfidf is not None and classes is not None:
            order = np.argsort(y_proba_tfidf[i])[::-1][:top_k]
            top_preds = [
                {"category": str(classes[j]), "prob": float(y_proba_tfidf[i][j])}
                for j in order
            ]
            if float(top_preds[0]["prob"]) < low_conf_threshold:
                reasons.append("low_ml_confidence")
        if not reasons:
            continue
        row = test_df.iloc[i]
        rows.append(
            {
                "transaction_id": row.get("id", ""),
                "merchant_raw": row.get("merchant", ""),
                "date": row.get("transaction_date", ""),
                "amount": row.get("amount", ""),
                "true_category": y_true[i],
                "rules_category": y_pred_rules[i],
                "rules_confidence": float(rules_conf[i]),
                "ml_category": y_pred_tfidf[i],
                "ml_top1_prob": float(top_preds[0]["prob"]) if top_preds else "",
                "ensemble_reason": "|".join(sorted(set(reasons))),
                "ml_top3": json.dumps(top_preds),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ml")
    parser.add_argument("--report", default="reports/phase3_eval.md")
    parser.add_argument("--assets-dir", default="reports/phase3_eval_assets")
    parser.add_argument("--rules-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-small", action="store_true", help="Allow n<30 without failing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_path = Path(args.report)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    git_hash = _git_commit_hash()

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
    y_proba_tfidf = baseline_pipe.predict_proba(test_df["text"])
    test_df.attrs["tfidf_classes"] = baseline_pipe.named_steps["classifier"].classes_

    # Rules and ensemble
    y_pred_rules, rules_conf = rules_predict(test_df)
    route_rules_mask = ensemble_mask(y_pred_rules, rules_conf, args.rules_threshold)
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

    needs_review_path = export_needs_review_csv(
        test_df=test_df,
        y_true=y_true,
        y_pred_rules=y_pred_rules,
        rules_conf=rules_conf,
        y_pred_tfidf=y_pred_tfidf,
        y_proba_tfidf=y_proba_tfidf,
        out_path=assets_dir / "needs_review.csv",
        low_conf_threshold=args.rules_threshold,
    )

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
        "## Run Metadata",
        "",
        f"- Run ID: `{run_id}`",
        f"- Seed: `{args.seed}`",
        f"- Git commit: `{git_hash}`",
        f"- Data dir: `{data_dir}`",
        "",
        "## Dataset / Split Info",
        "",
        f"- Train rows: {len(train_df)}",
        f"- Val rows: {len(val_df)}",
        f"- Test rows: {len(test_df)}",
        f"- Ensemble policy: use rules when confidence >= {args.rules_threshold:.2f}, else TF-IDF",
        f"- Ensemble routed to rules: {route_rules_mask.sum()} / {len(route_rules_mask)} ({route_rules_mask.mean():.1%})",
        f"- Ensemble routed to TF-IDF fallback: {(~route_rules_mask).sum()} / {len(route_rules_mask)} ({(~route_rules_mask).mean():.1%})",
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
            f"- Needs-review queue CSV: `{needs_review_path}`",
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

    confusions = top_confusions(y_true, y_pred_ensemble, limit=10)
    lines.extend(["", "## Top Confusions (Ensemble)", ""])
    if confusions:
        lines.append("| True | Predicted | Count |")
        lines.append("|---|---|---:|")
        for true_label, pred_label, count in confusions:
            lines.append(f"| {true_label} | {pred_label} | {count} |")
    else:
        lines.append("- No ensemble misclassifications on the current test split.")

    error_examples = top_error_examples(
        test_df,
        y_true=y_true,
        y_pred=y_pred_ensemble,
        confidences=np.maximum(rules_conf, y_proba_tfidf.max(axis=1) if y_proba_tfidf is not None else 0),
        limit=10,
    )
    if not error_examples.empty:
        error_examples_path = assets_dir / "top_error_examples_ensemble.csv"
        error_examples.to_csv(error_examples_path, index=False)
        lines.extend(
            [
                "",
                "## Top Error Examples (Ensemble)",
                "",
                f"- CSV: `{error_examples_path}`",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(f"Artifacts in {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
