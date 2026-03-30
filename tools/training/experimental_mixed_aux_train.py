#!/usr/bin/env python
"""Experimental gold + synthetic auxiliary training on a locked gold holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from sourcetax.models import data_prep, train_baseline
from sourcetax.normalization import normalize_merchant_name
from sourcetax.text import combine_text_fields

WEAK_CATEGORIES = [
    "COGS",
    "Payroll & Contractors",
    "Taxes & Licenses",
    "Insurance",
    "Professional Services",
    "Financial Fees",
    "Vehicle Expenses",
    "Rent & Utilities",
]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _serialize_split_ids(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for idx, row in df.iterrows():
        rid = row.get("id")
        if rid is None or str(rid).strip() == "":
            out.append(f"row_index:{idx}")
        else:
            out.append(str(rid))
    return out


def _apply_split_ids(df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    id_index = {}
    for idx, row in df.iterrows():
        rid = row.get("id")
        key = str(rid) if rid is not None and str(rid).strip() else f"row_index:{idx}"
        id_index[key] = idx
    rows = [id_index[x] for x in ids if x in id_index]
    return df.loc[rows].copy()


def _build_gold_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _norm_merchant(value: str) -> str:
        return normalize_merchant_name(str(value or ""), case="lower")

    out["text"] = out.apply(
        lambda row: combine_text_fields(
            [
                _norm_merchant(row.get("merchant", "")),
                str(row.get("description", "") or ""),
            ],
            lowercase=True,
        ),
        axis=1,
    )
    out = out[out["text"].str.len() > 0]
    return out


def _build_synth_df(path: Path) -> pd.DataFrame:
    rows = _load_jsonl(path)
    out_rows = []
    for idx, row in enumerate(rows):
        category = str(row.get("category_mapped") or "").strip()
        if not category:
            continue
        merchant = str(row.get("merchant_raw") or "").strip()
        merchant_norm = normalize_merchant_name(merchant, case="lower") if merchant else ""
        description = str(row.get("description") or "").strip()
        text = combine_text_fields([merchant_norm or merchant, description], lowercase=True)
        if not text:
            continue
        out_rows.append(
            {
                "id": f"synthetic_aux:{row.get('source_record_id') or idx}",
                "text": text,
                "merchant": merchant,
                "description": description,
                "category": category,
                "source": row.get("source") or "synthetic_gapfill",
                "is_synthetic": True,
            }
        )
    return pd.DataFrame(out_rows)


def _synthetic_row_cap(gold_train_rows: int, synthetic_ratio: float, synthetic_rows_available: int) -> int:
    requested = int(gold_train_rows * float(synthetic_ratio))
    return max(0, min(requested, int(synthetic_rows_available)))


def _per_class_and_confusion(y_true, y_pred) -> dict:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--synthetic-jsonl", default="data/ml/staging_training_rows_gapfill.jsonl")
    parser.add_argument(
        "--split-ids",
        default="artifacts/reports/gold_ml_baseline_split_ids_20260330T185251Z.json",
        help="Locked gold split manifest.",
    )
    parser.add_argument("--synthetic-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--baseline-metrics",
        default="artifacts/synthetic/gold_only_metrics.json",
        help="Optional baseline metrics JSON for direct comparison.",
    )
    args = parser.parse_args()

    split_path = Path(args.split_ids)
    if not split_path.exists():
        raise SystemExit(f"Missing split IDs file: {split_path}")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))

    run_id = args.run_id or f"mixed_aux_{int(args.synthetic_ratio * 100):02d}_{int(args.seed)}"
    metrics_out = Path(args.metrics_out) if args.metrics_out else Path(f"artifacts/synthetic/{run_id}_metrics.json")
    report_out = Path(args.report_out) if args.report_out else Path(f"artifacts/synthetic/{run_id}_report.md")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    gold_records = data_prep.load_gold_set(Path(args.gold))
    gold_df = data_prep.prepare_ml_records(gold_records)
    gold_df = _build_gold_text_features(gold_df)
    train_df = _apply_split_ids(gold_df, split_payload.get("train_ids", []))
    val_df = _apply_split_ids(gold_df, split_payload.get("val_ids", []))
    test_df = _apply_split_ids(gold_df, split_payload.get("test_ids", []))

    synth_df = _build_synth_df(Path(args.synthetic_jsonl))
    synth_cap = _synthetic_row_cap(len(train_df), float(args.synthetic_ratio), len(synth_df))
    synth_sample = (
        synth_df.sample(n=synth_cap, random_state=int(args.seed)).copy() if synth_cap > 0 else synth_df.head(0).copy()
    )

    mixed_train = pd.concat([train_df.assign(is_synthetic=False), synth_sample], ignore_index=True)
    pipeline, train_metrics = train_baseline.train_baseline(
        mixed_train[["text", "category"]],
        val_df[["text", "category"]],
        random_state=int(args.seed),
        tfidf_params={
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "max_features": 100000,
            "lowercase": True,
            "stop_words": "english",
        },
    )

    y_true = test_df["category"].astype(str).to_numpy()
    y_pred = pipeline.predict(test_df["text"])
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    breakdown = _per_class_and_confusion(y_true, y_pred)

    baseline_payload = {}
    baseline_path = Path(args.baseline_metrics)
    if baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    weak_category_deltas = {}
    base_pc = (baseline_payload.get("ml_breakdown") or {}).get("per_class_metrics", {})
    cur_pc = (breakdown.get("per_class_metrics") or {})
    for cat in WEAK_CATEGORIES:
        base_f1 = float((base_pc.get(cat) or {}).get("f1", 0.0))
        cur_f1 = float((cur_pc.get(cat) or {}).get("f1", 0.0))
        weak_category_deltas[cat] = {
            "gold_only_ml_f1": base_f1,
            "mixed_aux_f1": cur_f1,
            "delta_f1": cur_f1 - base_f1,
        }

    payload = {
        "run_id": run_id,
        "seed": int(args.seed),
        "synthetic_ratio_requested": float(args.synthetic_ratio),
        "gold_train_rows": int(len(train_df)),
        "gold_val_rows": int(len(val_df)),
        "gold_test_rows": int(len(test_df)),
        "synthetic_rows_available": int(len(synth_df)),
        "synthetic_rows_used": int(len(synth_sample)),
        "synthetic_share_of_mixed_train": float(len(synth_sample) / len(mixed_train)) if len(mixed_train) else 0.0,
        "train_metrics": _json_safe(train_metrics),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "breakdown": breakdown,
        "split_ids": str(split_path),
        "gold_path": args.gold,
        "synthetic_jsonl": args.synthetic_jsonl,
        "baseline_metrics_path": str(baseline_path) if baseline_path.exists() else "",
        "delta_vs_gold_only_ml_accuracy": accuracy - float(baseline_payload.get("ml_test_accuracy", 0.0)) if baseline_payload else None,
        "delta_vs_gold_only_ml_macro_f1": macro_f1 - float(baseline_payload.get("ml_test_macro_f1", 0.0)) if baseline_payload else None,
        "delta_vs_gold_only_ml_weighted_f1": weighted_f1 - float(baseline_payload.get("ml_test_weighted_f1", 0.0)) if baseline_payload else None,
        "weak_category_deltas": weak_category_deltas,
    }
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Experimental Mixed Auxiliary Training",
        "",
        f"- Run ID: `{run_id}`",
        f"- Synthetic ratio requested: `{float(args.synthetic_ratio):.2f}`",
        f"- Gold train rows: `{len(train_df)}`",
        f"- Synthetic rows used: `{len(synth_sample)}`",
        f"- Synthetic share of mixed train: `{payload['synthetic_share_of_mixed_train']:.4f}`",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: `{accuracy:.4f}`",
        f"- Macro F1: `{macro_f1:.4f}`",
        f"- Weighted F1: `{weighted_f1:.4f}`",
    ]
    if baseline_payload:
        lines.extend(
            [
                "",
                "## Delta vs Gold-Only ML Baseline",
                "",
                f"- d_accuracy: `{payload['delta_vs_gold_only_ml_accuracy']:+.4f}`",
                f"- d_macro_f1: `{payload['delta_vs_gold_only_ml_macro_f1']:+.4f}`",
                f"- d_weighted_f1: `{payload['delta_vs_gold_only_ml_weighted_f1']:+.4f}`",
                "",
                "## Weak-Category F1",
                "",
                "| category | gold_only_ml_f1 | mixed_aux_f1 | delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for cat in WEAK_CATEGORIES:
            cat_delta = weak_category_deltas[cat]
            lines.append(
                f"| {cat} | {cat_delta['gold_only_ml_f1']:.4f} | {cat_delta['mixed_aux_f1']:.4f} | {cat_delta['delta_f1']:+.4f} |"
            )
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote metrics to {metrics_out}")
    print(f"Wrote report to {report_out}")
    print(f"synthetic_rows_used={len(synth_sample)} synthetic_share={payload['synthetic_share_of_mixed_train']:.4f}")
    print(f"test_accuracy={accuracy:.4f} test_macro_f1={macro_f1:.4f} test_weighted_f1={weighted_f1:.4f}")
    if baseline_payload:
        print(f"delta_vs_gold_only_ml_macro_f1={payload['delta_vs_gold_only_ml_macro_f1']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
