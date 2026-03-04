#!/usr/bin/env python
"""Run apples-to-apples model comparison on a shared split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KEY_GOLD_LABELS = [
    "Repairs & Maintenance",
    "Rent & Utilities",
    "Financial Fees",
    "Income",
    "Meals & Entertainment",
]


def _run(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
        "config": config,
        "inputs": inputs,
        "outputs": outputs,
    }
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_path


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(name: str, m: dict) -> dict:
    return {
        "name": name,
        "accuracy": float(m.get("ml_test_accuracy", 0.0)),
        "macro_f1": float(m.get("ml_test_macro_f1", 0.0)),
        "weighted_f1": float(m.get("ml_test_weighted_f1", 0.0)),
        "top_labels": m.get("top_labels", []),
        "per_class": (m.get("ml_breakdown") or {}).get("per_class_metrics", {}),
        "confusion_matrix": (m.get("ml_breakdown") or {}).get("confusion_matrix", {}),
        "rules_coverage": None,
        "ml_coverage": None,
        "other_rate": float(m.get("ml_other_rate", 0.0)),
    }


def _rules_row(m: dict) -> dict:
    return {
        "name": "rules_only",
        "accuracy": float(m.get("rules_test_accuracy", 0.0)),
        "macro_f1": float(m.get("rules_test_macro_f1", 0.0)),
        "weighted_f1": float(m.get("rules_test_weighted_f1", 0.0)),
        "top_labels": m.get("top_labels", []),
        "per_class": ((m.get("rules_breakdown") or {}).get("per_class_metrics", {})),
        "confusion_matrix": ((m.get("rules_breakdown") or {}).get("confusion_matrix", {})),
        "rules_coverage": 1.0,
        "ml_coverage": 0.0,
        "other_rate": float(m.get("rules_other_rate", 0.0)),
    }


def _hybrid_row(m: dict) -> dict | None:
    if "hybrid_test_accuracy" not in m:
        return None
    return {
        "name": "rules_first_hybrid",
        "accuracy": float(m.get("hybrid_test_accuracy", 0.0)),
        "macro_f1": float(m.get("hybrid_test_macro_f1", 0.0)),
        "weighted_f1": float(m.get("hybrid_test_weighted_f1", 0.0)),
        "top_labels": m.get("top_labels", []),
        "per_class": (m.get("hybrid_breakdown") or {}).get("per_class_metrics", {}),
        "confusion_matrix": (m.get("hybrid_breakdown") or {}).get("confusion_matrix", {}),
        "rules_coverage": float(m.get("hybrid_rules_coverage", 0.0)),
        "ml_coverage": float(m.get("hybrid_ml_coverage", 0.0)),
        "other_rate": float(m.get("hybrid_other_rate", 0.0)),
    }


def _assert_locked_split(metrics_path: Path, expected_split_ids: Path) -> None:
    payload = _load_metrics(metrics_path)
    split_source = str(payload.get("split_source", ""))
    split_in = str(payload.get("split_ids_in", ""))
    if not split_source.startswith("from_file:"):
        raise SystemExit(
            f"Strict split lock failed for {metrics_path}: expected split_source from_file, got {split_source!r}"
        )
    if Path(split_in).resolve() != expected_split_ids.resolve():
        raise SystemExit(
            f"Strict split lock failed for {metrics_path}: expected split_ids_in={expected_split_ids}, got {split_in}"
        )


def _category_f1(row: dict, category: str) -> float:
    return float((row.get("per_class", {}).get(category) or {}).get("f1", 0.0))


def _confusion_pairs(row: dict) -> dict:
    cm = row.get("confusion_matrix") or {}
    labels = cm.get("labels") or []
    matrix = cm.get("matrix") or []
    out = {}
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int((matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0) or 0)
            if count > 0:
                out[f"{true_label} -> {pred_label}"] = count
    return out


def _top_confusion_deltas(baseline: dict, candidate: dict, n: int = 10):
    base = _confusion_pairs(baseline)
    cand = _confusion_pairs(candidate)
    keys = set(base.keys()) | set(cand.keys())
    deltas = []
    for key in keys:
        b = int(base.get(key, 0))
        c = int(cand.get(key, 0))
        d = c - b
        if d != 0:
            deltas.append((key, b, c, d))
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)
    return deltas[:n]


def _markdown_report(rows, baseline_name: str) -> str:
    baseline = next((r for r in rows if r["name"] == baseline_name), rows[0])
    lines = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append("| config | accuracy | macro_f1 | weighted_f1 |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |"
        )
    lines.append("")
    lines.append("## Delta vs Baseline")
    lines.append("")
    lines.append("| config | d_accuracy | d_macro_f1 | d_weighted_f1 |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['accuracy']-baseline['accuracy']:+.4f} | "
            f"{r['macro_f1']-baseline['macro_f1']:+.4f} | {r['weighted_f1']-baseline['weighted_f1']:+.4f} |"
        )
    lines.append("")
    lines.append("## ROI Table (Gold Transfer)")
    lines.append("")
    lines.append(
        "| model | accuracy | macro_f1 | repairs_maint_f1 | rent_util_f1 | financial_fees_f1 | income_f1 | meals_f1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{_category_f1(r, 'Repairs & Maintenance'):.4f} | "
            f"{_category_f1(r, 'Rent & Utilities'):.4f} | "
            f"{_category_f1(r, 'Financial Fees'):.4f} | "
            f"{_category_f1(r, 'Income'):.4f} | "
            f"{_category_f1(r, 'Meals & Entertainment'):.4f} |"
        )
    lines.append("")
    lines.append("## Routing Coverage")
    lines.append("")
    lines.append("| model | rules_coverage | ml_coverage | other_rate |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        rc = r.get("rules_coverage")
        mc = r.get("ml_coverage")
        other = float(r.get("other_rate", 0.0))
        if rc is None or mc is None:
            lines.append(f"| {r['name']} | n/a | n/a | {other:.4f} |")
        else:
            lines.append(f"| {r['name']} | {float(rc):.4f} | {float(mc):.4f} | {other:.4f} |")
    lines.append("")
    lines.append("## Top Categories by Support (from baseline)")
    lines.append("")
    top = baseline.get("top_labels", [])[:10]
    lines.append("| category | support |")
    lines.append("|---|---:|")
    for cat, n in top:
        lines.append(f"| {cat} | {n} |")
    top_categories = [c for c, _n in top]
    for r in rows:
        if r["name"] == baseline_name:
            continue
        lines.append("")
        lines.append(f"## Per-Class F1 Delta vs Baseline ({r['name']})")
        lines.append("")
        lines.append("| category | baseline_f1 | candidate_f1 | delta |")
        lines.append("|---|---:|---:|---:|")
        base_pc = baseline.get("per_class", {})
        cand_pc = r.get("per_class", {})
        for cat in top_categories:
            b = float((base_pc.get(cat) or {}).get("f1", 0.0))
            c = float((cand_pc.get(cat) or {}).get("f1", 0.0))
            lines.append(f"| {cat} | {b:.4f} | {c:.4f} | {c-b:+.4f} |")
        lines.append("")
        lines.append(f"## Confusion Pair Delta vs Baseline ({r['name']})")
        lines.append("")
        lines.append("| confusion_pair | baseline_count | candidate_count | delta |")
        lines.append("|---|---:|---:|---:|")
        for pair, b, c, d in _top_confusion_deltas(baseline, r, n=10):
            lines.append(f"| {pair} | {b} | {c} | {d:+d} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--vocab-from",
        default="data/external/mitulshah_corpus_train.parquet",
        help="Parquet for vocab warm-start config.",
    )
    parser.add_argument(
        "--pretrained-model",
        default="",
        help="Optional pretrained pipeline for warm-start config.",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Run comparison using only gold data (skip external warm-start variants).",
    )
    parser.add_argument(
        "--key-test-min-support",
        type=int,
        default=0,
        help="If >0, enforce minimum test support for key categories when generating baseline splits.",
    )
    parser.add_argument(
        "--key-categories",
        default="Repairs & Maintenance,Rent & Utilities,Financial Fees,Income,Meals & Entertainment",
        help="Comma-separated key categories used with --key-test-min-support.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_metrics = Path(f"artifacts/metrics/gold_ml_baseline_metrics_{run_id}_baseline.json")
    base_splits = Path(f"artifacts/reports/gold_ml_baseline_split_ids_{run_id}_baseline.json")
    all_metrics_paths = [base_metrics]

    common = [sys.executable, "tools/train_ml_baseline.py", "--gold", args.gold, "--seed", str(args.seed)]
    _run(
        common
        + [
            "--run-id",
            f"{run_id}_baseline",
            "--metrics-out",
            str(base_metrics),
            "--split-ids-out",
            str(base_splits),
            "--key-test-min-support",
            str(args.key_test_min_support),
            "--key-categories",
            str(args.key_categories),
        ]
    )

    baseline_metrics_payload = _load_metrics(base_metrics)
    metrics_rows = [_rules_row(baseline_metrics_payload), _metric_row("ml_baseline", baseline_metrics_payload)]
    hybrid_baseline = _hybrid_row(baseline_metrics_payload)
    if hybrid_baseline is not None:
        metrics_rows.append(hybrid_baseline)

    vocab_path = Path(args.vocab_from)
    if args.gold_only:
        print("Gold-only mode enabled: skipping external warm-start variants.")
    elif vocab_path.exists():
        vocab_metrics = Path(f"artifacts/metrics/gold_ml_baseline_metrics_{run_id}_warm_vocab.json")
        _run(
            common
            + [
                "--run-id",
                f"{run_id}_warm_vocab",
                "--split-ids-in",
                str(base_splits),
                "--require-split-ids-in",
                "--metrics-out",
                str(vocab_metrics),
                "--vectorizer-vocab-from",
                args.vocab_from,
            ]
        )
        _assert_locked_split(vocab_metrics, base_splits)
        all_metrics_paths.append(vocab_metrics)
        metrics_rows.append(_metric_row("ml_warm_vocab", _load_metrics(vocab_metrics)))
        hybrid_vocab = _hybrid_row(_load_metrics(vocab_metrics))
        if hybrid_vocab is not None:
            hybrid_vocab["name"] = "hybrid_warm_vocab"
            metrics_rows.append(hybrid_vocab)
    else:
        print(f"Skipping warm vocab config; file not found: {vocab_path}")

    if not args.gold_only and args.pretrained_model:
        pre_metrics = Path(f"artifacts/metrics/gold_ml_baseline_metrics_{run_id}_warm_pretrained.json")
        _run(
            common
            + [
                "--run-id",
                f"{run_id}_warm_pretrained",
                "--split-ids-in",
                str(base_splits),
                "--require-split-ids-in",
                "--metrics-out",
                str(pre_metrics),
                "--pretrained-model",
                args.pretrained_model,
            ]
        )
        _assert_locked_split(pre_metrics, base_splits)
        all_metrics_paths.append(pre_metrics)
        metrics_rows.append(_metric_row("ml_warm_pretrained", _load_metrics(pre_metrics)))
        hybrid_pre = _hybrid_row(_load_metrics(pre_metrics))
        if hybrid_pre is not None:
            hybrid_pre["name"] = "hybrid_warm_pretrained"
            metrics_rows.append(hybrid_pre)

    report_json = out_dir / f"model_comparison_{run_id}.json"
    report_md = out_dir / f"model_comparison_{run_id}.md"
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": metrics_rows,
        "roi_table": [
            {
                "model": r["name"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "key_class_f1": {label: _category_f1(r, label) for label in KEY_GOLD_LABELS},
            }
            for r in metrics_rows
        ],
        "confusion_delta_vs_baseline": {
            r["name"]: [
                {
                    "pair": pair,
                    "baseline_count": b,
                    "candidate_count": c,
                    "delta": d,
                }
                for pair, b, c, d in _top_confusion_deltas(
                    next((x for x in metrics_rows if x["name"] == "ml_baseline"), metrics_rows[0]),
                    r,
                    n=15,
                )
            ]
            for r in metrics_rows
            if r["name"] != "ml_baseline"
        },
        "baseline_split_ids": str(base_splits),
        "baseline_metrics": str(base_metrics),
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_markdown_report(metrics_rows, baseline_name="ml_baseline"), encoding="utf-8")
    inputs = {}
    gold_path = Path(args.gold)
    if gold_path.exists():
        inputs["gold_dataset"] = _dataset_file_info(gold_path)
    if vocab_path.exists():
        inputs["vocab_from"] = _dataset_file_info(vocab_path)
    if args.pretrained_model and Path(args.pretrained_model).exists():
        inputs["pretrained_model"] = _dataset_file_info(Path(args.pretrained_model))
    run_index_path = _write_run_index(
        run_id=run_id,
        config={
            "seed": int(args.seed),
            "gold": args.gold,
            "vocab_from": args.vocab_from,
            "pretrained_model": args.pretrained_model,
            "gold_only": bool(args.gold_only),
            "strict_split_lock": True,
            "key_test_min_support": int(args.key_test_min_support),
            "key_categories": str(args.key_categories),
        },
        inputs=inputs,
        outputs={
            "baseline_split_ids": str(base_splits),
            "metrics": [str(p) for p in all_metrics_paths],
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    )
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(f"Wrote {run_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
