#!/usr/bin/env python
"""Run end-to-end transfer evaluation and write one consolidated report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{float(v):.4f}"


def _category_f1(row: dict, key: str) -> float:
    return float((row.get("key_class_f1") or {}).get(key, 0.0))


def _find_roi_row(payload: dict, model_name: str) -> dict:
    for r in payload.get("roi_table", []):
        if r.get("model") == model_name:
            return r
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--mitul-corpus", default="data/external/mitulshah_corpus_train.parquet")
    parser.add_argument("--mitul-sample-size", type=int, default=500000)
    parser.add_argument("--mitul-val-size", type=float, default=0.1)
    parser.add_argument("--mitul-test-size", type=float, default=0.1)
    parser.add_argument("--mitul-max-features", type=int, default=100000)
    parser.add_argument(
        "--key-test-min-support",
        type=int,
        default=0,
        help="Minimum test support for key categories when generating gold baseline split.",
    )
    parser.add_argument(
        "--key-categories",
        default="Repairs & Maintenance,Rent & Utilities,Financial Fees,Income,Meals & Entertainment",
        help="Comma-separated key categories used with --key-test-min-support.",
    )
    parser.add_argument(
        "--strict-mitul",
        action="store_true",
        help="Fail if Mitul corpus is unavailable. Default: skip Mitul stages and still run gold comparison.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    mitul_sanity_metrics = Path(f"artifacts/metrics/mitulshah_baseline_metrics_{run_id}_mitul_sanity.json")
    mitul_robustness_json = reports_dir / f"mitul_robustness_{run_id}_mitul_robustness.json"
    gold_ref_json = reports_dir / f"model_comparison_{run_id}_gold_ref.json"
    transfer_json = reports_dir / f"model_comparison_{run_id}_transfer.json"
    consolidated_json = reports_dir / f"eval_transfer_{run_id}.json"
    consolidated_md = reports_dir / f"eval_transfer_{run_id}.md"

    steps = []
    allow_skip_mitul = not bool(args.strict_mitul)
    mitul_corpus_path = Path(args.mitul_corpus)
    mitul_available = mitul_corpus_path.exists()
    external_dir = Path("data/external")
    parquet_candidates = []
    if external_dir.exists():
        parquet_candidates = sorted(str(p) for p in external_dir.glob("*.parquet"))
    print(f"Mitul expected corpus path: {mitul_corpus_path}")
    print(f"Mitul corpus found: {'yes' if mitul_available else 'no'}")
    if not mitul_available and parquet_candidates:
        print("Nearby data/external/*.parquet candidates:")
        for candidate in parquet_candidates[:10]:
            print(f"- {candidate}")
    if not mitul_available and args.strict_mitul:
        raise SystemExit(f"Mitul corpus not found in strict mode: {mitul_corpus_path}")

    # A) Mitul sanity evaluation
    sanity_cmd = [
        sys.executable,
        "tools/train_mitulshah_baseline.py",
        "--corpus",
        args.mitul_corpus,
        "--sample-size",
        str(args.mitul_sample_size),
        "--val-size",
        str(args.mitul_val_size),
        "--test-size",
        str(args.mitul_test_size),
        "--seed",
        str(args.seed),
        "--max-features",
        str(args.mitul_max_features),
        "--run-id",
        f"{run_id}_mitul_sanity",
        "--metrics-out",
        str(mitul_sanity_metrics),
    ]
    if mitul_available:
        _run(sanity_cmd)
        steps.append({"step": "mitul_sanity", "status": "ok", "command": sanity_cmd})
    else:
        steps.append(
            {
                "step": "mitul_sanity",
                "status": "skipped" if allow_skip_mitul else "error",
                "reason": f"Mitul corpus not found: {mitul_corpus_path}",
                "command": sanity_cmd,
            }
        )

    # B) Mitul robustness sweep
    robustness_cmd = [
        sys.executable,
        "tools/eval_mitul_robustness.py",
        "--corpus",
        args.mitul_corpus,
        "--seed",
        str(args.seed),
        "--sample-size",
        str(min(int(args.mitul_sample_size), 200000)),
        "--max-features",
        str(args.mitul_max_features),
        "--run-id",
        f"{run_id}_mitul_robustness",
    ]
    if mitul_available:
        _run(robustness_cmd)
        steps.append({"step": "mitul_robustness", "status": "ok", "command": robustness_cmd})
    else:
        steps.append(
            {
                "step": "mitul_robustness",
                "status": "skipped" if allow_skip_mitul else "error",
                "reason": f"Mitul corpus not found: {mitul_corpus_path}",
                "command": robustness_cmd,
            }
        )

    # C1) Gold-only baseline reference
    gold_ref_cmd = [
        sys.executable,
        "tools/model_comparison.py",
        "--gold-only",
        "--gold",
        args.gold,
        "--run-id",
        f"{run_id}_gold_ref",
        "--seed",
        str(args.seed),
        "--key-test-min-support",
        str(args.key_test_min_support),
        "--key-categories",
        str(args.key_categories),
    ]
    _run(gold_ref_cmd)
    steps.append({"step": "gold_reference", "status": "ok", "command": gold_ref_cmd})

    # C2) Full transfer comparison (includes warm-start if artifacts exist)
    transfer_cmd = [
        sys.executable,
        "tools/model_comparison.py",
        "--gold",
        args.gold,
        "--vocab-from",
        args.mitul_corpus,
        "--run-id",
        f"{run_id}_transfer",
        "--seed",
        str(args.seed),
        "--key-test-min-support",
        str(args.key_test_min_support),
        "--key-categories",
        str(args.key_categories),
    ]
    _run(transfer_cmd)
    steps.append({"step": "gold_transfer", "status": "ok", "command": transfer_cmd})

    # Load outputs
    mitul_sanity = _load_json(mitul_sanity_metrics) if mitul_sanity_metrics.exists() else {}
    if not mitul_robustness_json.exists():
        # eval_mitul_robustness naming convention
        mitul_robustness_json = reports_dir / f"mitul_robustness_{run_id}_mitul_robustness.json"
    mitul_robustness = _load_json(mitul_robustness_json) if mitul_robustness_json.exists() else {}
    gold_ref = _load_json(gold_ref_json)
    transfer = _load_json(transfer_json)

    transfer_base = _find_roi_row(transfer, "ml_baseline")
    transfer_warm_vocab = _find_roi_row(transfer, "ml_warm_vocab")

    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "steps": steps,
        "artifacts": {
            "mitul_sanity_metrics": str(mitul_sanity_metrics),
            "mitul_robustness_json": str(mitul_robustness_json),
            "gold_ref_json": str(gold_ref_json),
            "transfer_json": str(transfer_json),
        },
        "summary": {
            "mitul_sanity_test_macro_f1": float(mitul_sanity.get("test_macro_f1", 0.0)) if mitul_sanity else None,
            "gold_ref_ml_baseline_macro_f1": float(_find_roi_row(gold_ref, "ml_baseline").get("macro_f1", 0.0)),
            "transfer_ml_baseline_macro_f1": float(transfer_base.get("macro_f1", 0.0)),
            "transfer_ml_warm_vocab_macro_f1": float(transfer_warm_vocab.get("macro_f1", 0.0))
            if transfer_warm_vocab
            else None,
            "delta_warm_vocab_vs_baseline_macro_f1": (
                float(transfer_warm_vocab.get("macro_f1", 0.0)) - float(transfer_base.get("macro_f1", 0.0))
                if transfer_warm_vocab
                else None
            ),
        },
        "mitul_corpus_check": {
            "expected_path": str(mitul_corpus_path),
            "found": bool(mitul_available),
            "parquet_candidates": parquet_candidates[:20],
        },
        "key_test_support_policy": {
            "key_test_min_support": int(args.key_test_min_support),
            "key_categories": [x.strip() for x in str(args.key_categories).split(",") if x.strip()],
        },
    }
    consolidated_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Transfer Evaluation")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append("")
    lines.append("## Stage Status")
    lines.append("")
    lines.append("| step | status | note |")
    lines.append("|---|---|---|")
    for s in steps:
        note = s.get("reason", "")
        lines.append(f"| {s.get('step')} | {s.get('status')} | {note} |")
    lines.append("")
    lines.append("## Mitul Corpus Check")
    lines.append("")
    lines.append(f"- expected_corpus_path: `{mitul_corpus_path}`")
    lines.append(f"- found: `{'yes' if mitul_available else 'no'}`")
    if parquet_candidates:
        lines.append("- nearby data/external/*.parquet candidates:")
        for candidate in parquet_candidates[:10]:
            lines.append(f"  - `{candidate}`")
    lines.append("")
    lines.append("## A) Mitul Sanity")
    lines.append("")
    if mitul_sanity:
        lines.append(f"- test_macro_f1: `{_fmt(mitul_sanity.get('test_macro_f1', 0.0))}`")
        lines.append(f"- test_accuracy: `{_fmt(mitul_sanity.get('test_accuracy', 0.0))}`")
        lines.append(f"- test_weighted_f1: `{_fmt(mitul_sanity.get('test_weighted_f1', 0.0))}`")
    else:
        lines.append("- skipped or unavailable.")
    lines.append("")
    lines.append("## B) Mitul Robustness")
    lines.append("")
    robustness_rows = mitul_robustness.get("rows", []) if mitul_robustness else []
    if robustness_rows:
        best = max(robustness_rows, key=lambda x: float(x.get("test_macro_f1", 0.0)))
        lines.append(f"- best_variant_by_macro_f1: `{best.get('name')}` (`{_fmt(best.get('test_macro_f1', 0.0))}`)")
        lines.append("")
        lines.append("| variant | test_macro_f1 | test_accuracy |")
        lines.append("|---|---:|---:|")
        for r in robustness_rows:
            lines.append(
                f"| {r.get('name')} | {_fmt(r.get('test_macro_f1', 0.0))} | {_fmt(r.get('test_accuracy', 0.0))} |"
            )
    else:
        lines.append("- skipped or unavailable.")
    lines.append("")
    lines.append("## C) Gold Transfer (Product Metric)")
    lines.append("")
    lines.append("| model | macro_f1 | accuracy | RPM_f1 | Rent_f1 | FinancialFees_f1 | Income_f1 | Meals_f1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in transfer.get("roi_table", []):
        lines.append(
            f"| {r.get('model')} | {_fmt(r.get('macro_f1', 0.0))} | {_fmt(r.get('accuracy', 0.0))} | "
            f"{_fmt(_category_f1(r, 'Repairs & Maintenance'))} | {_fmt(_category_f1(r, 'Rent & Utilities'))} | "
            f"{_fmt(_category_f1(r, 'Financial Fees'))} | {_fmt(_category_f1(r, 'Income'))} | "
            f"{_fmt(_category_f1(r, 'Meals & Entertainment'))} |"
        )
    lines.append("")
    if transfer_warm_vocab:
        delta = float(transfer_warm_vocab.get("macro_f1", 0.0)) - float(transfer_base.get("macro_f1", 0.0))
        lines.append(f"- delta_macro_f1 (warm_vocab - baseline): `{delta:+.4f}`")
    else:
        lines.append("- warm_vocab variant not present (likely missing Mitul corpus during transfer run).")
    lines.append("")
    lines.append("## Artifact Links")
    lines.append("")
    lines.append(f"- consolidated_json: `{consolidated_json}`")
    lines.append(f"- mitul_sanity_metrics: `{mitul_sanity_metrics}`")
    lines.append(f"- mitul_robustness_json: `{mitul_robustness_json}`")
    lines.append(f"- gold_ref_json: `{gold_ref_json}`")
    lines.append(f"- transfer_json: `{transfer_json}`")
    consolidated_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {consolidated_json}")
    print(f"Wrote {consolidated_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
