#!/usr/bin/env python
"""Run Mitul preprocessing robustness sweeps on fixed deterministic splits."""

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


def _markdown(rows: list[dict], baseline_name: str) -> str:
    base = next(r for r in rows if r["name"] == baseline_name)
    lines = []
    lines.append("# Mitul Robustness Sweep")
    lines.append("")
    lines.append("| variant | test_accuracy | test_macro_f1 | test_weighted_f1 | d_macro_f1_vs_baseline |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | "
            f"{r['test_weighted_f1']:.4f} | {r['test_macro_f1'] - base['test_macro_f1']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/external/mitulshah_corpus_train.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/reports")
    metrics_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        "tools/train_mitulshah_baseline.py",
        "--corpus",
        args.corpus,
        "--seed",
        str(args.seed),
        "--sample-size",
        str(args.sample_size),
        "--max-features",
        str(args.max_features),
    ]
    variants = [
        ("lower_baseline", []),
        ("raw_text", ["--text-case", "raw"]),
        ("strip_currency_country", ["--strip-currency-tokens", "--strip-country-tokens"]),
        ("append_country_currency", ["--append-country-currency"]),
    ]

    rows = []
    for name, extra in variants:
        metrics_path = metrics_dir / f"mitulshah_baseline_metrics_{run_id}_{name}.json"
        cmd = common + ["--run-id", f"{run_id}_{name}", "--metrics-out", str(metrics_path)] + extra
        _run(cmd)
        m = _load_json(metrics_path)
        rows.append(
            {
                "name": name,
                "metrics_path": str(metrics_path),
                "test_accuracy": float(m.get("test_accuracy", 0.0)),
                "test_macro_f1": float(m.get("test_macro_f1", 0.0)),
                "test_weighted_f1": float(m.get("test_weighted_f1", 0.0)),
                "preprocessing": m.get("preprocessing", {}),
            }
        )

    report_json = out_dir / f"mitul_robustness_{run_id}.json"
    report_md = out_dir / f"mitul_robustness_{run_id}.md"
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_variant": "lower_baseline",
        "rows": rows,
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_markdown(rows, baseline_name="lower_baseline"), encoding="utf-8")
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
