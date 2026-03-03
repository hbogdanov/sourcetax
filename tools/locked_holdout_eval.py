#!/usr/bin/env python
"""Train an enriched ML artifact on a locked train split and evaluate on locked holdout only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sourcetax import mapping, shadow_ml
from sourcetax.gold import filter_human_labeled_gold
from sourcetax.taxonomy import normalize_category_name


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default
    return default


def _read_gold(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def _record_key(rec: Dict[str, Any], idx: int) -> str:
    rid = str(rec.get("id") or "").strip()
    if rid:
        return f"id:{rid}"
    source = str(rec.get("source") or "").strip().lower()
    source_id = str(rec.get("source_record_id") or "").strip()
    if source and source_id:
        return f"src:{source}:{source_id}"
    merchant = str(rec.get("merchant_raw") or "").strip().lower()
    amount = str(rec.get("amount") if rec.get("amount") is not None else "")
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    date = str((raw or {}).get("date") or rec.get("date") or "").strip()
    fp = f"{source}|{merchant}|{amount}|{date}"
    return f"fp:{fp}" if fp.strip("|") else f"idx:{idx}"


def _description(rec: Dict[str, Any]) -> str:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return str(rec.get("description") or raw.get("description") or raw.get("ocr_text") or "").strip()


def _mcc(rec: Dict[str, Any]) -> str:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return str(rec.get("mcc") or raw.get("mcc") or "").strip()


def _mcc_description(rec: Dict[str, Any]) -> str:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return str(rec.get("mcc_description") or raw.get("mcc_description") or "").strip()


def _external_category(rec: Dict[str, Any]) -> str:
    raw = _safe_json_loads(rec.get("raw_payload"), {})
    if not isinstance(raw, dict):
        raw = {}
    return str(rec.get("category_external") or raw.get("category_external") or "").strip()


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


def _prepare(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(rows):
        label = normalize_category_name(rec.get("sourcetax_category_v1") or rec.get("category_final"))
        if not label:
            continue
        merchant_raw = str(rec.get("merchant_raw") or "").strip()
        desc = _description(rec)
        mcc = _mcc(rec)
        mcc_desc = _mcc_description(rec)
        ext = _external_category(rec)
        text, reasons = shadow_ml.build_enriched_text(
            merchant_raw=merchant_raw,
            description=desc,
            mcc=mcc,
            mcc_description=mcc_desc,
            category_external=ext,
            amount=rec.get("amount"),
        )
        if not text:
            continue
        rr = dict(rec)
        rr["_key"] = _record_key(rec, i)
        rr["_label"] = label
        rr["_text"] = text
        rr["_reason"] = reasons
        rr["_description"] = desc
        rr["_mcc"] = mcc
        rr["_mcc_description"] = mcc_desc
        rr["_category_external"] = ext
        out.append(rr)
    return out


def _create_or_load_split(
    rows: List[Dict[str, Any]],
    split_path: Path,
    *,
    holdout_size: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_key = {r["_key"]: r for r in rows}
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_keys = set(payload.get("train_keys") or [])
        holdout_keys = set(payload.get("holdout_keys") or [])
        train = [by_key[k] for k in train_keys if k in by_key]
        holdout = [by_key[k] for k in holdout_keys if k in by_key]
        if not train or not holdout:
            raise ValueError("Existing locked split does not match current gold rows.")
        return train, holdout, payload

    labels = [r["_label"] for r in rows]
    counts = Counter(labels)
    can_stratify = len(counts) > 1 and min(counts.values()) >= 2
    train, holdout = train_test_split(
        rows,
        test_size=holdout_size,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    payload = {
        "version": 1,
        "seed": seed,
        "holdout_size": holdout_size,
        "gold_rows_total": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_keys": sorted(r["_key"] for r in train),
        "holdout_keys": sorted(r["_key"] for r in holdout),
        "taxonomy_hash": shadow_ml.taxonomy_hash(),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return train, holdout, payload


def _rule_prediction(rec: Dict[str, Any]) -> Tuple[str, float]:
    pred, reasons = mapping.resolve_category_with_reason(
        merchant_raw=str(rec.get("merchant_raw") or "").strip() or None,
        description=rec.get("_description") or None,
        mcc=rec.get("_mcc") or None,
        mcc_description=rec.get("_mcc_description") or None,
        external_category=rec.get("_category_external") or None,
        amount=rec.get("amount"),
        fallback="Other Expense",
    )
    pred_n = normalize_category_name(pred) or "Other Expense"
    conf = 0.3
    if reasons:
        first = str(reasons[0])
        if first.startswith("financial_high:") or "_high:" in first or first.startswith("keyword:"):
            conf = 0.9
        elif first.startswith("financial_medium:") or "_medium:" in first:
            conf = 0.7
        elif first.startswith("mcc:") or first.startswith("mcc_description:"):
            conf = 0.85
        elif first.startswith("external:"):
            conf = 0.7
    return pred_n, conf


def _accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    good = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return good / len(y_true)


def _rule_bin(conf: float) -> str:
    if conf >= 0.90:
        return ">=0.90"
    if conf >= 0.80:
        return "0.80-0.89"
    if conf >= 0.70:
        return "0.70-0.79"
    return "<0.70"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--split-path", default="artifacts/holdout_locked/split_manifest.json")
    parser.add_argument("--artifact-path", default="artifacts/holdout_locked/ml_enriched_tfidf_pipeline.joblib")
    parser.add_argument("--metadata-path", default="artifacts/holdout_locked/ml_enriched_metadata.json")
    parser.add_argument("--report-dir", default="outputs/reports/holdout_locked")
    parser.add_argument("--holdout-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--threshold-alt", type=float, default=0.70)
    args = parser.parse_args()

    rows = _prepare(_read_gold(Path(args.gold)))
    if len(rows) < 50:
        print(f"Not enough usable gold rows: {len(rows)}")
        return 1

    train, holdout, split_payload = _create_or_load_split(
        rows,
        Path(args.split_path),
        holdout_size=float(args.holdout_size),
        seed=int(args.seed),
    )
    if not train or not holdout:
        print("Split failed: empty train or holdout.")
        return 1

    model = _build_pipeline()
    model.fit([r["_text"] for r in train], [r["_label"] for r in train])

    artifact_path = Path(args.artifact_path)
    metadata_path = Path(args.metadata_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    class_counts = Counter(r["_label"] for r in train)
    synthetic_rows = sum(1 for r in train if str(r.get("source") or "").strip().lower() == "synthetic_gapfill")
    meta = shadow_ml.artifact_metadata(
        train_rows=len(train),
        natural_rows=len(train) - synthetic_rows,
        synthetic_rows=synthetic_rows,
        class_counts=dict(class_counts),
    )
    meta["locked_holdout"] = {
        "split_path": str(Path(args.split_path)),
        "holdout_size": float(args.holdout_size),
        "seed": int(args.seed),
        "holdout_rows": len(holdout),
    }
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    y_true = [r["_label"] for r in holdout]
    y_ml = [normalize_category_name(p) or "Other Expense" for p in model.predict([r["_text"] for r in holdout])]

    ml_conf = []
    if hasattr(model, "predict_proba"):
        for probs in model.predict_proba([r["_text"] for r in holdout]):
            ml_conf.append(float(max(probs)) if len(probs) else 0.0)
    else:
        ml_conf = [0.0] * len(holdout)

    rule_preds: List[str] = []
    rule_conf: List[float] = []
    for rec in holdout:
        rp, rc = _rule_prediction(rec)
        rule_preds.append(rp)
        rule_conf.append(float(rc))

    y_hybrid = []
    y_hybrid_alt = []
    for rp, rc, mp in zip(rule_preds, rule_conf, y_ml):
        y_hybrid.append(rp if rc >= float(args.threshold) else mp)
        y_hybrid_alt.append(rp if rc >= float(args.threshold_alt) else mp)

    overall = {
        "n_holdout": len(holdout),
        "rules_accuracy": _accuracy(y_true, rule_preds),
        "ml_accuracy": _accuracy(y_true, y_ml),
        f"hybrid_t{int(args.threshold * 100)}_accuracy": _accuracy(y_true, y_hybrid),
        f"hybrid_t{int(args.threshold_alt * 100)}_accuracy": _accuracy(y_true, y_hybrid_alt),
    }

    bins = [">=0.90", "0.80-0.89", "0.70-0.79", "<0.70"]
    by_bin: Dict[str, Dict[str, Any]] = {}
    for b in bins:
        idx = [i for i, c in enumerate(rule_conf) if _rule_bin(c) == b]
        if not idx:
            by_bin[b] = {"n": 0, "rules_accuracy": 0.0, "ml_accuracy": 0.0, "hybrid_accuracy": 0.0}
            continue
        yt = [y_true[i] for i in idx]
        yr = [rule_preds[i] for i in idx]
        ym = [y_ml[i] for i in idx]
        yh = [y_hybrid[i] for i in idx]
        by_bin[b] = {
            "n": len(idx),
            "rules_accuracy": _accuracy(yt, yr),
            "ml_accuracy": _accuracy(yt, ym),
            "hybrid_accuracy": _accuracy(yt, yh),
        }

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "locked_holdout_summary.json"
    rows_path = out_dir / "locked_holdout_row_level.csv"
    bins_path = out_dir / "locked_holdout_by_conf_bin.csv"

    summary = {
        "split": {
            "path": str(Path(args.split_path)),
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "taxonomy_hash": split_payload.get("taxonomy_hash"),
        },
        "artifact": str(artifact_path),
        "metadata": str(metadata_path),
        "overall": overall,
        "by_conf_bin": by_bin,
        "threshold": float(args.threshold),
        "threshold_alt": float(args.threshold_alt),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "key",
                "id",
                "source",
                "merchant_raw",
                "description",
                "mcc_description",
                "amount",
                "gold_label",
                "rule_category",
                "rule_confidence",
                "ml_prediction",
                "ml_confidence",
                "hybrid_prediction",
                "hybrid_prediction_alt",
            ],
        )
        writer.writeheader()
        for rec, truth, rp, rc, mp, mc, hp, ha in zip(holdout, y_true, rule_preds, rule_conf, y_ml, ml_conf, y_hybrid, y_hybrid_alt):
            writer.writerow(
                {
                    "key": rec.get("_key"),
                    "id": rec.get("id"),
                    "source": rec.get("source"),
                    "merchant_raw": rec.get("merchant_raw"),
                    "description": rec.get("_description"),
                    "mcc_description": rec.get("_mcc_description"),
                    "amount": rec.get("amount"),
                    "gold_label": truth,
                    "rule_category": rp,
                    "rule_confidence": rc,
                    "ml_prediction": mp,
                    "ml_confidence": mc,
                    "hybrid_prediction": hp,
                    "hybrid_prediction_alt": ha,
                }
            )

    with bins_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bin", "n", "rules_accuracy", "ml_accuracy", "hybrid_accuracy"])
        writer.writeheader()
        for b in bins:
            row = dict(by_bin[b])
            row["bin"] = b
            writer.writerow(row)

    print("Locked holdout train/eval complete.")
    print(f"- split_manifest: {args.split_path}")
    print(f"- artifact: {artifact_path}")
    print(f"- metadata: {metadata_path}")
    print(f"- holdout_n: {overall['n_holdout']}")
    print(f"- rules_accuracy: {overall['rules_accuracy']:.3f}")
    print(f"- ml_accuracy: {overall['ml_accuracy']:.3f}")
    print(f"- hybrid_t{int(args.threshold * 100)}_accuracy: {overall[f'hybrid_t{int(args.threshold * 100)}_accuracy']:.3f}")
    print(f"- summary: {summary_path}")
    print(f"- by_conf_bin: {bins_path}")
    print(f"- row_level: {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

