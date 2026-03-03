#!/usr/bin/env python
"""Train and save canonical enriched TF-IDF shadow model artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sourcetax.gold import filter_human_labeled_gold
from sourcetax.shadow_ml import (
    ARTIFACT_DIR,
    METADATA_PATH,
    PIPELINE_PATH,
    artifact_metadata,
    build_enriched_text,
)
from sourcetax.taxonomy import normalize_category_name


def _load_gold(path: Path) -> List[Dict]:
    rows: List[Dict] = []
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


def _is_synthetic(rec: Dict) -> bool:
    return str(rec.get("source") or "").strip().lower() == "synthetic_gapfill"


def _prepare(rows: List[Dict], include_synthetic: bool) -> tuple[list[str], list[str], int, int, Dict[str, int]]:
    X: List[str] = []
    y: List[str] = []
    natural_n = 0
    synthetic_n = 0
    class_counts = Counter()
    for r in rows:
        if (not include_synthetic) and _is_synthetic(r):
            continue
        label = normalize_category_name(r.get("sourcetax_category_v1") or r.get("category_final"))
        if not label:
            continue
        text, _ = build_enriched_text(
            merchant_raw=str(r.get("merchant_raw") or "").strip(),
            description=str((r.get("description") or (r.get("raw_payload") or {}).get("description") or (r.get("raw_payload") or {}).get("ocr_text") or "")).strip(),
            mcc=str(r.get("mcc") or (r.get("raw_payload") or {}).get("mcc") or "").strip(),
            mcc_description=str(r.get("mcc_description") or (r.get("raw_payload") or {}).get("mcc_description") or "").strip(),
            category_external=str(r.get("category_external") or (r.get("raw_payload") or {}).get("category_external") or "").strip(),
            amount=r.get("amount"),
        )
        if not text:
            continue
        X.append(text)
        y.append(label)
        class_counts[label] += 1
        if _is_synthetic(r):
            synthetic_n += 1
        else:
            natural_n += 1
    return X, y, natural_n, synthetic_n, dict(class_counts)


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
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        default=True,
        help="Include synthetic_gapfill rows in training (default: true).",
    )
    parser.add_argument(
        "--natural-only",
        action="store_false",
        dest="include_synthetic",
        help="Train only on non-synthetic gold rows.",
    )
    args = parser.parse_args()

    rows = _load_gold(Path(args.gold))
    if not rows:
        print(f"No gold rows found at {args.gold}")
        return 1

    X, y, natural_n, synthetic_n, class_counts = _prepare(rows, include_synthetic=bool(args.include_synthetic))
    if len(X) < 50:
        print(f"Not enough rows to train robustly: {len(X)}")
        return 1

    pipe = _build_pipeline()
    pipe.fit(X, y)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore

    joblib.dump(pipe, PIPELINE_PATH)
    meta = artifact_metadata(
        train_rows=len(X),
        natural_rows=natural_n,
        synthetic_rows=synthetic_n,
        class_counts=class_counts,
    )
    meta["include_synthetic"] = bool(args.include_synthetic)
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    classes = list(pipe.named_steps["clf"].classes_)
    vocab = len(pipe.named_steps["tfidf"].vocabulary_)
    print("Enriched shadow model training complete.")
    print(f"- artifact: {PIPELINE_PATH}")
    print(f"- metadata: {METADATA_PATH}")
    print(f"- train_rows: {len(X)} (natural={natural_n}, synthetic={synthetic_n})")
    print(f"- classes: {len(classes)}")
    print(f"- vocab_size: {vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

