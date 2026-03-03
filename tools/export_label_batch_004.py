#!/usr/bin/env python
"""Export Batch 004 using low-confidence, sparse-category, confusion, and diversity slices."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from sourcetax.taxonomy import load_sourcetax_categories, normalize_category_name


def _fetch_all(conn: sqlite3.Connection, table: str) -> List[Dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY rowid")
    return [dict(r) for r in cur.fetchall()]


def _fingerprint_from_row(r: Dict) -> str:
    merchant = str(r.get("merchant_raw") or "").strip().lower()
    description = str(r.get("description") or "").strip().lower()
    amount = str(r.get("amount") if r.get("amount") is not None else "").strip().lower()
    return f"{merchant}\t{description}\t{amount}"


def _load_gold(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_gold_fingerprints(path: Path) -> Set[str]:
    return {_fingerprint_from_row(r) for r in _load_gold(path)}


def _natural_gold_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    for r in _load_gold(path):
        source = str(r.get("source") or "").strip().lower()
        if source == "synthetic_gapfill":
            continue
        cat = normalize_category_name(r.get("sourcetax_category_v1") or r.get("category_final"))
        if cat:
            counts[cat] += 1
    return counts


def _natural_gold_merchant_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    for r in _load_gold(path):
        source = str(r.get("source") or "").strip().lower()
        if source == "synthetic_gapfill":
            continue
        m = str(r.get("merchant_norm") or r.get("merchant_raw") or "").strip().lower()
        if m:
            counts[m] += 1
    return counts


def _parse_confusions(path: Path, top_n: int = 3) -> List[Tuple[str, str, int]]:
    if not path.exists():
        return []
    pairs: List[Tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gold = normalize_category_name(row.get("gold_label"))
            pred = normalize_category_name(row.get("predicted_label"))
            if not gold or not pred:
                continue
            if gold == pred:
                continue
            if gold == "Other Expense" or pred == "Other Expense":
                continue
            try:
                c = int(float(row.get("count") or 0))
            except Exception:
                c = 0
            pairs.append((gold, pred, c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def _to_output_row(r: Dict, batch_tag: str, target_category: str = "", confusion_pair: str = "") -> Dict:
    return {
        "rowid": r.get("rowid"),
        "batch_tag": batch_tag,
        "target_category": target_category,
        "confusion_pair": confusion_pair,
        "source": r.get("source"),
        "source_record_id": r.get("source_record_id"),
        "transaction_date": r.get("txn_ts"),
        "merchant_raw": r.get("merchant_raw"),
        "merchant_norm": r.get("merchant_norm"),
        "description": r.get("description"),
        "amount": r.get("amount"),
        "currency": r.get("currency"),
        "mcc": r.get("mcc"),
        "mcc_description": r.get("mcc_description"),
        "category_external": r.get("category_external"),
        "rule_category_suggested": r.get("rule_category"),
        "rule_confidence": r.get("rule_confidence"),
        "mapping_reason": r.get("mapping_reason"),
        "missing_mcc": r.get("missing_mcc"),
        "merchant_only": r.get("merchant_only"),
        "generic_description": r.get("generic_description"),
        "sourcetax_category_v1": "",
        "label_confidence": "",
        "label_notes": "",
    }


def _pick_low_conf(
    rows: List[Dict],
    selected_rowids: Set[int],
    seen_fps: Set[str],
    *,
    target: int,
    conf_max: float,
    dominant_merchants: Set[str],
) -> List[Dict]:
    def score(r: Dict) -> float:
        conf = float(r.get("rule_confidence") or 0.0)
        s = (1.0 - conf) * 4.0
        s += 1.5 if int(r.get("missing_mcc") or 0) else 0.0
        s += 1.2 if int(r.get("merchant_only") or 0) else 0.0
        s += 1.0 if int(r.get("generic_description") or 0) else 0.0
        reason = str(r.get("mapping_reason") or "")
        if "fallback:Other Expense" in reason:
            s += 1.2
        m = str(r.get("merchant_norm") or r.get("merchant_raw") or "").strip().lower()
        if m and m not in dominant_merchants:
            s += 0.6
        return s

    cands = [r for r in rows if float(r.get("rule_confidence") or 0.0) < conf_max]
    cands.sort(key=score, reverse=True)
    out: List[Dict] = []
    for r in cands:
        rid = int(r.get("rowid") or 0)
        if rid in selected_rowids:
            continue
        fp = _fingerprint_from_row(r)
        if fp in seen_fps:
            continue
        selected_rowids.add(rid)
        seen_fps.add(fp)
        out.append(r)
        if len(out) >= target:
            break
    return out


def _pick_sparse_category(
    rows: List[Dict],
    selected_rowids: Set[int],
    seen_fps: Set[str],
    *,
    sparse_categories: List[str],
    per_category: int,
) -> Tuple[List[Tuple[Dict, str]], Dict[str, int]]:
    out: List[Tuple[Dict, str]] = []
    shortages: Dict[str, int] = {}
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        cat = normalize_category_name(r.get("rule_category"))
        if cat:
            by_cat[cat].append(r)
    for cat in sparse_categories:
        picked = 0
        cands = sorted(by_cat.get(cat, []), key=lambda x: float(x.get("rule_confidence") or 0.0))
        for r in cands:
            rid = int(r.get("rowid") or 0)
            if rid in selected_rowids:
                continue
            fp = _fingerprint_from_row(r)
            if fp in seen_fps:
                continue
            selected_rowids.add(rid)
            seen_fps.add(fp)
            out.append((r, cat))
            picked += 1
            if picked >= per_category:
                break
        if picked < per_category:
            shortages[cat] = per_category - picked
    return out, shortages


def _pick_confusions(
    rows: List[Dict],
    selected_rowids: Set[int],
    seen_fps: Set[str],
    *,
    pairs: List[Tuple[str, str, int]],
    per_pair: int,
    conf_min: float,
    conf_max: float,
) -> Tuple[List[Tuple[Dict, str]], Dict[str, int]]:
    out: List[Tuple[Dict, str]] = []
    shortages: Dict[str, int] = {}
    for a, b, _count in pairs:
        pair_key = f"{a}<->{b}"
        picked = 0
        cands = [
            r
            for r in rows
            if normalize_category_name(r.get("rule_category")) in {a, b}
            and conf_min <= float(r.get("rule_confidence") or 0.0) <= conf_max
        ]
        cands.sort(key=lambda x: float(x.get("rule_confidence") or 0.0))
        for r in cands:
            rid = int(r.get("rowid") or 0)
            if rid in selected_rowids:
                continue
            fp = _fingerprint_from_row(r)
            if fp in seen_fps:
                continue
            selected_rowids.add(rid)
            seen_fps.add(fp)
            out.append((r, pair_key))
            picked += 1
            if picked >= per_pair:
                break
        if picked < per_pair:
            shortages[pair_key] = per_pair - picked
    return out, shortages


def _pick_long_tail(
    rows: List[Dict],
    selected_rowids: Set[int],
    seen_fps: Set[str],
    *,
    target: int,
    dominant_merchants: Set[str],
) -> List[Dict]:
    cands = []
    for r in rows:
        m = str(r.get("merchant_norm") or r.get("merchant_raw") or "").strip().lower()
        if not m:
            continue
        if m in dominant_merchants:
            continue
        cands.append(r)
    cands.sort(key=lambda x: float(x.get("rule_confidence") or 0.0))
    out: List[Dict] = []
    for r in cands:
        rid = int(r.get("rowid") or 0)
        if rid in selected_rowids:
            continue
        fp = _fingerprint_from_row(r)
        if fp in seen_fps:
            continue
        selected_rowids.add(rid)
        seen_fps.add(fp)
        out.append(r)
        if len(out) >= target:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--table", default="staging_transactions_enriched")
    parser.add_argument("--gold-path", default="data/gold/gold_transactions.jsonl")
    parser.add_argument("--confusions-csv", default="outputs/reports/eval_top_confusions.csv")
    parser.add_argument("--out", default="outputs/label_batch_004.csv")
    parser.add_argument("--summary-out", default="outputs/reports/label_batch_004_summary.json")
    parser.add_argument("--total", type=int, default=180)
    parser.add_argument("--low-conf-target", type=int, default=90)
    parser.add_argument("--low-conf-max", type=float, default=0.75)
    parser.add_argument("--sparse-min-natural", type=int, default=15)
    parser.add_argument("--sparse-target-total", type=int, default=45)
    parser.add_argument("--sparse-max-categories", type=int, default=3)
    parser.add_argument("--per-sparse-category", type=int, default=20)
    parser.add_argument("--confusion-top-n", type=int, default=3)
    parser.add_argument("--confusion-target-total", type=int, default=30)
    parser.add_argument("--per-confusion-pair", type=int, default=15)
    parser.add_argument("--confusion-min-conf", type=float, default=0.60)
    parser.add_argument("--confusion-max-conf", type=float, default=0.85)
    parser.add_argument("--diversity-target", type=int, default=15)
    parser.add_argument("--dominant-merchant-min-gold", type=int, default=3)
    args = parser.parse_args()

    db = Path(args.staging_db)
    if not db.exists():
        print(f"Missing staging DB: {db}")
        return 1
    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_all(conn, args.table)
    except sqlite3.OperationalError:
        print(f"Missing table {args.table}. Run tools/build_staging_enriched.py first.")
        conn.close()
        return 1
    conn.close()

    gold_path = Path(args.gold_path)
    seen_fps: Set[str] = _load_gold_fingerprints(gold_path)
    natural_counts = _natural_gold_counts(gold_path)
    gold_merchants = _natural_gold_merchant_counts(gold_path)
    dominant_merchants = {m for m, n in gold_merchants.items() if n >= args.dominant_merchant_min_gold}

    taxonomy_categories = load_sourcetax_categories(include_uncategorized=False)
    sparse_categories = [c for c in taxonomy_categories if natural_counts.get(c, 0) < args.sparse_min_natural]
    sparse_categories.sort(key=lambda c: (natural_counts.get(c, 0), c))
    sparse_categories = sparse_categories[: max(1, int(args.sparse_max_categories))]

    conf_pairs = _parse_confusions(Path(args.confusions_csv), top_n=args.confusion_top_n)

    selected_rowids: Set[int] = set()
    output_rows: List[Dict] = []

    low_conf: List[Dict] = []
    if int(args.low_conf_target) > 0:
        low_conf = _pick_low_conf(
            rows,
            selected_rowids,
            seen_fps,
            target=args.low_conf_target,
            conf_max=args.low_conf_max,
            dominant_merchants=dominant_merchants,
        )
    output_rows.extend(_to_output_row(r, "B004_low_conf") for r in low_conf)

    sparse_rows: List[Tuple[Dict, str]] = []
    sparse_shortages: Dict[str, int] = {}
    if int(args.sparse_target_total) > 0 and sparse_categories:
        sparse_per_category = max(1, int(args.sparse_target_total) // max(1, len(sparse_categories)))
        sparse_rows, sparse_shortages = _pick_sparse_category(
            rows,
            selected_rowids,
            seen_fps,
            sparse_categories=sparse_categories,
            per_category=min(args.per_sparse_category, sparse_per_category),
        )
        sparse_rows = sparse_rows[: int(args.sparse_target_total)]
    output_rows.extend(_to_output_row(r, "B004_sparse_category", target_category=cat) for r, cat in sparse_rows)

    conf_rows: List[Tuple[Dict, str]] = []
    conf_shortages: Dict[str, int] = {}
    if int(args.confusion_target_total) > 0 and conf_pairs:
        conf_rows, conf_shortages = _pick_confusions(
            rows,
            selected_rowids,
            seen_fps,
            pairs=conf_pairs,
            per_pair=args.per_confusion_pair,
            conf_min=args.confusion_min_conf,
            conf_max=args.confusion_max_conf,
        )
        conf_rows = conf_rows[: int(args.confusion_target_total)]
    output_rows.extend(_to_output_row(r, "B004_confusion_pair", confusion_pair=pair) for r, pair in conf_rows)

    diversity_rows: List[Dict] = []
    if int(args.diversity_target) > 0:
        diversity_rows = _pick_long_tail(
            rows,
            selected_rowids,
            seen_fps,
            target=args.diversity_target,
            dominant_merchants=dominant_merchants,
        )
    output_rows.extend(_to_output_row(r, "B004_merchant_diversity") for r in diversity_rows)

    if len(output_rows) < args.total:
        for r in rows:
            rid = int(r.get("rowid") or 0)
            if rid in selected_rowids:
                continue
            fp = _fingerprint_from_row(r)
            if fp in seen_fps:
                continue
            selected_rowids.add(rid)
            seen_fps.add(fp)
            output_rows.append(_to_output_row(r, "B004_backfill"))
            if len(output_rows) >= args.total:
                break

    output_rows = output_rows[: args.total]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(output_rows)

    summary = {
        "out": str(out),
        "total_rows": len(output_rows),
        "staging_rows": len(rows),
        "gold_fingerprint_exclusions": len(_load_gold_fingerprints(gold_path)),
        "low_conf_target": args.low_conf_target,
        "low_conf_selected": len(low_conf),
        "sparse_categories_min_natural": args.sparse_min_natural,
        "sparse_categories": sparse_categories,
        "sparse_selected": len(sparse_rows),
        "sparse_shortages": sparse_shortages,
        "confusion_pairs": [{"a": a, "b": b, "count": c} for a, b, c in conf_pairs],
        "confusion_selected": len(conf_rows),
        "confusion_shortages": conf_shortages,
        "diversity_selected": len(diversity_rows),
        "dominant_merchants_threshold": args.dominant_merchant_min_gold,
        "dominant_merchants_count": len(dominant_merchants),
        "batch_tag_counts": dict(Counter(r["batch_tag"] for r in output_rows)),
    }
    s_out = Path(args.summary_out)
    s_out.parent.mkdir(parents=True, exist_ok=True)
    s_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Batch 004 export complete.")
    print(f"- out: {out}")
    print(f"- total_rows: {len(output_rows)}")
    print(f"- low_conf_selected: {len(low_conf)}")
    print(f"- sparse_selected: {len(sparse_rows)}")
    print(f"- confusion_selected: {len(conf_rows)}")
    print(f"- diversity_selected: {len(diversity_rows)}")
    print(f"- summary: {s_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
