#!/usr/bin/env python
"""Export a balanced gold-label candidate batch from enriched staging rows."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_ANCHOR_CATEGORIES = [
    "Rent & Utilities",
    "Equipment & Software",
    "Payroll & Contractors",
    "Taxes & Licenses",
    "Insurance",
]


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


def _load_gold_fingerprints(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            out.add(_fingerprint_from_row(row))
    return out


def _pick_batch_a(rows: List[Dict], selected: Set[int], target: int, anchor_categories: List[str]) -> List[Dict]:
    per_cat = max(10, target // max(len(anchor_categories), 1))
    out: List[Dict] = []
    for cat in anchor_categories:
        candidates = [
            r
            for r in rows
            if r["rowid"] not in selected
            and (r.get("rule_category") or "") == cat
            and float(r.get("rule_confidence") or 0) >= 0.85
        ]
        for r in candidates[:per_cat]:
            out.append(r)
            selected.add(r["rowid"])
            if len(out) >= target:
                return out
    if len(out) < target:
        fallback = [
            r for r in rows if r["rowid"] not in selected and float(r.get("rule_confidence") or 0) >= 0.85
        ]
        for r in fallback[: target - len(out)]:
            out.append(r)
            selected.add(r["rowid"])
    return out


def _pick_batch_b(rows: List[Dict], selected: Set[int], target: int) -> List[Dict]:
    def hard_score(r: Dict) -> float:
        score = 0.0
        conf = float(r.get("rule_confidence") or 0.0)
        score += (1.0 - conf) * 3.0
        score += 1.5 if int(r.get("missing_mcc") or 0) else 0.0
        score += 1.2 if int(r.get("merchant_only") or 0) else 0.0
        score += 1.2 if int(r.get("generic_description") or 0) else 0.0
        reason = str(r.get("mapping_reason") or "")
        if "fallback:Other Expense" in reason:
            score += 2.0
        return score

    candidates = [r for r in rows if r["rowid"] not in selected]
    candidates.sort(key=hard_score, reverse=True)
    out = candidates[:target]
    for r in out:
        selected.add(r["rowid"])
    return out


def _pick_batch_c(rows: List[Dict], selected: Set[int], target: int) -> List[Dict]:
    counts: Dict[str, int] = {}
    for r in rows:
        cat = str(r.get("rule_category") or "Other Expense")
        counts[cat] = counts.get(cat, 0) + 1
    rare_first = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    categories = [c for c, _ in rare_first]

    out: List[Dict] = []
    idx = 0
    while len(out) < target and categories:
        cat = categories[idx % len(categories)]
        idx += 1
        picked = None
        for r in rows:
            if r["rowid"] in selected:
                continue
            if (r.get("rule_category") or "") == cat:
                picked = r
                break
        if picked is None:
            categories = [c for c in categories if c != cat]
            continue
        out.append(picked)
        selected.add(picked["rowid"])

    if len(out) < target:
        fallback = [r for r in rows if r["rowid"] not in selected]
        for r in fallback[: target - len(out)]:
            out.append(r)
            selected.add(r["rowid"])
    return out


def _to_output_row(r: Dict, batch_tag: str) -> Dict:
    return {
        "rowid": r.get("rowid"),
        "batch_tag": batch_tag,
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


def _dedupe_rows(
    labeled_rows: List[Tuple[Dict, str]],
    *,
    seen_fingerprints: Set[str],
) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    for row, tag in labeled_rows:
        fp = _fingerprint_from_row(row)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        out.append((row, tag))
    return out


def _append_backfill(
    *,
    rows: List[Dict],
    selected_rowids: Set[int],
    seen_fingerprints: Set[str],
    needed: int,
) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    if needed <= 0:
        return out
    for r in rows:
        rid = int(r.get("rowid") or 0)
        if rid in selected_rowids:
            continue
        fp = _fingerprint_from_row(r)
        if fp in seen_fingerprints:
            continue
        selected_rowids.add(rid)
        seen_fingerprints.add(fp)
        out.append((r, "Z_backfill"))
        if len(out) >= needed:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--table", default="staging_transactions_enriched")
    parser.add_argument("--out", default="artifacts/labeling/label_batch_001.csv")
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--batch-a", type=int, default=60)
    parser.add_argument("--batch-b", type=int, default=80)
    parser.add_argument("--batch-c", type=int, default=60)
    parser.add_argument(
        "--anchor-categories",
        default=",".join(DEFAULT_ANCHOR_CATEGORIES),
        help="Comma-separated categories for easy anchors.",
    )
    parser.add_argument(
        "--gold-path",
        default="data/gold/gold_transactions.jsonl",
        help="Existing gold file used for optional de-dupe.",
    )
    parser.add_argument(
        "--exclude-gold-fingerprints",
        action="store_true",
        default=True,
        help="Skip rows already present in gold by merchant+description+amount fingerprint.",
    )
    parser.add_argument(
        "--allow-gold-fingerprints",
        action="store_false",
        dest="exclude_gold_fingerprints",
        help="Allow rows that duplicate existing gold fingerprints.",
    )
    args = parser.parse_args()

    db = Path(args.staging_db)
    if not db.exists():
        print(f"Missing staging DB: {db}")
        return 1
    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_all(conn, args.table)
    except sqlite3.OperationalError:
        print(
            f"Missing table {args.table}. Run tools/build_staging_enriched.py first."
        )
        conn.close()
        return 1
    conn.close()

    selected: Set[int] = set()
    anchors = [c.strip() for c in args.anchor_categories.split(",") if c.strip()]
    batch_a = _pick_batch_a(rows, selected, args.batch_a, anchors)
    batch_b = _pick_batch_b(rows, selected, args.batch_b)
    batch_c = _pick_batch_c(rows, selected, args.batch_c)

    initial_labeled_rows: List[Tuple[Dict, str]] = (
        [(r, "A_easy_anchor") for r in batch_a]
        + [(r, "B_hard_ambiguous") for r in batch_b]
        + [(r, "C_underrepresented") for r in batch_c]
    )
    selected_rowids: Set[int] = set(int(r.get("rowid") or 0) for r, _ in initial_labeled_rows)

    seen_fps: Set[str] = set()
    if args.exclude_gold_fingerprints:
        seen_fps |= _load_gold_fingerprints(Path(args.gold_path))
    deduped = _dedupe_rows(initial_labeled_rows, seen_fingerprints=seen_fps)

    if len(deduped) < args.total:
        deduped.extend(
            _append_backfill(
                rows=rows,
                selected_rowids=selected_rowids,
                seen_fingerprints=seen_fps,
                needed=args.total - len(deduped),
            )
        )

    merged = [_to_output_row(r, tag) for r, tag in deduped[: args.total]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(merged[0].keys()) if merged else []
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(merged)

    print("Label batch export complete.")
    print(f"- out: {out}")
    print(f"- total_rows: {len(merged)}")
    print(f"- batch_a: {len(batch_a)}")
    print(f"- batch_b: {len(batch_b)}")
    print(f"- batch_c: {len(batch_c)}")
    if args.exclude_gold_fingerprints:
        print(f"- excluded_existing_gold_fingerprints: true ({args.gold_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


