#!/usr/bin/env python
"""Generate realistic receiptâ†”bank matched pairs + non-matches for eval realism.

Writes:
1) synthetic receipt and bank rows into staging DB
2) matching mini gold set JSONL (default 50 positive pairs + negatives)
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sourcetax import staging
from sourcetax.normalization import generate_noisy_merchant_raw


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return default
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except Exception:
            continue
    return None


def _sample_offset_days(rng: random.Random) -> int:
    """Offset distribution: {0,1,2} with probs {0.55,0.35,0.10}."""
    x = rng.random()
    if x < 0.55:
        return 0
    if x < 0.90:
        return 1
    return 2


def _sample_tip_rate(rng: random.Random) -> float:
    return rng.uniform(0.0, 0.25)


def _sample_rounding_noise(rng: random.Random) -> float:
    return rng.uniform(-0.5, 0.5)


def _load_receipt_candidates(staging_db: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(staging_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source, source_record_id, receipt_ts, merchant_raw, total, tax, currency, ocr_text,
               structured_fields_json, raw_payload_json
        FROM staging_receipts
        WHERE merchant_raw IS NOT NULL
          AND TRIM(merchant_raw) <> ''
          AND total IS NOT NULL
          AND total > 0
        ORDER BY rowid
        """
    )
    rows = []
    for r in cur.fetchall():
        rows.append(
            {
                "source": r["source"],
                "source_record_id": r["source_record_id"],
                "receipt_ts": r["receipt_ts"],
                "merchant_raw": r["merchant_raw"],
                "total": float(r["total"]),
                "tax": r["tax"],
                "currency": r["currency"] or "USD",
                "ocr_text": r["ocr_text"],
                "structured_fields_json": _safe_json_loads(r["structured_fields_json"], {}),
                "raw_payload_json": _safe_json_loads(r["raw_payload_json"], {}),
            }
        )
    conn.close()
    return rows


def _make_positive_pair(
    *,
    base_receipt: Dict[str, Any],
    pair_idx: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    group_id = f"pair_{pair_idx:05d}"
    base_dt = _parse_iso_date(base_receipt.get("receipt_ts")) or date.today()
    offset_days = _sample_offset_days(rng)
    txn_date = base_dt + timedelta(days=offset_days)

    total = abs(float(base_receipt["total"]))
    tip_amt = total * _sample_tip_rate(rng) if rng.random() < 0.35 else 0.0
    rounding = _sample_rounding_noise(rng)
    bank_amount = -(total + tip_amt + rounding)

    canonical_merchant = str(base_receipt["merchant_raw"])
    noisy = generate_noisy_merchant_raw(canonical_merchant, n=1, seed=rng.randint(1, 10**9))[0]

    receipt_id = f"{group_id}_r"
    bank_id = f"{group_id}_b"

    receipt_row = {
        "source": "synthetic_pair_receipt",
        "source_record_id": receipt_id,
        "receipt_ts": base_dt.isoformat(),
        "merchant_raw": canonical_merchant,
        "total": total,
        "tax": base_receipt.get("tax"),
        "currency": base_receipt.get("currency") or "USD",
        "ocr_text": base_receipt.get("ocr_text"),
        "structured_fields_json": {
            "group_id": group_id,
            "expected_match": True,
            "parent_receipt_source": base_receipt.get("source"),
            "parent_receipt_id": base_receipt.get("source_record_id"),
            **(_safe_json_loads(base_receipt.get("structured_fields_json"), {})),
        },
        "raw_payload_json": {
            "group_id": group_id,
            "pair_role": "receipt",
            "expected_match": True,
            "generator": "tools/generate_pairs.py",
            "parent_receipt_source": base_receipt.get("source"),
            "parent_receipt_id": base_receipt.get("source_record_id"),
            **(_safe_json_loads(base_receipt.get("raw_payload_json"), {})),
        },
    }
    bank_row = {
        "source": "synthetic_pair_bank",
        "source_record_id": bank_id,
        "txn_ts": txn_date.isoformat(),
        "amount": round(bank_amount, 2),
        "currency": base_receipt.get("currency") or "USD",
        "merchant_raw": noisy,
        "description": noisy,
        "mcc": None,
        "mcc_description": None,
        "category_external": None,
        "subcategory_external": None,
        "raw_payload_json": {
            "group_id": group_id,
            "pair_role": "bank",
            "expected_match": True,
            "generator": "tools/generate_pairs.py",
            "offset_days": offset_days,
            "tip_amount": round(tip_amt, 2),
            "rounding_noise": round(rounding, 2),
            "canonical_merchant": canonical_merchant,
        },
    }
    mini = {
        "group_id": group_id,
        "receipt_source_record_id": receipt_id,
        "bank_source_record_id": bank_id,
        "should_match": True,
        "reason": "synthetic_positive_pair",
    }
    return receipt_row, bank_row, mini


def _make_negative_case(
    *,
    from_positive: Dict[str, Any],
    all_merchants: Sequence[str],
    neg_idx: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    group_id = from_positive["group_id"]
    receipt_id = from_positive["receipt_source_record_id"]
    base_date = _parse_iso_date(from_positive["receipt_date"]) or date.today()
    total = abs(float(from_positive["receipt_total"]))
    currency = from_positive.get("currency") or "USD"
    canonical = from_positive["receipt_merchant"]
    noisy_base = generate_noisy_merchant_raw(canonical, n=1, seed=rng.randint(1, 10**9))[0]

    mode = rng.choice(["same_date_diff_merchant", "same_merchant_diff_amount", "same_amount_diff_date"])
    if mode == "same_date_diff_merchant":
        alt = canonical
        if all_merchants:
            alt = rng.choice(all_merchants)
            if alt.upper() == canonical.upper() and len(all_merchants) > 1:
                alt = all_merchants[(all_merchants.index(alt) + 1) % len(all_merchants)]
        merchant_raw = generate_noisy_merchant_raw(alt, n=1, seed=rng.randint(1, 10**9))[0]
        txn_date = base_date
        amount = -total
    elif mode == "same_merchant_diff_amount":
        merchant_raw = noisy_base
        txn_date = base_date
        amount = -(total * rng.uniform(1.15, 1.6))
    else:
        merchant_raw = noisy_base
        txn_date = base_date + timedelta(days=rng.randint(5, 10))
        amount = -total

    bank_id = f"{group_id}_n{neg_idx:03d}"
    bank_row = {
        "source": "synthetic_pair_bank_negative",
        "source_record_id": bank_id,
        "txn_ts": txn_date.isoformat(),
        "amount": round(amount, 2),
        "currency": currency,
        "merchant_raw": merchant_raw,
        "description": merchant_raw,
        "mcc": None,
        "mcc_description": None,
        "category_external": None,
        "subcategory_external": None,
        "raw_payload_json": {
            "group_id": group_id,
            "pair_role": "bank_negative",
            "expected_match": False,
            "generator": "tools/generate_pairs.py",
            "non_match_mode": mode,
            "target_receipt_source_record_id": receipt_id,
        },
    }
    mini = {
        "group_id": group_id,
        "receipt_source_record_id": receipt_id,
        "bank_source_record_id": bank_id,
        "should_match": False,
        "reason": mode,
    }
    return bank_row, mini


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--pair-count", type=int, default=50, help="Number of positive matched pairs.")
    parser.add_argument(
        "--mini-out",
        default="data/gold/matching_gold_mini_set.jsonl",
        help="Output JSONL path for pair labels (positive + negatives).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-neg-per-10", type=int, default=2)
    parser.add_argument("--max-neg-per-10", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    staging_db = Path(args.staging_db)
    if not staging_db.exists():
        print(f"Staging DB not found: {staging_db}")
        print("Import receipts first (e.g., import_receipts_sroie.py).")
        return 1

    candidates = _load_receipt_candidates(staging_db)
    if not candidates:
        print("No eligible receipt candidates found in staging_receipts.")
        print("Need merchant_raw + total > 0 rows.")
        return 1

    rng = random.Random(args.seed)
    all_merchants = sorted({str(c["merchant_raw"]) for c in candidates if c.get("merchant_raw")})

    synthetic_receipts: List[Dict[str, Any]] = []
    synthetic_banks: List[Dict[str, Any]] = []
    mini_rows: List[Dict[str, Any]] = []
    positive_info: List[Dict[str, Any]] = []

    for i in range(args.pair_count):
        base = rng.choice(candidates)
        receipt_row, bank_row, mini = _make_positive_pair(base_receipt=base, pair_idx=i + 1, rng=rng)
        synthetic_receipts.append(receipt_row)
        synthetic_banks.append(bank_row)
        mini_rows.append(mini)
        positive_info.append(
            {
                "group_id": mini["group_id"],
                "receipt_source_record_id": mini["receipt_source_record_id"],
                "receipt_date": receipt_row["receipt_ts"],
                "receipt_total": receipt_row["total"],
                "receipt_merchant": receipt_row["merchant_raw"],
                "currency": receipt_row["currency"],
            }
        )

    neg_counter = 0
    for i in range(0, len(positive_info), 10):
        block = positive_info[i : i + 10]
        n_neg = rng.randint(args.min_neg_per_10, args.max_neg_per_10)
        for _ in range(n_neg):
            neg_counter += 1
            pos = rng.choice(block)
            bank_row, mini = _make_negative_case(
                from_positive=pos,
                all_merchants=all_merchants,
                neg_idx=neg_counter,
                rng=rng,
            )
            synthetic_banks.append(bank_row)
            mini_rows.append(mini)

    inserted_receipts = staging.insert_staging_receipts(
        synthetic_receipts, path=staging_db, batch_size=args.batch_size
    )
    inserted_banks = staging.insert_staging_transactions(
        synthetic_banks, path=staging_db, batch_size=args.batch_size
    )

    outp = Path(args.mini_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fh:
        for row in mini_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = staging.get_staging_counts(staging_db)
    print("Pair generation complete.")
    print(f"- positive_pairs: {args.pair_count}")
    print(f"- negatives: {len(mini_rows) - args.pair_count}")
    print(f"- inserted_receipts: {inserted_receipts}")
    print(f"- inserted_bank_rows: {inserted_banks}")
    print(f"- mini_set_out: {outp}")
    print(f"- staging_transactions_total: {counts['staging_transactions']}")
    print(f"- staging_receipts_total: {counts['staging_receipts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


