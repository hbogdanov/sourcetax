"""Phase 4 reconciliation queues and summary metrics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .categorization import categorize_by_keywords, load_merchant_category_map


def _rows(query: str, params: tuple = (), db_path: str = "data/store.db") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, params)
    out = [dict(r) for r in cur.fetchall()]
    conn.close()
    return out


def unmatched_receipts(db_path: str = "data/store.db") -> List[Dict[str, Any]]:
    return _rows(
        """
        SELECT id, transaction_date, merchant_raw, merchant_norm, amount, confidence,
               matched_transaction_id, source
        FROM canonical_records
        WHERE source = 'receipt' AND matched_transaction_id IS NULL
        ORDER BY transaction_date DESC, id
        """,
        db_path=db_path,
    )


def unmatched_bank_transactions(db_path: str = "data/store.db") -> List[Dict[str, Any]]:
    return _rows(
        """
        SELECT id, transaction_date, merchant_raw, merchant_norm, amount, confidence, source
        FROM canonical_records
        WHERE source IN ('bank', 'toast', 'quickbooks')
          AND id NOT IN (
            SELECT matched_transaction_id
            FROM canonical_records
            WHERE matched_transaction_id IS NOT NULL
          )
        ORDER BY transaction_date DESC, id
        """,
        db_path=db_path,
    )


def low_confidence_categorizations(
    db_path: str = "data/store.db",
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    return _rows(
        """
        SELECT id, transaction_date, merchant_raw, merchant_norm, amount, source,
               category_pred, category_final, confidence
        FROM canonical_records
        WHERE COALESCE(category_final, '') = ''
          AND confidence IS NOT NULL
          AND confidence < WARNING:
        ORDER BY confidence ASC, transaction_date DESC
        """,
        (threshold,),
        db_path=db_path,
    )


def conflicts_queue(db_path: str = "data/store.db") -> List[Dict[str, Any]]:
    """Rules-vs-ML disagreement queue.

    Uses raw_payload hints when present:
    - raw_payload.ml_prediction
    - raw_payload.model_pred
    - raw_payload.ensemble_decision (optional)
    Falls back to recomputed rules-only prediction for `merchant_raw`.
    """
    rows = _rows(
        """
        SELECT id, transaction_date, merchant_raw, merchant_norm, amount, source,
               category_pred, category_final, confidence, raw_payload
        FROM canonical_records
        ORDER BY transaction_date DESC, id
        """,
        db_path=db_path,
    )
    merchant_map = load_merchant_category_map()
    conflicts: List[Dict[str, Any]] = []
    for row in rows:
        raw_payload = row.get("raw_payload")
        if isinstance(raw_payload, str):
            try:
                raw_payload = json.loads(raw_payload)
            except Exception:
                raw_payload = {}
        if not isinstance(raw_payload, dict):
            raw_payload = {}

        merchant = row.get("merchant_raw") or ""
        # Recompute rules prediction using keyword path for a deterministic rule-only comparator.
        rules_pred_conf = categorize_by_keywords(merchant)
        if not rules_pred_conf:
            rules_pred = None
            rules_conf = None
            # If merchant map exact/fuzzy was used upstream it may already be in category_pred.
        else:
            rules_pred, rules_conf = rules_pred_conf

        ml_pred = raw_payload.get("ml_prediction") or raw_payload.get("model_pred")
        ensemble_decision = raw_payload.get("ensemble_decision")

        if ml_pred and rules_pred and str(ml_pred) != str(rules_pred):
            conflicts.append(
                {
                    "id": row.get("id"),
                    "transaction_date": row.get("transaction_date"),
                    "merchant_raw": merchant,
                    "amount": row.get("amount"),
                    "source": row.get("source"),
                    "rules_pred": rules_pred,
                    "rules_conf": rules_conf,
                    "ml_pred": ml_pred,
                    "ensemble_decision": ensemble_decision,
                    "category_pred": row.get("category_pred"),
                    "category_final": row.get("category_final"),
                    "confidence": row.get("confidence"),
                }
            )
    return conflicts


def summary_metrics(db_path: str = "data/store.db", low_conf_threshold: float = 0.7) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS n FROM canonical_records")
    total_records = int(cur.fetchone()["n"])

    cur.execute("SELECT COUNT(*) AS n FROM canonical_records WHERE source = 'receipt'")
    total_receipts = int(cur.fetchone()["n"])

    cur.execute(
        "SELECT COUNT(*) AS n FROM canonical_records WHERE source = 'receipt' AND matched_transaction_id IS NOT NULL"
    )
    matched_receipts = int(cur.fetchone()["n"])

    cur.execute("SELECT AVG(confidence) AS avg_conf FROM canonical_records WHERE confidence IS NOT NULL")
    avg_conf = cur.fetchone()["avg_conf"]

    cur.execute(
        """
        SELECT COALESCE(merchant_norm, merchant_raw, '') AS merchant_key,
               COUNT(*) AS cnt,
               AVG(COALESCE(confidence, 0)) AS avg_conf
        FROM canonical_records
        GROUP BY merchant_key
        HAVING COUNT(*) >= 1
        ORDER BY avg_conf ASC, cnt DESC, merchant_key ASC
        LIMIT 10
        """
    )
    ambiguous_merchants = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT COUNT(*) AS n
        FROM canonical_records
        WHERE COALESCE(category_final, '') = ''
          AND confidence IS NOT NULL
          AND confidence < WARNING:
        """,
        (low_conf_threshold,),
    )
    low_conf_count = int(cur.fetchone()["n"])

    conn.close()

    conflicts = conflicts_queue(db_path=db_path)
    return {
        "total_records": total_records,
        "total_receipts": total_receipts,
        "matched_receipts": matched_receipts,
        "match_rate": (matched_receipts / total_receipts) if total_receipts else 0.0,
        "avg_confidence": float(avg_conf) if avg_conf is not None else None,
        "low_confidence_queue_size": low_conf_count,
        "conflicts_queue_size": len(conflicts),
        "top_ambiguous_merchants": ambiguous_merchants,
    }


def export_reconciliation_reports(
    db_path: str = "data/store.db",
    out_dir: str = "outputs/reconciliation",
    low_conf_threshold: float = 0.7,
) -> Dict[str, str]:
    """Write reconciliation queues and summary metrics to CSV/JSON files."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    queues = {
        "unmatched_receipts": unmatched_receipts(db_path),
        "unmatched_bank_txns": unmatched_bank_transactions(db_path),
        "low_confidence_categorizations": low_confidence_categorizations(db_path, low_conf_threshold),
        "conflicts": conflicts_queue(db_path),
    }
    outputs: Dict[str, str] = {}
    for name, rows in queues.items():
        path = out_base / f"{name}.csv"
        # Stable columns even when empty.
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
        import csv

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
        outputs[name] = str(path)

    summary = summary_metrics(db_path=db_path, low_conf_threshold=low_conf_threshold)
    summary_path = out_base / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["summary_metrics"] = str(summary_path)
    return outputs
