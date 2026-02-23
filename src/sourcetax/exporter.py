import sqlite3
from pathlib import Path
import csv
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .taxonomy import load_merchant_map
from .categorization import KEYWORD_RULES


PIPELINE_VERSION = "phase4"


def _safe_json_loads(value, default):
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
        text = value.strip()
        if not text:
            return default
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            return default
    return default


def generate_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _detect_rule_hits(merchant: str | None) -> list[str]:
    if not merchant:
        return []
    merchant_upper = str(merchant).upper()
    hits = []
    for keyword in KEYWORD_RULES.keys():
        if keyword in merchant_upper:
            hits.append(keyword)
    return hits


def _infer_label_source(record: Dict[str, Any]) -> str:
    raw_payload = record.get("raw_payload") or {}
    if record.get("category_final"):
        return "human"
    source_hint = str(raw_payload.get("label_source", "")).strip().lower()
    if source_hint in {"rules", "ml", "ensemble", "human"}:
        return source_hint
    if raw_payload.get("ml_prediction") or raw_payload.get("model_name"):
        return "ml"
    if raw_payload.get("ensemble_decision"):
        return "ensemble"
    return "rules"


def _effective_category(record: Dict[str, Any]) -> str:
    return record.get("category_final") or record.get("category_pred") or "Uncategorized"


def _to_record_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    evidence = _safe_json_loads(d.get("evidence_keys"), [])
    raw_payload = _safe_json_loads(d.get("raw_payload"), {})
    tags = _safe_json_loads(d.get("tags"), [])
    d["evidence_keys"] = evidence if isinstance(evidence, list) else [str(evidence)]
    d["raw_payload"] = raw_payload if isinstance(raw_payload, dict) else {"raw_payload_value": raw_payload}
    d["tags"] = tags if isinstance(tags, list) else [str(tags)]
    return d


def fetch_records_full(db_path: str = "data/store.db") -> List[Dict[str, Any]]:
    """Fetch canonical records with all known fields for Phase 4 exports."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, source, source_record_id, transaction_date, merchant_raw, merchant_norm,
               amount, currency, direction, payment_method,
               category_pred, category_final, confidence,
               matched_transaction_id, match_score,
               evidence_keys, raw_payload, tags
        FROM canonical_records
        ORDER BY transaction_date, id
        """
    )
    rows = [_to_record_dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def fetch_all_records(db_path: str = "data/store.db"):
    """Fetch all canonical records, including Phase 2 fields for categorization."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """SELECT id, merchant_raw, transaction_date, amount, currency, payment_method, 
                  source, direction, category_pred, category_final, raw_payload 
           FROM canonical_records"""
    )
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        raw = json.loads(r[10]) if r[10] else {}
        if not isinstance(raw, dict):
            raw = {}
        yield {
            "id": r[0],
            "merchant_raw": r[1],
            "transaction_date": r[2],
            "amount": r[3],
            "currency": r[4],
            "payment_method": r[5],
            "source": r[6],
            "direction": r[7],
            "category_pred": r[8],
            "category_final": r[9],  # user override
            "raw_payload": raw,
        }


def export_transactions_enriched_csv(
    db_path: str = "data/store.db",
    out_path: str = "outputs/accounting_transactions_enriched.csv",
    pipeline_version: str = PIPELINE_VERSION,
    run_id: Optional[str] = None,
) -> str:
    """Accounting-grade transaction export with audit metadata per row."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = generate_run_id("tx")

    records = fetch_records_full(db_path)
    fieldnames = [
        "transaction_id",
        "date",
        "posting_date",
        "merchant_raw",
        "merchant_normalized",
        "amount",
        "currency",
        "direction",
        "predicted_category",
        "final_category",
        "effective_category",
        "confidence",
        "label_source",
        "receipt_id",
        "matched_transaction_id",
        "match_score",
        "evidence_pointers",
        "source",
        "source_record_id",
        "pipeline_version",
        "run_id",
    ]

    with outp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            evidence_keys = rec.get("evidence_keys") or []
            source = rec.get("source") or ""
            receipt_id = rec["id"] if source == "receipt" else ""
            writer.writerow(
                {
                    "transaction_id": rec.get("id") or "",
                    "date": rec.get("transaction_date") or "",
                    "posting_date": rec.get("transaction_date") or "",
                    "merchant_raw": rec.get("merchant_raw") or "",
                    "merchant_normalized": rec.get("merchant_norm") or "",
                    "amount": rec.get("amount") if rec.get("amount") is not None else "",
                    "currency": rec.get("currency") or "USD",
                    "direction": rec.get("direction") or "",
                    "predicted_category": rec.get("category_pred") or "",
                    "final_category": rec.get("category_final") or "",
                    "effective_category": _effective_category(rec),
                    "confidence": rec.get("confidence") if rec.get("confidence") is not None else "",
                    "label_source": _infer_label_source(rec),
                    "receipt_id": receipt_id,
                    "matched_transaction_id": rec.get("matched_transaction_id") or "",
                    "match_score": rec.get("match_score") if rec.get("match_score") is not None else "",
                    "evidence_pointers": json.dumps(evidence_keys, ensure_ascii=False),
                    "source": source,
                    "source_record_id": rec.get("source_record_id") or "",
                    "pipeline_version": pipeline_version,
                    "run_id": run_id,
                }
            )
    return str(outp)


def _schedule_c_account_name(category: str, direction: str) -> str:
    if direction == "income":
        return "Income"
    return category or "Uncategorized"


def export_gl_lines_csv(
    db_path: str = "data/store.db",
    out_path: str = "outputs/gl_lines.csv",
    pipeline_version: str = PIPELINE_VERSION,
    run_id: Optional[str] = None,
) -> str:
    """Export double-entry-ish GL lines (expense/income vs clearing account)."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = generate_run_id("gl")

    records = fetch_records_full(db_path)
    fieldnames = [
        "transaction_id",
        "posting_date",
        "account",
        "debit",
        "credit",
        "memo",
        "entity",
        "source",
        "evidence",
        "label_source",
        "pipeline_version",
        "run_id",
    ]

    with outp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            amt = rec.get("amount")
            if amt is None:
                continue
            amt = float(amt)
            if amt <= 0:
                continue
            direction = rec.get("direction") or "expense"
            category = _effective_category(rec)
            evidence = {
                "matched_transaction_id": rec.get("matched_transaction_id"),
                "match_score": rec.get("match_score"),
                "evidence_keys": rec.get("evidence_keys") or [],
            }
            common = {
                "transaction_id": rec.get("id") or "",
                "posting_date": rec.get("transaction_date") or "",
                "memo": rec.get("merchant_raw") or "",
                "entity": rec.get("merchant_raw") or "",
                "source": rec.get("source") or "",
                "evidence": json.dumps(evidence, ensure_ascii=False),
                "label_source": _infer_label_source(rec),
                "pipeline_version": pipeline_version,
                "run_id": run_id,
            }
            if direction == "income":
                primary = {
                    **common,
                    "account": "Bank Clearing",
                    "debit": f"{amt:.2f}",
                    "credit": "",
                }
                contra = {
                    **common,
                    "account": _schedule_c_account_name(category, direction),
                    "debit": "",
                    "credit": f"{amt:.2f}",
                }
            else:
                primary = {
                    **common,
                    "account": _schedule_c_account_name(category, direction),
                    "debit": f"{amt:.2f}",
                    "credit": "",
                }
                contra = {
                    **common,
                    "account": "Bank Clearing",
                    "debit": "",
                    "credit": f"{amt:.2f}",
                }
            writer.writerow(primary)
            writer.writerow(contra)
    return str(outp)


def export_audit_trail_jsonl(
    db_path: str = "data/store.db",
    out_path: str = "outputs/audit_trail.jsonl",
    pipeline_version: str = PIPELINE_VERSION,
    run_id: Optional[str] = None,
) -> str:
    """Export audit trail JSONL with inputs, transformations, rule hits, and final rationale."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = generate_run_id("audit")

    records = fetch_records_full(db_path)
    with outp.open("w", encoding="utf-8") as fh:
        for rec in records:
            raw_payload = rec.get("raw_payload") or {}
            merchant_raw = rec.get("merchant_raw")
            merchant_norm = rec.get("merchant_norm")
            rule_hits = _detect_rule_hits(merchant_raw)
            rule_rationale = []
            for hit in rule_hits:
                category, confidence = KEYWORD_RULES[hit]
                rule_rationale.append(
                    {"keyword": hit, "category": category, "confidence": confidence}
                )

            final_category = _effective_category(rec)
            label_source = _infer_label_source(rec)
            confidence = rec.get("confidence")

            decision_reason = {
                "label_source": label_source,
                "final_category": final_category,
                "confidence": confidence,
                "category_pred": rec.get("category_pred"),
                "category_final": rec.get("category_final"),
                "matched_transaction_id": rec.get("matched_transaction_id"),
                "match_score": rec.get("match_score"),
            }
            if label_source == "human":
                decision_reason["why"] = "human override present (category_final)"
            elif rule_rationale:
                decision_reason["why"] = "rules matched merchant keywords"
            elif rec.get("category_pred"):
                decision_reason["why"] = "predicted category present"
            else:
                decision_reason["why"] = "default Uncategorized"

            audit_row = {
                "transaction_id": rec.get("id"),
                "run_id": run_id,
                "pipeline_version": pipeline_version,
                "model_version_hash": str(raw_payload.get("model_version_hash") or raw_payload.get("model_hash") or "none"),
                "inputs": {
                    "source": rec.get("source"),
                    "source_record_id": rec.get("source_record_id"),
                    "transaction_date": rec.get("transaction_date"),
                    "merchant_raw": merchant_raw,
                    "amount": rec.get("amount"),
                    "currency": rec.get("currency"),
                    "direction": rec.get("direction"),
                    "payment_method": rec.get("payment_method"),
                    "raw_payload": raw_payload,
                },
                "transformations": {
                    "merchant_normalized": merchant_norm,
                    "evidence_keys": rec.get("evidence_keys") or [],
                    "tags": rec.get("tags") or [],
                },
                "rule_hits": rule_rationale,
                "match_scores": {
                    "matched_transaction_id": rec.get("matched_transaction_id"),
                    "match_score": rec.get("match_score"),
                },
                "final_decision": decision_reason,
            }
            fh.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
    return str(outp)


def export_accounting_grade_bundle(
    db_path: str = "data/store.db",
    out_dir: str = "outputs",
    pipeline_version: str = PIPELINE_VERSION,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """Export enriched transactions, GL lines, and audit trail in one run."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    run_id = run_id or generate_run_id("phase4")
    return {
        "transactions_enriched": export_transactions_enriched_csv(
            db_path=db_path,
            out_path=str(out_base / "accounting_transactions_enriched.csv"),
            pipeline_version=pipeline_version,
            run_id=run_id,
        ),
        "gl_lines": export_gl_lines_csv(
            db_path=db_path,
            out_path=str(out_base / "gl_lines.csv"),
            pipeline_version=pipeline_version,
            run_id=run_id,
        ),
        "audit_trail_jsonl": export_audit_trail_jsonl(
            db_path=db_path,
            out_path=str(out_base / "audit_trail.jsonl"),
            pipeline_version=pipeline_version,
            run_id=run_id,
        ),
        "run_id": run_id,
    }


def generate_quickbooks_csv(
    out_path: str = "outputs/quickbooks_import.csv", db_path: str = "data/store.db"
):
    """Generate QuickBooks import CSV using category_final (user override) if present."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    merchant_map = load_merchant_map()
    with outp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # QuickBooks columns: Date, Description, Amount, Payee, Category
        writer.writerow(["Date", "Description", "Amount", "Payee", "Category"])
        for rec in fetch_all_records(db_path):
            date = rec["transaction_date"] or ""
            merchant = rec["merchant_raw"] or rec["raw_payload"].get("description") or ""
            amount = rec["amount"] if rec["amount"] is not None else ""
            
            # Use category_final (user override) if present, else category_pred
            category = rec["category_final"] or rec["category_pred"] or "Uncategorized"
            
            writer.writerow([date, merchant, amount, merchant, category])
    return str(outp)


def compute_schedule_c_totals(db_path: str = "data/store.db"):
    """Compute Schedule C totals using category_final (user override) if present."""
    totals = {}
    count_by_category = {}
    
    # Only consider expense transactions
    for rec in fetch_all_records(db_path):
        amt = rec["amount"]
        direction = rec.get("direction")
        if amt is None or direction != "expense":
            continue
        
        # Use category_final (user override) if present, else category_pred
        category = rec["category_final"] or rec["category_pred"] or "Uncategorized"
        
        totals.setdefault(category, 0.0)
        count_by_category.setdefault(category, 0)
        totals[category] += amt
        count_by_category[category] += 1
    
    return totals, count_by_category


def write_schedule_c_csv(
    totals: dict, count_by_category: dict = None, out_path: str = "outputs/schedule_c_totals.csv"
):
    """Write Schedule C totals with optional transaction counts."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    if count_by_category is None:
        count_by_category = {}
    
    with outp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Category", "Amount", "Count"])
        for category, amt in sorted(totals.items()):
            count = count_by_category.get(category, 0)
            writer.writerow([category, f"{amt:.2f}", count])
    return str(outp)


def export_audit_pack(db_path: str = "data/store.db", out_path: str = "outputs/audit_pack.csv"):
    """
    Export audit pack: all transactions with category, match status, confidence.
    
    Columns: date, merchant, amount, direction, category, match_id, match_score, confidence, source
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            transaction_date, merchant_raw, amount, direction, 
            category_final, category_pred, matched_transaction_id, match_score, 
            confidence, source
        FROM canonical_records
        ORDER BY transaction_date DESC
    """)
    
    with outp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Date", "Merchant", "Amount", "Direction", 
            "Category (User)", "Category (Predicted)", 
            "Matched Transaction ID", "Match Score", "Confidence", "Source"
        ])
        
        for row in cur.fetchall():
            category = row["category_final"] or row["category_pred"] or "Uncategorized"
            writer.writerow([
                row["transaction_date"],
                row["merchant_raw"],
                f"{row['amount']:.2f}" if row["amount"] else "",
                row["direction"],
                row["category_final"] or "",
                row["category_pred"] or "",
                row["matched_transaction_id"] or "",
                f"{row['match_score']:.2%}" if row["match_score"] else "",
                f"{row['confidence']:.1%}" if row["confidence"] else "",
                row["source"],
            ])
    
    conn.close()
    return str(outp)


def export_metrics(db_path: str = "data/store.db") -> dict:
    """Calculate and return audit metrics."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Total records
    cur.execute("SELECT COUNT(*) FROM canonical_records")
    total_records = cur.fetchone()[0]
    
    # Receipts
    cur.execute("SELECT COUNT(*) FROM canonical_records WHERE source = 'receipt'")
    total_receipts = cur.fetchone()[0]
    
    # Matched receipts
    cur.execute(
        "SELECT COUNT(*) FROM canonical_records WHERE source = 'receipt' AND matched_transaction_id IS NOT NULL"
    )
    matched_receipts = cur.fetchone()[0]
    
    # Records needing review (low confidence and no final category)
    cur.execute(
        "SELECT COUNT(*) FROM canonical_records WHERE confidence < 0.7 AND category_final IS NULL"
    )
    needs_review = cur.fetchone()[0]
    
    # Expenses only
    cur.execute(
        "SELECT COUNT(*) FROM canonical_records WHERE direction = 'expense'"
    )
    total_expenses = cur.fetchone()[0]
    
    cur.execute(
        "SELECT SUM(amount) FROM canonical_records WHERE direction = 'expense'"
    )
    total_amount_result = cur.fetchone()[0]
    total_amount = float(total_amount_result) if total_amount_result else 0.0
    
    conn.close()
    
    return {
        "total_records": total_records,
        "total_receipts": total_receipts,
        "matched_receipts": matched_receipts,
        "unmatched_receipts": total_receipts - matched_receipts,
        "needs_review": needs_review,
        "total_expenses": total_expenses,
        "total_amount": total_amount,
        "match_rate": matched_receipts / total_receipts if total_receipts > 0 else 0.0,
    }


def count_gold_records(gold_path: str = "data/gold/gold_transactions.jsonl") -> int:
    """Count JSONL rows in the gold set file."""
    p = Path(gold_path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def export_gold_transactions_jsonl(
    db_path: str = "data/store.db",
    gold_path: str = "data/gold/gold_transactions.jsonl",
    append: bool = True,
) -> dict:
    """Export reviewed records (`category_final`) into the ML gold JSONL dataset.

    Appends new records by default and de-duplicates by `id` against the existing gold set.
    """
    outp = Path(gold_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if append and outp.exists():
        with outp.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec_id = rec.get("id")
                if rec_id:
                    existing_ids.add(rec_id)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, source, transaction_date, merchant_raw, merchant_norm, amount,
               category_pred, category_final, raw_payload, matched_transaction_id, match_score
        FROM canonical_records
        WHERE category_final IS NOT NULL AND TRIM(category_final) != ''
        ORDER BY transaction_date DESC, id
        """
    )

    rows = cur.fetchall()
    conn.close()

    mode = "a" if append else "w"
    exported = 0
    skipped_existing = 0

    with outp.open(mode, encoding="utf-8") as fh:
        for row in rows:
            if row["id"] and row["id"] in existing_ids:
                skipped_existing += 1
                continue

            raw_payload = row["raw_payload"]
            if isinstance(raw_payload, str):
                try:
                    raw_payload = json.loads(raw_payload)
                except Exception:
                    raw_payload = {"raw_payload_text": raw_payload}
            if not isinstance(raw_payload, dict):
                raw_payload = {}

            record = {
                "id": row["id"],
                "source": row["source"],
                "transaction_date": row["transaction_date"],
                "merchant_raw": row["merchant_raw"],
                "merchant_norm": row["merchant_norm"],
                "amount": row["amount"],
                "category_pred": row["category_pred"],
                "category_final": row["category_final"],
                "matched_transaction_id": row["matched_transaction_id"],
                "match_score": row["match_score"],
                "raw_payload": raw_payload,
            }
            fh.write(json.dumps(record) + "\n")
            exported += 1

    return {
        "path": str(outp),
        "exported": exported,
        "skipped_existing": skipped_existing,
        "total_after": count_gold_records(str(outp)),
    }
