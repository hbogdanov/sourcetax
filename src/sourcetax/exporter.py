import sqlite3
from pathlib import Path
import csv
import json
from .taxonomy import load_merchant_map


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
