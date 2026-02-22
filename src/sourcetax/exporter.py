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
