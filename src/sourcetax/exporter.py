import sqlite3
from pathlib import Path
import csv
import json
from .taxonomy import load_merchant_map


def fetch_all_records(db_path: str = 'data/store.db'):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT id, merchant_name, transaction_date, amount, currency, payment_method, source, direction, raw_payload FROM canonical_records')
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        raw = json.loads(r[8]) if r[8] else {}
        if not isinstance(raw, dict):
            raw = {}
        yield {
            'id': r[0],
            'merchant_name': r[1],
            'transaction_date': r[2],
            'amount': r[3],
            'currency': r[4],
            'payment_method': r[5],
            'source': r[6],
            'direction': r[7],
            'raw_payload': raw
        }


def generate_quickbooks_csv(out_path: str = 'outputs/quickbooks_import.csv', db_path: str = 'data/store.db'):
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    merchant_map = load_merchant_map()
    with outp.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        # Simple QuickBooks-like columns: Date, Description, Amount, Payee, Account
        writer.writerow(['Date', 'Description', 'Amount', 'Payee', 'Account'])
        for rec in fetch_all_records(db_path):
            payee = rec['merchant_name'] or rec['raw_payload'].get('description') or ''
            amount = rec['amount'] if rec['amount'] is not None else ''
            desc = rec['raw_payload'].get('description') or ''
            mapping = None
            if payee:
                mapping = merchant_map.get(payee.strip().upper())
            account = mapping['category_name'] if mapping else 'Uncategorized'
            writer.writerow([rec['transaction_date'] or '', desc or '', amount, payee or '', account])
    return str(outp)


def compute_schedule_c_totals(db_path: str = 'data/store.db'):
    merchant_map = load_merchant_map()
    totals = {}
    # only consider expense transactions
    for rec in fetch_all_records(db_path):
        amt = rec['amount']
        direction = rec.get('direction')
        if amt is None or direction != 'expense':
            continue
        payee = (rec['merchant_name'] or '').strip().upper()
        mapping = merchant_map.get(payee)
        code = mapping['category_code'] if mapping else 'OTH'
        totals.setdefault(code, 0.0)
        totals[code] += amt
    return totals


def write_schedule_c_csv(totals: dict, out_path: str = 'outputs/schedule_c_totals.csv'):
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['CategoryCode', 'Amount'])
        for code, amt in totals.items():
            writer.writerow([code, f"{amt:.2f}"])
    return str(outp)
