import csv
from typing import Dict, Iterable
from pathlib import Path
from .schema import CanonicalRecord
from . import storage
import datetime


def normalize_to_canonical(row: Dict[str, str], source: str) -> Dict:
    """Normalize known source row to canonical schema.
    Supported sources: 'toast', 'bank', 'quickbooks'
    Returns a dict suitable for `CanonicalRecord.from_normalized`.
    """
    out = {
        'id': None,
        'merchant_name': None,
        'transaction_date': None,
        'amount': None,
        'currency': 'USD',
        'payment_method': None,
        'source': source,
        'direction': None,  # 'income' or 'expense'
        'category_code': None,
        'source_record_id': None,
        'raw_payload': dict(row),
        'confidence': {'merchant': 0.9, 'date': 0.9, 'amount': 0.9},
        'tags': []
    }

    if source == 'toast':
        out['id'] = row.get('order_id')
        out['merchant_name'] = row.get('location')
        out['transaction_date'] = row.get('date')
        out['amount'] = abs(parse_num(row.get('total')) or 0)
        out['payment_method'] = row.get('payment_type')
        out['source_record_id'] = row.get('order_id')
        # Toast: positive amounts are sales/income
        amount_val = parse_num(row.get('total'))
        out['direction'] = 'income' if (amount_val or 0) >= 0 else 'expense'
    elif source == 'bank':
        out['merchant_name'] = row.get('description')
        out['transaction_date'] = row.get('date')
        amount_val = parse_num(row.get('amount'))
        out['amount'] = abs(amount_val) if amount_val else None
        out['payment_method'] = row.get('transaction_type')
        # Bank: negative = expense, positive = income
        out['direction'] = 'expense' if (amount_val or 0) < 0 else 'income'
    elif source == 'quickbooks':
        out['merchant_name'] = row.get('Payee') or row.get('Description')
        out['transaction_date'] = row.get('Date')
        amount_val = parse_num(row.get('Amount'))
        out['amount'] = abs(amount_val) if amount_val else None
        out['payment_method'] = row.get('Account')
        out['source_record_id'] = row.get('Transaction ID') or row.get('Payee')
        # QuickBooks: negative = expense, positive = income
        out['direction'] = 'expense' if (amount_val or 0) < 0 else 'income'
    else:
        out['raw_payload'] = row

    # normalize date format to ISO if possible
    if out['transaction_date']:
        try:
            # common formats: YYYY-MM-DD or MM/DD/YYYY
            if '-' in out['transaction_date']:
                dt = datetime.datetime.fromisoformat(out['transaction_date'])
            else:
                dt = datetime.datetime.strptime(out['transaction_date'], '%m/%d/%Y')
            out['transaction_date'] = dt.date().isoformat()
        except Exception:
            pass

    return out


def parse_num(s: str):
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        try:
            return float(str(s).replace('$', '').replace(',', '').strip())
        except Exception:
            return None


def read_csv(path: str, source: str) -> Iterable[CanonicalRecord]:
    p = Path(path)
    with p.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            norm = normalize_to_canonical(r, source)
            yield CanonicalRecord.from_normalized(norm)


def ingest_and_store(path: str, source: str, db_path: str = 'data/store.db') -> int:
    storage.DB_PATH = Path(db_path)
    count = 0
    for rec in read_csv(path, source):
        storage.insert_record(rec.to_row(), path=storage.DB_PATH)
        count += 1
    return count


if __name__ == '__main__':
    # demo: ingest sample files and persist to SQLite
    n = ingest_and_store('data/samples/toast_sample.csv', 'toast')
    print('ingested toast records:', n)
    n = ingest_and_store('data/samples/bank_sample.csv', 'bank')
    print('ingested bank records:', n)
