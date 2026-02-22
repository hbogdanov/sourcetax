import sqlite3
from pathlib import Path
import json
from typing import Dict, Any

DB_PATH = Path('data') / 'store.db'


def ensure_db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS canonical_records (
        rowid INTEGER PRIMARY KEY,
        id TEXT,
        merchant_name TEXT,
        transaction_date TEXT,
        amount REAL,
        currency TEXT,
        payment_method TEXT,
        source TEXT,
        direction TEXT,
        category_code TEXT,
        source_record_id TEXT,
        raw_payload TEXT,
        confidence TEXT,
        tags TEXT
    )
    ''')
    conn.commit()
    conn.close()


def insert_record(rec: Dict[str, Any], path: Path = DB_PATH):
    ensure_db(path)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO canonical_records (id, merchant_name, transaction_date, amount, currency, payment_method, source, direction, category_code, source_record_id, raw_payload, confidence, tags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (
            rec.get('id'),
            rec.get('merchant_name'),
            rec.get('transaction_date'),
            rec.get('amount'),
            rec.get('currency'),
            rec.get('payment_method'),
            rec.get('source'),
            rec.get('direction'),
            rec.get('category_code'),
            rec.get('source_record_id'),
            json.dumps(rec.get('raw_payload') or {}),
            json.dumps(rec.get('confidence') or {}),
            json.dumps(rec.get('tags') or [])
        )
    )
    conn.commit()
    conn.close()
