import csv
from typing import Dict, Iterable
from pathlib import Path
from .schema import CanonicalRecord
from . import storage
from . import receipts
from .normalization import normalize_merchant_name
import datetime


def normalize_to_canonical(row: Dict[str, str], source: str) -> Dict:
    """Normalize known source row to canonical schema.
    Supported sources: 'toast', 'bank', 'quickbooks', 'receipt'
    Returns a dict suitable for `CanonicalRecord.from_normalized`.
    
    Phase 2: includes merchant_raw/merchant_norm, category_pred, single confidence float.
    """
    merchant_raw = None
    if source == "toast":
        merchant_raw = row.get("location")
    elif source == "bank":
        merchant_raw = row.get("description")
    elif source == "quickbooks":
        merchant_raw = row.get("Payee") or row.get("Description")
    elif source == "receipt":
        merchant_raw = row.get("merchant")

    out = {
        "id": None,
        "record_id": None,
        "source": source,
        "source_record_id": None,
        "transaction_date": None,
        "merchant_raw": merchant_raw,
        "merchant_norm": normalize_merchant_name(merchant_raw, case="lower") if merchant_raw else None,
        "merchant_name": merchant_raw,  # backward compat
        "amount": None,
        "currency": "USD",
        "direction": None,  # 'income' or 'expense'
        "payment_method": None,
        "category_pred": None,  # filled by taxonomy.apply()
        "category_final": None,  # user override (not set here)
        "confidence": 0.9,  # default; updated per source
        "matched_transaction_id": None,
        "match_score": None,
        "evidence_keys": [],
        "raw_payload": dict(row),
        "tags": [],
    }

    if source == "toast":
        out["id"] = row.get("order_id")
        out["source_record_id"] = row.get("order_id")
        out["transaction_date"] = row.get("date")
        out["amount"] = abs(parse_num(row.get("total")) or 0)
        out["payment_method"] = row.get("payment_type")
        amount_val = parse_num(row.get("total"))
        out["direction"] = "income" if (amount_val or 0) >= 0 else "expense"
        out["evidence_keys"] = ["toast_order"]

    elif source == "bank":
        out["transaction_date"] = row.get("date")
        amount_val = parse_num(row.get("amount"))
        out["amount"] = abs(amount_val) if amount_val else None
        out["payment_method"] = row.get("transaction_type")
        out["direction"] = "expense" if (amount_val or 0) < 0 else "income"
        out["evidence_keys"] = ["bank_transaction"]
        out["confidence"] = 0.95  # bank transactions are reliable

    elif source == "quickbooks":
        out["transaction_date"] = row.get("Date")
        amount_val = parse_num(row.get("Amount"))
        out["amount"] = abs(amount_val) if amount_val else None
        out["payment_method"] = row.get("Account")
        out["source_record_id"] = row.get("Transaction ID") or row.get("Payee")
        out["direction"] = "expense" if (amount_val or 0) < 0 else "income"
        out["evidence_keys"] = ["quickbooks_transaction"]
        out["confidence"] = 0.95

    elif source == "receipt":
        out["transaction_date"] = row.get("date")
        amount_val = parse_num(row.get("total"))
        out["amount"] = abs(amount_val) if amount_val else None
        out["direction"] = row.get("direction", "expense")
        out["evidence_keys"] = [row.get("receipt_file", "receipt")]
        out["confidence"] = float(row.get("confidence", 0.7))  # receipts are noisy

    # Normalize date format to ISO if possible
    if out["transaction_date"]:
        try:
            if "-" in out["transaction_date"]:
                dt = datetime.datetime.fromisoformat(out["transaction_date"])
            else:
                dt = datetime.datetime.strptime(out["transaction_date"], "%m/%d/%Y")
            out["transaction_date"] = dt.date().isoformat()
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
            return float(str(s).replace("$", "").replace(",", "").strip())
        except Exception:
            return None


def read_csv(path: str, source: str) -> Iterable[CanonicalRecord]:
    p = Path(path)
    with p.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            norm = normalize_to_canonical(r, source)
            yield CanonicalRecord.from_normalized(norm)


def ingest_and_store(path: str, source: str, db_path: str = "data/store.db") -> int:
    storage.DB_PATH = Path(db_path)
    count = 0
    for rec in read_csv(path, source):
        storage.insert_record(rec.to_row(), path=storage.DB_PATH)
        count += 1
    return count


def ingest_receipt_file(
    receipt_path: str | Path,
    db_path: str = "data/store.db",
    ocr_method: str = "tesseract",
) -> bool:
    """
    Ingest a single receipt file (JPG, PNG, PDF).
    Extract OCR text, parse fields, store as canonical record.
    
    Returns True if successful, False if extraction failed.
    """
    receipt_path = Path(receipt_path)
    if not receipt_path.exists():
        return False
    
    try:
        # Extract OCR + fields
        extracted = receipts.ingest_receipt(receipt_path, ocr_method=ocr_method)
        fields = extracted["extracted_fields"]
        
        # Normalize to canonical
        row = {
            "date": fields.get("date"),
            "merchant": fields.get("merchant"),
            "total": fields.get("total"),
            "tax": fields.get("tax"),
            "tip": fields.get("tip"),
            "receipt_file": receipt_path.name,
            "confidence": 0.7,  # receipts are noisy
        }
        norm = normalize_to_canonical(row, source="receipt")
        norm["raw_payload"]["ocr_text"] = extracted["ocr_full_text"]
        
        # Store
        storage.DB_PATH = Path(db_path)
        rec = CanonicalRecord.from_normalized(norm)
        storage.insert_record(rec.to_row(), path=storage.DB_PATH)
        return True
    except Exception as e:
        print(f"Error ingesting receipt {receipt_path}: {e}")
        return False


if __name__ == "__main__":
    # demo: ingest sample files and persist to SQLite
    n = ingest_and_store("data/samples/toast_sample.csv", "toast")
    print("ingested toast records:", n)
    n = ingest_and_store("data/samples/bank_sample.csv", "bank")
    print("ingested bank records:", n)
