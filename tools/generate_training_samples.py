import csv
import json
import os
from datetime import datetime

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLES = os.path.join(ROOT, "data", "samples")
OUTDIR = os.path.join(ROOT, "data", "combined_dataset")
os.makedirs(OUTDIR, exist_ok=True)


def to_iso(d):
    try:
        return datetime.fromisoformat(d).date().isoformat()
    except Exception:
        return d


def process_quickbooks(path, out_path):
    with (
        open(path, newline="", encoding="utf-8") as f,
        open(out_path, "w", encoding="utf-8") as out,
    ):
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("Date"):
                continue
            rec = {
                "source": "quickbooks",
                "merchant_name": r.get("Payee") or r.get("Description"),
                "transaction_date": to_iso(r.get("Date", "")),
                "amount": float(r.get("Amount") or 0),
                "currency": "USD",
                "raw": r,
                "type": "accounting",
            }
            out.write(json.dumps(rec) + "\n")


def process_toast_accounting(path, out_path):
    with (
        open(path, newline="", encoding="utf-8") as f,
        open(out_path, "a", encoding="utf-8") as out,
    ):
        reader = csv.DictReader(f)
        for r in reader:
            rec = {
                "source": "toast",
                "order_id": r.get("Order ID"),
                "merchant_name": "Toast Merchant",
                "transaction_date": to_iso(r.get("Date", "")),
                "amount": float(r.get("Total") or 0),
                "gl_account": r.get("GL Account"),
                "raw": r,
                "type": "pos",
            }
            out.write(json.dumps(rec) + "\n")


def process_plaid(path, out_path):
    with open(path, encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as out:
        data = json.load(f)
        for r in data.get("transactions", []):
            rec = {
                "source": "bank",
                "merchant_name": r.get("merchant_name") or r.get("name"),
                "transaction_date": to_iso(r.get("date", "")),
                "amount": float(r.get("amount") or 0),
                "raw": r,
                "type": "bank",
            }
            out.write(json.dumps(rec) + "\n")


def process_receipts_txt(receipts_dir, out_path):
    for name in os.listdir(receipts_dir):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(receipts_dir, name)
        with open(path, encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as out:
            text = f.read()
            rec = {
                "source": "receipt",
                "merchant_name": None,
                "transaction_date": None,
                "amount": None,
                "raw_text": text,
                "filename": name,
                "type": "receipt",
            }
            out.write(json.dumps(rec) + "\n")


def main():
    out_path = os.path.join(OUTDIR, "combined_samples.jsonl")
    # Start fresh
    if os.path.exists(out_path):
        os.remove(out_path)

    qb = os.path.join(SAMPLES, "quickbooks_sample.csv")
    if os.path.exists(qb):
        process_quickbooks(qb, out_path)

    toast = os.path.join(SAMPLES, "toast_accounting_real_sample.csv")
    if os.path.exists(toast):
        process_toast_accounting(toast, out_path)

    plaid = os.path.join(SAMPLES, "plaid_sample_small.json")
    if os.path.exists(plaid):
        process_plaid(plaid, out_path)

    receipts_dir = os.path.join(SAMPLES, "receipts")
    if os.path.exists(receipts_dir):
        process_receipts_txt(receipts_dir, out_path)

    # If any CORD json files exist, attempt to import minimal fields
    cord_dir = os.path.join(SAMPLES, "cord")
    for name in os.listdir(cord_dir) if os.path.exists(cord_dir) else []:
        if name.lower().endswith(".json"):
            path = os.path.join(cord_dir, name)
            with open(path, encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as out:
                try:
                    data = json.load(f)
                    # CORD uses 'valid_line' entries; we'll flatten basic metadata if present
                    rec = {"source": "cord_sample", "raw": data, "type": "receipt"}
                    out.write(json.dumps(rec) + "\n")
                except Exception:
                    continue

    print("Wrote combined samples to", out_path)


if __name__ == "__main__":
    main()
