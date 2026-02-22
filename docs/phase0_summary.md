# Phase 0 — Data Collection & Canonical Schema

**Status:** ✅ COMPLETE

Representative sample data, public datasets, and foundational pipeline infrastructure collected and integrated.

## What Was Built

### Canonical Schema

Implemented in `src/sourcetax/schema.py`:

```python
class CanonicalRecord:
    id: str                          # optional source id (order_id, transaction_id)
    merchant_name: str
    transaction_date: str            # ISO format
    amount: float
    currency: str
    payment_method: str
    source: str                      # 'toast', 'bank', 'receipt', 'funsd', etc.
    raw_payload: dict               # original row or OCR text
    confidence: dict                # per-field confidence scores
    tags: list                      # freeform tags
```

### Sample Data

**Real-format examples added under `data/samples/`:**

- `quickbooks_sample.csv` — 7 QBO import-ready transaction rows (meals, refunds, deposits, utilities, rent, fees, payments)
- `toast_accounting_real_sample.csv` — 7 Toast POS order lines with GL account mappings (sales, services, gift cards, fees)
- `plaid_sample_small.json` — 3 Plaid-format bank transactions (coffee shop, client income, utilities)
- `bank_sample.csv`, `bank_chase.csv`, `bank_bofa.csv` — multiple bank CSV templates
- `receipts/receipt1.txt`, `receipt2.txt`, `receipt3.txt` — 3 synthetic OCR-text receipt samples

### Public Datasets

- **FUNSD** (`data/forms/funsd/`): 149 form examples downloaded from Hugging Face, images stored in `data/forms/funsd/images/`, metadata appended to combined_samples.jsonl
- **CORD** (`data/samples/cord/`): small subset (README + sample image) from GitHub archive

### Taxonomy & Mappings

- `data/taxonomy/schedule_c_taxonomy.json` — Schedule C category codes and line references  
- `data/mappings/merchant_category.csv` — starter merchant-to-category mapping (rule-based baseline)

### Storage & Pipeline

- `src/sourcetax/ingest.py` — ingestion + normalization: reads CSV/JSON, produces `CanonicalRecord` instances
- `src/sourcetax/storage.py` — SQLite persistence with `ensure_db()` and `insert_record()` helpers
- `data/store.db` — SQLite database with `canonical_records` table (populated by ingest demo)
- `src/sourcetax/exporter.py` — QuickBooks CSV and Schedule C totals generator

### Combined Canonical Dataset

**File:** `data/combined_dataset/combined_samples.jsonl`

**Composition (169 records total):**
- QuickBooks: 7 examples
- Toast POS: 7 examples
- Bank (Plaid): 3 examples
- Receipt text: 3 examples
- FUNSD forms: 149 examples

**Schema:** Each line is a JSON object with `source`, `merchant_name`, `transaction_date`, `amount`, `type`, and `raw` fields.

## How to Use Phase 0 Artifacts

### Regenerate Combined Dataset

```powershell
# Convert QuickBooks, Toast, Plaid, receipt samples -> JSONL
python tools/generate_training_samples.py

# Append FUNSD metadata
python tools/convert_funsd_to_combined.py

# Report composition
python tools/count_combined.py
```

### Run End-to-End Demo (Ingest → Export)

```powershell
# Ingests samples, persists to DB, generates QuickBooks CSV and Schedule C totals
python tools/generate_reports.py
```

Output files:
- `outputs/quickbooks_import.csv` — ready to import to QuickBooks
- `outputs/schedule_c_totals.csv` — Schedule C category totals

### Fetch Additional Public Datasets (FUNSD already done)

```powershell
python tools/fetch_public_datasets.py
```

Attempts to download CORD and SROIE via Hugging Face (may not succeed in this environment; FUNSD already cached). Kaggle-hosted retail datasets require manual credentials setup.

## Key Implementation Details

- **Ingestion:** Normalizes dates to ISO format, parses amounts safely, handles missing values gracefully
- **Storage:** SQLite for development and small-to-medium datasets; extensible to PostgreSQL/BigQuery later
- **Exports:** Configurable Schedule C mapping; QuickBooks CSV uses standard Intuit import format
- **Taxonomy:** Extensible JSON structure for adding new categories or schedules (C, E, etc.)

## Files Modified/Added

**Core:**
- `src/sourcetax/core.py` — simple extraction primitives
- `src/sourcetax/schema.py` — canonical record dataclass + to_row/from_normalized helpers
- `src/sourcetax/ingest.py` — CSV/JSON readers and normalizers
- `src/sourcetax/storage.py` — SQLite DDL and insert helpers
- `src/sourcetax/exporter.py` — QuickBooks + Schedule C CSV generation
- `src/sourcetax/taxonomy.py` — category and merchant mapping loaders

**Data:**
- `data/samples/` — 4 real-format CSVs + 1 JSON + 1 README
- `data/forms/funsd/` — 149 form images and metadata JSON files
- `data/combined_dataset/combined_samples.jsonl` — canonical JSONL (169 records)
- `data/combined_dataset/README.md` — dataset description
- `data/taxonomy/schedule_c_taxonomy.json` — Schedule C taxonomy
- `data/mappings/merchant_category.csv` — merchant mappings

**Tools:**
- `tools/generate_training_samples.py` — convert samples to JSONL
- `tools/convert_funsd_to_combined.py` — append FUNSD to JSONL
- `tools/count_combined.py` — dataset composition report
- `tools/generate_reports.py` — end-to-end demo
- `tools/fetch_public_datasets.py` — HF dataset fetcher

**Documentation:**
- `docs/phase0_summary.md` — this file
- Updated `README.md` — full project overview

## Next: Phase 1 ✅

Phase 1 artifacts are already in place:
- Ingestion pipeline (`src/sourcetax/ingest.py`) reads samples and normalizes to canonical schema
- Storage module persists records to SQLite
- Exporter generates QuickBooks CSV and Schedule C totals
- Demo runnable via `python tools/generate_reports.py`

## Next: Phase 2 (Receipt Extraction / OCR)

Build OCR + field extraction pipeline:
- Use Tesseract for receipt image OCR
- Parse FUNSD forms using layout models
- Extract key fields (date, merchant, amount, tax) with confidence scores
- Write results back to canonical records

## Contributing

All Phase 0 artifacts ready for Phase 1+ development.

For questions or issues, see the root README.md.
