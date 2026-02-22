# SourceTax — Transaction Classification & Tax Automation

An AI-driven pipeline that ingests transaction data from banks, POS systems, and receipt images, then normalizes, extracts, categorizes, and exports to QuickBooks and Schedule C tax forms.

## Quick Start

### Prerequisites

**Tesseract OCR** (for receipt image extraction)

This project uses `pytesseract`, which requires the [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) binary to be installed:

**Windows:**  
Download and install from [Tesseract releases](https://github.com/UB-Mannheim/tesseract/releases). Default path: `C:\Program Files\Tesseract-OCR`. After install, set in your Python code or environment:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**macOS:**  
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**  
```bash
sudo apt-get install tesseract-ocr
```

### Installation

```bash
# Clone repo
git clone <repo>
cd sourcetax

# Install with core dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For enhanced matching (optional, recommended)
pip install -e ".[matching]"
```

### Run Demo

```bash
# Full end-to-end: ingest samples, match receipts, categorize, export
python tools/generate_reports.py

# Outputs:
#   outputs/quickbooks_import.csv     — QB import format
#   outputs/schedule_c_totals.csv     — Schedule C category breakdown
#   outputs/audit_pack.csv            — full transaction trail with confidence
```

### Launch Review UI

```bash
# Interactive Streamlit dashboard for reviewing/overriding categories
streamlit run app_review.py

# Opens http://localhost:8501
# Features:
#   - Dashboard: metrics, auto-match/categorize buttons
#   - Unmatched receipts/transactions: review and link
#   - Category override: change predictions and save overrides
```

## Architecture

A modular, staged pipeline:

1. **Ingest** (`src/sourcetax/ingest.py`)  
   Read CSVs (QB, bank, Toast) and receipt image files into canonical schema.

2. **Normalize** (`src/sourcetax/schema.py`)  
   Standardize dates, amounts, merchant names, directions across all sources.

3. **Store** (`src/sourcetax/storage.py`)  
   Persist canonical records to SQLite for querying and iteration.

4. **Extract** (`src/sourcetax/receipts.py` — Phase 2)  
   OCR receipt images (Tesseract/EasyOCR), parse fields (date, merchant, total, tax).

5. **Match** (`src/sourcetax/matching.py` — Phase 2)  
   Link receipt documents to bank transactions with fuzzy matching.

6. **Categorize** (`src/sourcetax/categorization.py` — Phase 2)  
   Rules-based classification (merchant exact/fuzzy match, keywords, user overrides).

7. **Export** (`src/sourcetax/exporter.py`)  
   Generate QuickBooks CSV, Schedule C totals, audit pack.

## Phase Completion Status

### Phase 0 — ✅ Complete

**Foundation:** Canonical schema, sample data, basic CSV ingestion, SQLite storage.

**What's Here:**
- Real-format sample data (QB, Toast, bank CSV)
- Public datasets (FUNSD forms, CORD receipts)
- `CanonicalRecord` dataclass with 13 fields
- CSV ingestors for toast, bank, quickbooks
- SQLite schema and basic I/O

### Phase 1 — ✅ Complete (phase1 branch)

**Schema & Cleanup:** Enhanced canonical schema, fixed encoding, code formatting, repo hygiene.

**What's Here:**
- Added direction field (expense|income), category fields, source traceability
- Fixed pyproject.toml to PEP 621
- Updated .gitignore to exclude large artifacts
- All tests pass, demo works deterministically

### Phase 2 — ✅ Complete (phase2 branch)

**Receipt Extraction, Matching, Categorization, Review UI.**

**Phase 2.1 — Receipt OCR:**
- `src/sourcetax/receipts.py` — Tesseract/EasyOCR integration
- Field extraction: date, merchant, total, tax, tip
- Heuristic-based (regex + keyword search)

**Phase 2.2 — Transaction Matching:**
- `src/sourcetax/matching.py` — Receipt ↔ bank fuzzy matching
- Scoring: date (±3d), amount (±$10), merchant (>80% similarity)
- `list_unmatched_receipts()`, `list_unmatched_transactions()` for UI

**Phase 2.3 — Categorization Rules Engine:**
- `src/sourcetax/categorization.py` — Priority rules (learned > exact > fuzzy > keywords)
- Keyword rules: UBER→Travel, STARBUCKS→Meals, HOME DEPOT→Supplies, etc.
- Learned overrides: user sets merchant category once, applies to all

**Phase 2.4 — Exports & Audit:**
- QB CSV uses `category_final` (user overrides)
- Schedule C totals by category
- Audit pack: full record trail with match scores + confidence
- Metrics export: record counts, expense totals, match rates

**UI:**
- `app_review.py` — Streamlit dashboard
  - Dashboard: metrics, auto-match/auto-categorize buttons
  - Unmatched receipts: list + detail view with OCR excerpt
  - Unmatched transactions: identify orphans
  - Match review: approve matches, override categories, save
- `src/sourcetax/taxonomy.py` — category and merchant mapping helpers
- `data/samples/` — real-like export examples
- `data/forms/funsd/` — FUNSD dataset (149 forms + images)
- `data/combined_dataset/combined_samples.jsonl` — unified training/dev dataset

**To regenerate samples or combined dataset:**
```bash
python tools/generate_training_samples.py    # QuickBooks + Toast + Plaid -> combined_samples.jsonl
python tools/convert_funsd_to_combined.py    # Append FUNSD to combined_samples.jsonl
python tools/count_combined.py               # Show composition counts
python tools/generate_reports.py             # Ingest samples and generate exports
```

**Dataset Composition (combined_samples.jsonl):**
- QuickBooks transactions: 7 examples
- Toast POS transactions: 7 examples
- Bank transactions (Plaid-format): 3 examples
- Receipt text samples: 3 examples
- FUNSD forms: 149 examples (with images in `data/forms/funsd/images/`)
- **Total:** 169 canonical JSONL records

## Using the Pipeline

### Python API

```python
from pathlib import Path
from sourcetax import storage, ingest, matching, categorization, exporter

# 1. Ingest from CSV
records = ingest.read_csv("data/samples/bank_chase.csv", source="bank")
for record in records:
    print(record.merchant_raw, record.amount)

# 2. Store to database
db_path = Path("data/store.db")
ingest.ingest_and_store(["data/samples/bank_chase.csv"], source="bank", db_path=db_path)

# 3. Retrieve from database
records = storage.get_all_records(db_path)
print(f"Total records: {len(records)}")

# 4. Match receipts to transactions
matched = matching.match_all_receipts(records, db_path=db_path)
unmatched_receipts = matching.list_unmatched_receipts(records)

# 5. Categorize
categorized = categorization.categorize_all_records(records, db_path=db_path)

# 6. Export
exporter.generate_quickbooks_csv(records, Path("outputs/quickbooks_import.csv"))
exporter.compute_schedule_c_totals(records)
```

## Data Sources

**Sample Data in `data/samples/`:**
- QB: `bank_bofa.csv`, `bank_chase.csv`, `bank_sample.csv` (real-format CSV)
- Toast: `toast_sample.csv` (POS transactions)
- Plaid: `plaid_sample_small.json` (bank API format)
- Receipts: `receipt_sample.txt` (OCR text)

**Public Datasets:**
- **FUNSD** (149 forms): `data/forms/funsd/9.json` … `data/forms/funsd_127.json` + images
- **CORD** receipts: `data/samples/cord/` (subset used in Phase 2 evaluation)

**Taxonomy & Mappings:**
- `data/taxonomy/schedule_c_taxonomy.json` — Schedule C category hierarchy
- `data/mappings/merchant_category.csv` — merchant → category rules (format: merchant,category_code,category_name,notes)

## API Documentation

### Core Modules

**`src/sourcetax/schema.py`**  
`CanonicalRecord` dataclass (13 fields):
- `id`, `source`, `source_record_id`, `transaction_date`, `merchant_raw`, `merchant_norm`
- `amount` (float), `currency` (str), `direction` (str: expense|income), `payment_method`
- `category_pred`, `category_final`, `confidence` (float)
- `matched_transaction_id`, `match_score`, `evidence_keys`
- `raw_payload` (dict)

**Schema Invariants (locked for Phase 3 ML):**
- **Direction:** Always "expense" (amount positive) or "income" (amount negative). Sign is implicit in direction, not amount.
- **Merchant normalization:** Use `normalize_merchant(text)` consistently. Examples: "STARBUCKS COFFEE 123" → "Starbucks", "UBER TRIP" → "Uber".
- **Category coding:** Always populate `category_code` (stable: 'meals', 'travel', 'office', etc.) alongside `category_name` (human-readable: 'Meals and Lodging'). This ensures ML models can use both.
- **Amounts:** Always store with one convention (expenses positive). Direction field disambiguates, not the sign.
- **Dates:** ISO 8601 format (YYYY-MM-DD). No timezones. OCR extracts best guess; gold set has ground truth.
- **Confidence:** Float 0.0–1.0. Metrics:
  - For rules-based categorization: 0.9–1.0 (confident)
  - For fuzzy matches: score of best match (0.0–1.0)
  - For ML predictions: probability of predicted class
- **Match score:** 0.0–1.0 if matched_transaction_id is set. Otherwise null.

**`src/sourcetax/ingest.py`**  
Key functions:
- `read_csv(path, source) → Iterator[dict]` — Read CSV, yield records
- `normalize_to_canonical(row, source) → CanonicalRecord` — Convert any source to canonical
- `ingest_and_store(paths, source, db_path)` — Read + store to SQLite

**`src/sourcetax/storage.py`**  
Key functions:
- `ensure_db(db_path)` — Create schema if not exists
- `insert_record(record, db_path)` — Persist canonical record
- `get_all_records(db_path) → List[CanonicalRecord]` — Retrieve all from database

**`src/sourcetax/receipts.py`** (Phase 2.1)  
Key functions:
- `extract_ocr_text(image_path, backend='tesseract') → str` — OCR receipt image
- `parse_receipt_text(text) → dict` — Extract date, merchant, total, tax, tip
- `ingest_receipt_file(image_path, source='receipt') → CanonicalRecord` — End-to-end

**`src/sourcetax/matching.py`** (Phase 2.2)  
Key functions:
- `match_receipt_to_bank(receipt, bank_txns) → (score, matched_txn)` — Single match
- `match_all_receipts(records, db_path=None) → List[CanonicalRecord]` — Bulk matching
- `list_unmatched_receipts(records) → List[CanonicalRecord]` — Orphan receipts
- `list_unmatched_transactions(records) → List[CanonicalRecord]` — Orphan bank transactions

**`src/sourcetax/categorization.py`** (Phase 2.3)  
Key functions:
- `categorize_record(record) → CanonicalRecord` — Apply rules once
- `categorize_all_records(records, db_path=None) → List[CanonicalRecord]` — Bulk categorization
- `load_merchant_category_map(csv_path) → dict` — Load merchant → category learned overrides

**`src/sourcetax/exporter.py`** (Phase 2.4)  
Key functions:
- `generate_quickbooks_csv(records, output_path)` — QB format (uses `category_final`)
- `compute_schedule_c_totals(records) → dict` — By-category breakdown
- `write_schedule_c_csv(totals, output_path)` — SC CSV export
- `export_audit_pack(records, output_path)` — Full detail export
- `export_metrics(records) → dict` — Count, total, match rate, etc.

## Directory Structure

```
sourcetax/
├── src/sourcetax/           # Core packages
│   ├── __init__.py         # Package marker
│   ├── schema.py           # CanonicalRecord dataclass
│   ├── ingest.py           # Ingestion pipeline (CSV/receipt)
│   ├── storage.py          # SQLite persistence
│   ├── receipts.py         # OCR + field extraction (Phase 2.1)
│   ├── matching.py         # Receipt↔bank matching (Phase 2.2)
│   ├── categorization.py   # Rules engine (Phase 2.3)
│   └── exporter.py         # QB/SC/audit exports (Phase 2.4)
├── data/
│   ├── samples/            # Sample data (QB, Toast, Plaid, receipts)
│   ├── forms/funsd/        # FUNSD dataset (149 forms)
│   ├── combined_dataset/   # combined_samples.jsonl
│   ├── taxonomy/           # Schedule C taxonomy JSON
│   ├── mappings/           # Merchant category mapping CSV
│   └── store.db            # SQLite database
├── outputs/                # Generated exports (QB CSV, Schedule C)
├── tools/                  # Utility scripts
│   ├── generate_reports.py # End-to-end demo (runs all phases)
│   ├── generate_training_samples.py
│   ├── convert_funsd_to_combined.py
│   ├── count_combined.py
│   └── fetch_public_datasets.py
├── app_review.py           # Streamlit dashboard UI
├── tests/                  # Test suite
├── pyproject.toml          # Package metadata + dependencies
└── README.md               # This file
```

## Evaluation

A gold standard dataset is included in `data/gold/gold_transactions.jsonl` (~10 hand-labeled canonical records) for evaluation:

```bash
# Run evaluation on current pipeline
python tools/eval.py

# Output: Categorization accuracy, matching precision/recall, OCR extraction accuracy
```

**Gold Dataset:**
- ~10 curated transactions covering major sources (bank, receipt, Toast, QuickBooks)
- Hand-labeled `category_final` (ground truth)
- Receipt↔bank links for matching evaluation
- OCR extraction ground truth (merchant, date, amount)

**Metrics Tracked:**
- **Categorization Accuracy:** Rules engine + overrides vs. ground truth
- **Matching Precision/Recall:** Receipt→bank matching performance
- **Extraction Accuracy:** Merchant, date, amount OCR correctness

Before Phase 3 ML work, expand gold dataset to ~200 records using `app_review.py` (mark good matches, override categories, save). This becomes your evaluation set.

## Roadmap

### Phase 3 — ML Categorization Baseline

**Status:** Foundation locked. Gold dataset + eval script ready.

**Work:**
- Expand gold set to 200 records using review UI
- Train TF-IDF + logistic regression baseline on golden labels
- Evaluate: confusion matrix, precision/recall by Schedule C category
- Compare: rules-only vs. ML vs. ensemble

### Phase 4 — Advanced Exports & Reconciliation

**Planned:** GL entries, journal transactions, audit trail.

### Phase 5 — Multi-User & Cloud

**Planned:** Web UI, user accounts, cloud storage.

## Contributing

Report issues, submit PRs. Development tools:

```bash
black src/sourcetax tests          # Format code
ruff check src/sourcetax tests     # Lint
pytest tests/ -v                   # Run tests
```

## License

MIT
