# SourceTax — Transaction Classification & Tax Automation

An AI-driven pipeline that ingests transaction data from POS systems, banks, and receipts, then normalizes, extracts, categorizes, and exports to QuickBooks and Schedule C tax forms.

## Quick Start

```bash
# Clone and install
git clone <repo>
cd sourcetax
pip install -e .

# Run demo ingestion and export
python tools/generate_reports.py

# Expected output: outputs/quickbooks_import.csv, outputs/schedule_c_totals.csv
```

## Architecture

A modular, staged pipeline:

1. **Ingest** (`src/sourcetax/ingest.py`)  
   Read receipts, bank CSVs, Toast POS exports into a canonical schema.

2. **Normalize** (`src/sourcetax/schema.py`)  
   Standardize dates, amounts, merchant names across all sources.

3. **Store** (`src/sourcetax/storage.py`)  
   Persist canonical records to SQLite for querying and experimentation.

4. **Extract** (Phase 2)  
   OCR receipt images (Tesseract), parse structured forms (FUNSD), etc.

5. **Categorize** (Phase 3)  
   Rule-based and ML-based classification to Schedule C categories.

6. **Export** (`src/sourcetax/exporter.py`)  
   Generate QuickBooks CSV, Schedule C totals, audit reports.

## Phase 0 — Complete ✅

**Status:** Representative sample data and public datasets collected. Canonical schema and basic ingestion/export pipeline implemented.

**What's Included:**
- Real-format sample data: QuickBooks CSV, Toast accounting, Plaid bank transactions, receipt text samples.
- Public datasets: FUNSD forms (149 examples + images), CORD subset in `data/samples/cord/`.
- Canonical schema: `src/sourcetax/schema.py` with `CanonicalRecord` dataclass.
- Ingestion pipeline: `src/sourcetax/ingest.py` with normalizers for CSV/JSON sources.
- Storage: SQLite database at `data/store.db` with `canonical_records` table.
- Export: `src/sourcetax/exporter.py` generates QuickBooks CSV and Schedule C totals.
- Taxonomy: `data/taxonomy/schedule_c_taxonomy.json` and merchant→category mapping `data/mappings/merchant_category.csv`.
- Combined dataset: `data/combined_dataset/combined_samples.jsonl` (169 canonical JSONL records: 7 QB, 7 Toast, 3 bank, 3 receipt, 149 FUNSD).

**Key Files:**
- `src/sourcetax/core.py` — simple extraction primitives
- `src/sourcetax/schema.py` — canonical record definition
- `src/sourcetax/ingest.py` — CSV/JSON reading and normalization
- `src/sourcetax/storage.py` — SQLite persistence
- `src/sourcetax/exporter.py` — QuickBooks + Schedule C generation
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

## Phase 1 — Complete ✅

**Status:** Canonical schema, ingestion, storage, and basic export implemented.

- Ingests real-format data (QuickBooks, Toast, Plaid/bank, receipt text) into canonical records.
- Normalizes dates, amounts, merchant names.
- Persists records to SQLite.
- Exports to QuickBooks CSV and Schedule C totals.

## Phase 2 — Receipt Extraction

**Planned:** OCR + field extraction from receipt images and forms.

## Phase 3 — Categorization

**Planned:** Rule-based + ML-based category assignment (Schedule C taxonomy).

## Phase 4 — Advanced Exports

**Planned:** GL reconciliation, audit reports, TurboTax/IRS Form integration.

## Directory Structure

```
sourcetax/
├── src/sourcetax/           # Core packages
│   ├── app.py              # Entry point
│   ├── core.py             # Extraction primitives
│   ├── ingest.py           # Ingestion pipeline
│   ├── schema.py           # Canonical record schema
│   ├── storage.py          # SQLite helpers
│   ├── exporter.py         # QuickBooks/Schedule C export
│   └── taxonomy.py         # Taxonomy and merchant mapping
├── data/
│   ├── samples/            # Sample data (recap QuickBooks, Toast, Plaid, receipts)
│   ├── forms/funsd/        # FUNSD dataset (forms + images)
│   ├── combined_dataset/   # combined_samples.jsonl (unified canonical JSONL)
│   ├── taxonomy/           # Schedule C taxonomy JSON
│   ├── mappings/           # Merchant category mapping CSV
│   └── store.db            # SQLite database (canonical_records table)
├── outputs/                # Generated exports (QuickBooks CSV, Schedule C)
├── tools/                  # Utility scripts (fetch, convert, generate reports)
├── docs/                   # Documentation (Phase 0 summary)
├── tests/                  # Test suite
├── pyproject.toml          # Package metadata and dependencies
└── README.md               # This file
```

## Contributing

Next phases (2–6) will add OCR, ML categorization, advanced exports, UI, and productization.

## License

See LICENSE.
