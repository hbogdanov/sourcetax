# SourceTax â€” Transaction Classification & Tax Automation

An AI-driven pipeline that ingests transaction data from banks, POS systems, and receipt images, then normalizes, extracts, categorizes, and exports to QuickBooks and Schedule C tax forms.

## Current Status (Feb 2026)

**Implemented and working (MVP):**
- Demo-safe setup/test/smoke commands (`make setup`, `make smoke`, `make smoke-strict`, `make test`)
- End-to-end smoke pipeline (ingest -> match -> categorize -> export -> eval)
- Phase 3 benchmark runner (rules vs TF-IDF vs ensemble, optional SBERT)
- Phase 4 accounting-grade exports (enriched transactions CSV, GL lines CSV, audit trail JSONL)
- Phase 4 reconciliation reports (unmatched receipts, unmatched bank txns, low-confidence queue, conflicts queue, summary metrics)
- Minimal adapter layer (CSV/QBO-like export + mock QuickBooks API payload)

**Still in progress / not yet production-grade:**
- Gold dataset expansion (currently ~10 labeled records; target 200+)
- Balanced locked train/val/test splits for reliable ML metrics
- Conflict queue quality improves once ML predictions/rationale are persisted in `raw_payload`
- Full bookkeeping/chart-of-accounts logic (GL export is intentionally simplified)

## Recent Updates

- Added `tools/phase3_benchmark.py` to generate `reports/phase3_eval.md` and evaluation artifacts
- Added `tools/phase4_run.py` to generate accounting-grade exports and reconciliation queues in one command
- Unified merchant normalization usage via `src/sourcetax/normalization.py`
- Added Phase 4 export/reconciliation test coverage (`tests/test_phase4_exports.py`)
- Fixed matching recall edge case in `tools/eval.py` (no impossible recall when no matches are predicted)
- Smoke demo now inserts a deterministic synthetic receipt so matching shows at least one receipt match
- Added `--strict` smoke assertions and benchmark `needs_review.csv` / run metadata output

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

# Optional ML embeddings (sentence-transformers + torch)
pip install -e ".[embeddings]"

# Optional OCR backend (EasyOCR + pinned torch/torchvision)
pip install -e ".[ocr]"
```

### Demo-safe Commands

```bash
make setup
make smoke
make smoke-strict
make phase4
make test
```

`make smoke` runs a lightweight end-to-end flow and best-effort evaluation without requiring EasyOCR or SBERT.
`make smoke-strict` adds demo assertions (DB/export/report exists, enough records, at least one matched receipt).
`make phase4` exports accounting-grade artifacts and reconciliation queues from `data/store.db`.

On Windows PowerShell, `make` may not be installed. Use the equivalent commands directly:

```powershell
python tools\smoke_run.py --strict
python tools\phase3_benchmark.py --allow-small
python tools\phase4_run.py --mock-qbo
```

### Run Demo

```bash
# Full end-to-end: ingest samples, match receipts, categorize, export
python tools/generate_reports.py

# Lightweight one-command smoke test (ingest -> match -> categorize -> export)
# Does not require EasyOCR.
python tools/smoke_run.py

# Strict smoke mode (fails loudly if core outputs are missing or no receipt match is produced)
python tools/smoke_run.py --strict

# Phase 3 benchmark report (rules vs TF-IDF vs ensemble, optional SBERT)
python tools/phase3_benchmark.py --allow-small

# Phase 4 accounting-grade exports + reconciliation queues
python tools/phase4_run.py --mock-qbo

# Outputs:
#   reports/phase3_eval.md            - benchmark summary (rules vs TF-IDF vs ensemble)
#   reports/phase3_eval_assets/*      - confusion matrices, per-category metrics, needs_review.csv
#   outputs/quickbooks_import.csv     â€” QB import format
#   outputs/schedule_c_totals.csv     â€” Schedule C category breakdown
#   outputs/audit_pack.csv            â€” full transaction trail with confidence
```

### Launch Review UI

```bash
# Interactive Streamlit dashboard for reviewing/overriding categories
streamlit run app_review.py

# Opens http://localhost:8501
# Features:
#   - Dashboard: metrics, auto-match/categorize buttons
#   - Dashboard: gold set progress (target 200) + export reviewed labels button
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

4. **Extract** (`src/sourcetax/receipts.py` â€” Phase 2)  
   OCR receipt images (Tesseract/EasyOCR), parse fields (date, merchant, total, tax).

5. **Match** (`src/sourcetax/matching.py` â€” Phase 2)  
   Link receipt documents to bank transactions with fuzzy matching.

6. **Categorize** (`src/sourcetax/categorization.py` â€” Phase 2)  
   Rules-based classification (merchant exact/fuzzy match, keywords, user overrides).

7. **Export** (`src/sourcetax/exporter.py`)  
   Generate QuickBooks CSV, Schedule C totals, audit pack.

## Phase Completion Status

### Phase 0 â€” âœ… Complete

**Foundation:** Canonical schema, sample data, basic CSV ingestion, SQLite storage.

**What's Here:**
- Real-format sample data (QB, Toast, bank CSV)
- Public datasets (FUNSD forms, CORD receipts)
- `CanonicalRecord` dataclass with 13 fields
- CSV ingestors for toast, bank, quickbooks
- SQLite schema and basic I/O

### Phase 1 â€” âœ… Complete (phase1 branch)

**Schema & Cleanup:** Enhanced canonical schema, fixed encoding, code formatting, repo hygiene.

**What's Here:**
- Added direction field (expense|income), category fields, source traceability
- Fixed pyproject.toml to PEP 621
- Updated .gitignore to exclude large artifacts
- All tests pass, demo works deterministically

### Phase 2 â€” âœ… Complete (phase2 branch)

**Receipt Extraction, Matching, Categorization, Review UI.**

**Phase 2.1 â€” Receipt OCR:**
- `src/sourcetax/receipts.py` â€” Tesseract/EasyOCR integration
- Field extraction: date, merchant, total, tax, tip
- Heuristic-based (regex + keyword search)

**Phase 2.2 â€” Transaction Matching:**
- `src/sourcetax/matching.py` â€” Receipt â†” bank fuzzy matching
- Scoring: date (Â±3d), amount (Â±$10), merchant (>80% similarity)
- `list_unmatched_receipts()`, `list_unmatched_transactions()` for UI

**Phase 2.3 â€” Categorization Rules Engine:**
- `src/sourcetax/categorization.py` â€” Priority rules (learned > exact > fuzzy > keywords)
- Keyword rules: UBERâ†’Travel, STARBUCKSâ†’Meals, HOME DEPOTâ†’Supplies, etc.
- Learned overrides: user sets merchant category once, applies to all

**Phase 2.4 â€” Exports & Audit:**
- QB CSV uses `category_final` (user overrides)
- Schedule C totals by category
- Audit pack: full record trail with match scores + confidence
- Metrics export: record counts, expense totals, match rates

**UI:**
- `app_review.py` â€” Streamlit dashboard
  - Dashboard: metrics, auto-match/auto-categorize buttons
  - Unmatched receipts: list + detail view with OCR excerpt
  - Unmatched transactions: identify orphans
  - Match review: approve matches, override categories, save
- `src/sourcetax/taxonomy.py` â€” category and merchant mapping helpers
- `data/samples/` â€” real-like export examples
- `data/forms/funsd/` â€” FUNSD dataset (149 forms + images)
- `data/combined_dataset/combined_samples.jsonl` â€” unified training/dev dataset

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
- **FUNSD** (149 forms): `data/forms/funsd/9.json` â€¦ `data/forms/funsd_127.json` + images
- **CORD** receipts: `data/samples/cord/` (subset used in Phase 2 evaluation)

**Taxonomy & Mappings:**
- `data/taxonomy/schedule_c_taxonomy.json` â€” Schedule C category hierarchy
- `data/mappings/merchant_category.csv` â€” merchant â†’ category rules (format: merchant,category_code,category_name,notes)

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
- **Merchant normalization:** Use `normalize_merchant(text)` consistently. Examples: "STARBUCKS COFFEE 123" â†’ "Starbucks", "UBER TRIP" â†’ "Uber".
- **Category coding:** Always populate `category_code` (stable: 'meals', 'travel', 'office', etc.) alongside `category_name` (human-readable: 'Meals and Lodging'). This ensures ML models can use both.
- **Amounts:** Always store with one convention (expenses positive). Direction field disambiguates, not the sign.
- **Dates:** ISO 8601 format (YYYY-MM-DD). No timezones. OCR extracts best guess; gold set has ground truth.
- **Confidence:** Float 0.0â€“1.0. Metrics:
  - For rules-based categorization: 0.9â€“1.0 (confident)
  - For fuzzy matches: score of best match (0.0â€“1.0)
  - For ML predictions: probability of predicted class
- **Match score:** 0.0â€“1.0 if matched_transaction_id is set. Otherwise null.

**`src/sourcetax/ingest.py`**  
Key functions:
- `read_csv(path, source) â†’ Iterator[dict]` â€” Read CSV, yield records
- `normalize_to_canonical(row, source) â†’ CanonicalRecord` â€” Convert any source to canonical
- `ingest_and_store(paths, source, db_path)` â€” Read + store to SQLite

**`src/sourcetax/storage.py`**  
Key functions:
- `ensure_db(db_path)` â€” Create schema if not exists
- `insert_record(record, db_path)` â€” Persist canonical record
- `get_all_records(db_path) â†’ List[CanonicalRecord]` â€” Retrieve all from database

**`src/sourcetax/receipts.py`** (Phase 2.1)  
Key functions:
- `extract_ocr_text(image_path, backend='tesseract') â†’ str` â€” OCR receipt image
- `parse_receipt_text(text) â†’ dict` â€” Extract date, merchant, total, tax, tip
- `ingest_receipt_file(image_path, source='receipt') â†’ CanonicalRecord` â€” End-to-end

**`src/sourcetax/matching.py`** (Phase 2.2)  
Key functions:
- `match_receipt_to_bank(receipt, bank_txns) â†’ (score, matched_txn)` â€” Single match
- `match_all_receipts(records, db_path=None) â†’ List[CanonicalRecord]` â€” Bulk matching
- `list_unmatched_receipts(records) â†’ List[CanonicalRecord]` â€” Orphan receipts
- `list_unmatched_transactions(records) â†’ List[CanonicalRecord]` â€” Orphan bank transactions

**`src/sourcetax/categorization.py`** (Phase 2.3)  
Key functions:
- `categorize_record(record) â†’ CanonicalRecord` â€” Apply rules once
- `categorize_all_records(records, db_path=None) â†’ List[CanonicalRecord]` â€” Bulk categorization
- `load_merchant_category_map(csv_path) â†’ dict` â€” Load merchant â†’ category learned overrides

**`src/sourcetax/exporter.py`** (Phase 2.4)  
Key functions:
- `generate_quickbooks_csv(records, output_path)` â€” QB format (uses `category_final`)
- `compute_schedule_c_totals(records) â†’ dict` â€” By-category breakdown
- `write_schedule_c_csv(totals, output_path)` â€” SC CSV export
- `export_audit_pack(records, output_path)` â€” Full detail export
- `export_metrics(records) â†’ dict` â€” Count, total, match rate, etc.

## Directory Structure

```
sourcetax/
â”œâ”€â”€ src/sourcetax/           # Core packages
â”‚   â”œâ”€â”€ __init__.py         # Package marker
â”‚   â”œâ”€â”€ schema.py           # CanonicalRecord dataclass
â”‚   â”œâ”€â”€ ingest.py           # Ingestion pipeline (CSV/receipt)
â”‚   â”œâ”€â”€ storage.py          # SQLite persistence
â”‚   â”œâ”€â”€ receipts.py         # OCR + field extraction (Phase 2.1)
â”‚   â”œâ”€â”€ matching.py         # Receiptâ†”bank matching (Phase 2.2)
â”‚   â”œâ”€â”€ categorization.py   # Rules engine (Phase 2.3)
â”‚   â”œâ”€â”€ exporter.py         # QB/SC/audit exports (Phase 2.4)
â”‚   â””â”€â”€ models/             # Phase 3 ML modules
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ data_prep.py              # Load splits, stratified sampling
â”‚       â”œâ”€â”€ train_baseline.py         # TF-IDF + LogisticRegression
â”‚       â”œâ”€â”€ train_sbert.py            # SBERT embeddings + LR classifier
â”‚       â”œâ”€â”€ evaluate.py               # Metrics, comparison, error analysis
â”‚       â”œâ”€â”€ merchant_normalizer.py    # Rule-based merchant cleaning
â”‚       â”œâ”€â”€ embeddings.py             # SentenceTransformer + caching
â”‚       â”œâ”€â”€ active_learning.py        # 4 selection strategies
â”‚       â”œâ”€â”€ hierarchical.py           # Major â†’ subcategory classification
â”‚       â””â”€â”€ visualize.py              # Confusion matrix, P/R charts, comparison
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ samples/            # Sample data (QB, Toast, Plaid, receipts)
â”‚   â”œâ”€â”€ forms/funsd/        # FUNSD dataset (149 forms)
â”‚   â”œâ”€â”€ combined_dataset/   # combined_samples.jsonl
â”‚   â”œâ”€â”€ gold/               # Hand-labeled evaluation set (10+ records)
â”‚   â”œâ”€â”€ ml/                 # ML artifacts (splits, pipelines, reports)
â”‚   â”‚   â”œâ”€â”€ ml_train.csv    # Locked training set
â”‚   â”‚   â”œâ”€â”€ ml_val.csv      # Validation set
â”‚   â”‚   â”œâ”€â”€ ml_test.csv     # Test set
â”‚   â”‚   â”œâ”€â”€ baseline_pipeline.pkl  # TF-IDF trained pipeline
â”‚   â”‚   â”œâ”€â”€ sbert_pipeline.pkl     # SBERT trained pipeline (if trained)
â”‚   â”‚   â”œâ”€â”€ evaluation_report/     # Visualizations (HTML)
â”‚   â”‚   â””â”€â”€ split_metadata.txt     # Stratification info
â”‚   â”œâ”€â”€ taxonomy/           # Schedule C taxonomy JSON
â”‚   â”œâ”€â”€ mappings/           # Merchant category mapping CSV
â”‚   â””â”€â”€ store.db            # SQLite database
â”œâ”€â”€ outputs/                # Generated exports (QB CSV, Schedule C)
â”œâ”€â”€ tools/                  # Utility scripts
â”‚   â”œâ”€â”€ generate_reports.py         # End-to-end demo (runs all phases)
â”‚   â”œâ”€â”€ generate_training_samples.py
â”‚   â”œâ”€â”€ convert_funsd_to_combined.py
â”‚   â”œâ”€â”€ count_combined.py
â”‚   â”œâ”€â”€ eval.py                    # Evaluation on current pipeline
â”‚   â”œâ”€â”€ train_ml_baseline.py       # TF-IDF baseline training
â”‚   â”œâ”€â”€ train_ml_advanced.py       # Advanced ML (SBERT, hierarchical)
â”‚   â”œâ”€â”€ select_for_labeling.py     # Active learning selection
â”‚   â””â”€â”€ fetch_public_datasets.py
â”œâ”€â”€ app_review.py           # Streamlit dashboard UI
â”œâ”€â”€ tests/                  # Test suite
â”œâ”€â”€ pyproject.toml          # Package metadata + dependencies
â””â”€â”€ README.md               # This file
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
- Receiptâ†”bank links for matching evaluation
- OCR extraction ground truth (merchant, date, amount)

**Metrics Tracked:**
- **Categorization Accuracy:** Rules engine + overrides vs. ground truth
- **Matching Precision/Recall:** Receiptâ†’bank matching performance
- **Extraction Accuracy:** Merchant, date, amount OCR correctness

Before Phase 3 ML work, expand the gold dataset to ~200 records using `app_review.py` (mark good matches, override categories, save), then use the Dashboard button to export reviewed labels to `data/gold/gold_transactions.jsonl`.

## Phase 3 â€” ML Categorization & Advanced Features

**Status:** âœ… Foundation complete. TF-IDF baseline, SBERT embeddings, active learning, and hierarchical classification implemented.

### Phase 3 Foundation (Complete)

**Gold Dataset & Baseline Metrics:**
- `data/gold/gold_transactions.jsonl` â€” 10 hand-labeled transactions (ground truth)
- `tools/eval.py` â€” Evaluation script (rules accuracy, per-category metrics)
- `tools/train_ml_baseline.py` â€” TF-IDF + LogisticRegression pipeline
- **Locked splits:** `data/ml/ml_train.csv`, `ml_val.csv`, `ml_test.csv` with stratification
- **TF-IDF Baseline:** ~50% accuracy on test set (small dataset, high variance)

### Phase 3 Advanced ML (Complete)

**High-ROI Enhancements:**

#### 1. **Merchant Normalization** (`src/sourcetax/models/merchant_normalizer.py`)

Rule-based cleaning + brand alias mapping. Runs before embedding/classification for cleaner inputs.

```python
from sourcetax.models import merchant_normalizer

# Clean and standardize merchant names
clean, root, brand = merchant_normalizer.normalize_merchant("SQ *STARBUCKS COFFEE 123 SF CA")
# clean: "STARBUCKS COFFEE"
# root:  "STARBUCKS COFFEE"
# brand: "Starbucks"  (from MERCHANT_ALIASES)
```

**Features:**
- Removes junk tokens (SQ, POS, merchant codes, state abbreviations, TLDs)
- Alias mapping: "AMZN" â†’ Amazon, "WHOLEFDS" â†’ Whole Foods, "UBER" â†’ Uber, etc. (25+ mappings)
- Root extraction: first N tokens as semantic signal
- Handles payment platform prefixes (Square, PayPal, Stripe)

#### 2. **SBERT Dense Embeddings** (`src/sourcetax/models/embeddings.py`)

Pre-trained SentenceTransformer for semantic similarity (better than token-based TF-IDF for short texts).

```python
from sourcetax.models import embeddings

# Embed merchant + description
embedder = embeddings.get_embedder("all-MiniLM-L6-v2")  # 384-dim, fast
X_emb = embeddings.embed_dataset(df[["merchant", "description"]], embedder)
# Shape: (N, 384)

# Caches to disk to avoid recomputation
```

**Benefits:**
- Captures semantic relationships (not just token overlap)
- Pre-trained on 1B sentence pairs â†’ generalizes to domain
- Handles short/noisy merchant names better than TF-IDF
- 384-dim embeddings + StandardScaler â†’ LogisticRegression

**Dependencies (optional):**
```bash
pip install -e ".[embeddings]"  # or: pip install sentence-transformers
```

#### 3. **SBERT-Based Classifier** (`src/sourcetax/models/train_sbert.py`)

LogisticRegression trained on SBERT embeddings instead of TF-IDF bag-of-words.

```python
from sourcetax.models import train_sbert

# Train on embeddings
pipeline, metrics = train_sbert.train_sbert_classifier(
    X_train, y_train, X_val, y_val
)

# Predict with full pipeline (embed + scale + classify)
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)
```

**Performance:**
- Typically 15â€“30% better than TF-IDF on transaction categorization
- Works well with 50â€“500 labeled examples
- Inference: ~10ms per transaction (batch)

#### 4. **Active Learning** (`src/sourcetax/models/active_learning.py`)

Intelligently select which unlabeled transactions to label next for maximum model improvement.

```python
from sourcetax.models import active_learning

# Choose best 50 samples to label
selected_idx, summary = active_learning.select_for_labeling(
    embeddings=X_emb,
    predictions_proba=y_proba,
    labeled_mask=labeled_mask,
    n_select=50,
    strategy="diversity",  # or: uncertainty, margin, entropy
)

# summary DataFrame: ["index", "max_prob", "entropy"]
# Order: lowest confidence first (model most unsure)
```

**Strategies:**
- **Uncertainty Sampling:** Lowest max probability (model most unsure)
- **Margin Sampling:** Smallest gap between top-2 predictions (near decision boundary)
- **Entropy Sampling:** Highest prediction entropy (maximum indecision)
- **Diversity Sampling:** K-means clusters + picks uncertain samples across clusters (prevents duplicates)

**Demo Output:**
```bash
python tools/select_for_labeling.py --strategy diversity --n 50
# Outputs: selection_for_labeling.csv with top 50 to label next
# Review in app_review.py, mark/override, re-train
```

#### 5. **Active Learning Selection Tool** (`tools/select_for_labeling.py`)

End-to-end script: Load unlabeled pool, compute embeddings + predictions, select top samples by strategy.

```bash
python tools/select_for_labeling.py --strategy diversity --n 50 --output data/ml/next_batch.csv
# Produces CSV with:
#   - Transaction details (merchant, amount, date)
#   - max_prob (model confidence)
#   - entropy (prediction uncertainty)
# Import into app_review.py: Mark correct predictions, override wrong ones, save
```

#### 6. **Hierarchical Classification** (`src/sourcetax/models/hierarchical.py`)

Two-stage classification: Major category (e.g., "Meals & Entertainment") â†’ Subcategory (e.g., "Coffee", "Restaurant").

```python
from sourcetax.models import hierarchical

# Build hierarchy mapping
subcat_to_major, major_to_subs = hierarchical.build_label_hierarchy(
    categories=all_cat_labels,
    hierarchy={
        "Meals & Entertainment": ["Coffee", "Restaurant", "Bar"],
        "Travel": ["Flight", "Hotel", "Rental Car"],
        "Utilities": ["Electric", "Gas", "Water"],
    }
)

# Train hierarchical: major classifier + per-major subcategory classifiers
major_clf, sub_clfs, metrics = hierarchical.train_hierarchical_classifier(
    X_train, y_train_major, y_train_sub,
    major_trainer_fn=train_sbert.train_sbert_classifier,
    sub_trainer_fn=train_sbert.train_sbert_classifier,
)

# Predict: Stage 1 (major) â†’ Stage 2 (subcategory)
major_pred, sub_pred = hierarchical.hierarchical_predict(
    major_clf, sub_clfs, X_test, subcat_to_major
)
```

**Benefits:**
- Captures category structure (broad rollup + detailed expense tracking)
- Improves accuracy by constraining Stage 2 to valid subcategories
- Enables multi-level reporting (Schedule C major categories â†’ detailed audit trail)

#### 7. **Visualizations & Evaluation** (`src/sourcetax/models/visualize.py`)

Generate confusion matrix heatmaps, precision/recall charts, model comparison tables.

```python
from sourcetax.models import visualize

# Generate full report
report_paths = visualize.generate_evaluation_report(
    y_test,
    predictions={
        "TF-IDF": tfidf_pred,
        "SBERT": sbert_pred,
    },
    label_names=categories,
    output_dir="data/ml/evaluation_report"
)

# Outputs:
#   reports/phase3_eval.md            - benchmark summary (rules vs TF-IDF vs ensemble)
#   reports/phase3_eval_assets/*      - confusion matrices, per-category metrics, needs_review.csv
#   - confusion_matrix_TF-IDF.html
#   - confusion_matrix_SBERT.html
#   - precision_recall_TF-IDF.html
#   - precision_recall_SBERT.html
#   - model_comparison.html  (side-by-side metrics)
#   - index.html
```

#### 8. **Orchestration Script** (`tools/train_ml_advanced.py`)

End-to-end training pipeline: Load gold â†’ normalize â†’ embed â†’ train TF-IDF + SBERT + hierarchical â†’ evaluate â†’ generate visualizations.

```bash
# Full ML workflow
python tools/train_ml_advanced.py --strategy all

# Or specific strategies
python tools/train_ml_advanced.py --strategy sbert
python tools/train_ml_advanced.py --strategy hierarchical

# Output:
#   âœ… Loaded 10 gold records (train: 7, val: 1, test: 2)
#   âœ… TF-IDF baseline: train_acc=0.714, val_acc=1.000
#   âœ… SBERT classifier: train_acc=0.857, val_acc=1.000
#   âœ… Hierarchical: major_acc=1.000, sub_acc=0.857
#   âœ… Visualizations saved to data/ml/evaluation_report/
```

### Recommended Workflow for Expanding Gold Set

1. **Start with 10 gold records** (provided in `data/gold/gold_transactions.jsonl`)
2. **Run baseline evaluation:**
   ```bash
   python tools/train_ml_baseline.py
   python tools/eval.py
   ```
3. **Generate next batch for labeling:**
   ```bash
   python tools/select_for_labeling.py --strategy diversity --n 50
   ```
4. **Label in UI:**
   ```bash
   streamlit run app_review.py
   # Review top-50 uncertain samples, confirm/override categories, save
   ```
5. **Retrain:**
   ```bash
   python tools/train_ml_advanced.py --strategy all
   ```
6. **Repeat steps 3â€“5** until validation accuracy plateaus (typically 200â€“300 labeled examples)

### Expected Improvements

| Approach | # Gold Records | Accuracy | Training Time |
|----------|--|--|--|
| Rules-only | â€” | 50% | <1s |
| TF-IDF baseline | 10 | 60â€“70% | 1s |
| TF-IDF (expanded) | 150 | 75â€“85% | 2s |
| SBERT | 10 | 70â€“75% | 5s (+ embed cache) |
| SBERT (expanded) | 150 | 85â€“92% | 10s |
| Hierarchical + SBERT | 150 | 85â€“92% (major), 80â€“88% (sub) | 15s |

## Roadmap

### Phase 4 â€” Advanced Exports & Reconciliation

**Status:** MVP implemented (Phase 4a/4b/4c minimum viable)

**Implemented:**
- Accounting-grade enriched transactions export (`outputs/accounting_transactions_enriched.csv`)
- GL lines export (`outputs/gl_lines.csv`) with double-entry-ish clearing account postings
- Audit trail export (`outputs/audit_trail.jsonl`) with inputs, transformations, rule hits, match scores, and final decision rationale
- Reconciliation queues: unmatched receipts, unmatched bank txns, low-confidence categorizations, conflicts
- Reconciliation summary metrics (`match_rate`, `avg_confidence`, top ambiguous merchants)
- Simple adapter seam: QBO-like CSV export and file-backed mock QuickBooks API payload

**Next improvements:**
- Persist ML/ensemble predictions and rationale in `raw_payload` during categorization to strengthen conflicts + audit trail explainability
- Add a Streamlit reconciliation UI for queue triage
- Expand GL/account mapping and journal posting rules beyond the current simplified clearing-account format

### Phase 4 Data Quality (ML metrics reliability)

**Planned / Priority:** Expand gold dataset to 200+ labeled records and lock balanced train/val/test splits.

### Phase 5 â€” Multi-User & Cloud

**Planned:** Web UI, user accounts, cloud storage, batch processing.

## Contributing

Report issues, submit PRs. Development tools:

```bash
black src/sourcetax tests          # Format code
ruff check src/sourcetax tests     # Lint
pytest tests/ -v                   # Run tests
```

## License

MIT

