# Data Layout

SourceTax uses tiered data directories:

- `data/raw/`: raw ingested files (not in git)
- `data/external/`: fetched third-party corpora/dumps (not in git)
- `data/interim/`: generated staging/canonical/intermediate datasets (not in git)
- `data/gold/`: curated human-labeled gold data (in git)
- `data/taxonomy/`: taxonomy contracts (in git)
- `data/mappings/`: deterministic mapping tables (in git)
- `data/samples/`: tiny demo samples only (in git)

Path conventions:

- Canonical app DB default: `data/interim/store.db`
- Staging DB default: `data/interim/staging.db`
- Reports/eval outputs default: `artifacts/reports/`
- Export outputs default: `artifacts/exports/`
