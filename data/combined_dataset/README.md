Combined Samples
================

This folder contains a canonical JSONL file `combined_samples.jsonl` used for development and ML experiments.

Each line is a JSON object with a minimal canonical shape combining receipts, POS, bank, and form samples. Common keys include:

- `source`: one of `quickbooks`, `toast`, `bank`, `receipt`, `cord_sample`, `funsd`, etc.
- `merchant_name`, `transaction_date`, `amount` when available
- `raw` or `raw_text` containing original payload or OCR text
- `type`: logical type such as `accounting`, `pos`, `bank`, `receipt`, `form`

Purpose
-------

This file is intended as a small, realistic development corpus to:

- Exercise ingestion/normalization code paths.
- Serve as a starting training/test set for extraction and categorization models.

If you want larger samples, run `tools/fetch_public_datasets.py` to pull FUNSD (done), and optionally fetch CORD/SROIE when available or upload Kaggle credentials for retail receipts.
