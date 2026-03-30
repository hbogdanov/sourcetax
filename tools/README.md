# Tools Layout

Core scripts are grouped by use case:

- `tools/data_pipeline/` - ingestion, smoke runs, exports, product-style entrypoints
- `tools/training/` - model training workflows
- `tools/evaluation/` - benchmarking and comparison

Compatibility wrappers remain at the legacy `tools/*.py` paths so existing commands and tests keep working.
