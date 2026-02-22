CORD dataset (receipt OCR + parsing)

Source: https://github.com/clovaai/cord
License: CC-BY-4.0
Notes:
- Contains images and JSON annotations for receipt parsing tasks (menu items, subtotal, tax, total, etc.)
- CORD v1/v2 sample is available via Hugging Face Datasets; includes a 1,000-sample subset suitable for MVP training/evaluation.
- Useful for training post-OCR parsing models (key-value extraction, line grouping).

Suggested use in SourceTax:
- Use CORD v1 1,000-sample subset to build an initial receipt key-value extractor.
- Map CORD labels (subtotal, tax, total, menu.item) to the canonical schema fields.
