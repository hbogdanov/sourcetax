Sample data sources for SourceTax

This file lists representative sample formats and public datasets to use for ingestion, extraction, and training.

1) POS / Toast-like exports
- Toast provides several export types (Orders/Payments/Accounting/Menu). For MVP focus on Sales/Payment exports (daily settlements, order lines, payment detail). If you can get real Toast CSV/Excel files from a restaurant account, use those.
- Representative schema (CSV columns): order_id, date, location, item_id, quantity, price, total, payment_type, tip, tax, refund, void

2) Bank transaction CSVs (common templates)
- Most banks export CSV with columns like: Date, Description, Amount, Transaction Type, Balance.
- Plaid sandbox provides sample transaction payloads for testing programmatically: https://plaid.com/docs/sandbox/
- QuickBooks bank import notes: https://quickbooks.intuit.com/learn-support/en-us/help-article/import-export-data/import-bank-transactions-quickbooks-online/L4MSA0eU9_US_en_US

3) Receipt image / OCR datasets
- CORD (Korean/English multi-language receipt dataset, includes OCR annotations): https://github.com/clovaai/cord
- SROIE (ICDAR 2019) — Scanned Receipt OCR and Information Extraction: https://rrc.cvc.uab.es/
- Kaggle has several receipt OCR datasets (search "receipt OCR"), useful for training extraction models when allowed by license.

4) Accounting / GL exports
- Toast AccountingReport.xls or similar contains GL-coded rows (date, location, GL account, description, amount) — very useful if the merchant configured GL mapping.

5) Public transaction / sample datasets to seed models
- Plaid sandbox transactions (programmable; representative bank payloads): https://plaid.com/docs/sandbox/
- Example POS CSVs and sample receipts on Kaggle (search "POS" or "receipts")
- Use SROIE / CORD for receipt OCR and key-value extraction model training.

Practical notes
- For MVP, prioritize: daily revenue totals, payment breakdown, taxes collected, refunds/voids, tips.
- Start with synthetic or small real exports you control; normalize to canonical schema and add confidence scores.

Canonical schema (for normalization)
- id: unique id
- merchant_name
- transaction_date (ISO 8601)
- amount (positive for income, negative for refund/expense as appropriate)
- currency
- payment_method
- source (toast/bank/receipt)
- raw_payload (original row or OCR text)
- confidence: {merchant: float, date: float, amount: float}
- tags: list

References and links
- CORD: https://github.com/clovaai/cord
- SROIE / ICDAR: https://rrc.cvc.uab.es/
- Plaid sandbox docs: https://plaid.com/docs/sandbox/
- QuickBooks bank import guidance: https://quickbooks.intuit.com/learn-support/en-us/help-article/import-export-data/import-bank-transactions-quickbooks-online/L4MSA0eU9_US_en_US


"How to get sample data quickly"
- Ask a friendly local restaurant for a few Toast export files (Orders/Payments) and a sample AccountingReport.xls
- Export 1-3 months of bank CSV from a test business checking account (scrub PII if sharing)
- Collect 20–50 receipt images (JPEG/PNG/PDF) from everyday purchases for OCR training/evaluation

License caution
- Be mindful of PII and bank data sensitivity. Use synthetic data or redacted real exports for developing and testing.
