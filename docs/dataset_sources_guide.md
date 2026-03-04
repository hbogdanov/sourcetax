# SourceTax Dataset Sources Guide

This is a practical reference of external datasets that are useful (or not useful) for SourceTax workflows.

## Combined Useful Datasets (Kaggle + Hugging Face)

### Transaction Text / Merchant Pattern Datasets

1. Mitul Shah - Transaction Categorization

- Source: Hugging Face
- Link: https://huggingface.co/datasets/mitulshah/transaction-categorization
- Size: ~4.5M rows of transaction descriptions with consumer-style category labels
- Best Use: Vocabulary/embedding pretraining, text pattern learning
- Not Ideal: Labels do not match SourceTax tax taxonomy directly

2. BusinessTransactions

- Source: Hugging Face
- Link: https://huggingface.co/datasets/utkarshugale/BusinessTransactions
- Size: ~17.5K business transaction strings with Foursquare business categories
- Best Use: Merchant name/category extraction, normalization features
- Not Ideal: Category scheme does not align with tax categories directly

3. Credit Card Transactions Dataset (Kaggle)

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset
- Contents: Credit card transaction logs with merchant/amount/time
- Best Use: Merchant vocab expansion, clustering, unsupervised text learning

4. Financial Transactions Dataset (Kaggle - Fraud)

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
- Contents: Bank/credit transactions labeled for fraud
- Best Use: Feature engineering practice, robust text handling

5. USA Banking Transactions (Kaggle)

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/pradeepkumar2424/usa-banking-transactions-dataset-2023-2024
- Size: ~5K simulated banking transactions
- Best Use: Parser/CLI testing, sanity checks

### Supplemental / Related Datasets

6. Paysim1 Synthetic Transactions

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/ealaxi/paysim1
- Contents: Synthetic large transaction logs (numeric-heavy)
- Best Use: Stress testing ingestion, edge case handling
- Not Ideal: Weak for text-centric categorization

7. Financial Phrasebank (Kaggle)

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- Contents: Financial news sentiment dataset
- Best Use: NLP feature learning (context encoding)
- Not Ideal: Not transaction text

8. Financial NER Dataset (Hugging Face)

- Source: Hugging Face
- Link: https://huggingface.co/datasets/Josephgflowers/Financial-NER-NLP
- Contents: Named entity recognition over financial text
- Best Use: Extraction modules (merchant/entities from description)
- Not Ideal: Not direct taxonomy labeling

9. Financial QA (Hugging Face)

- Source: Hugging Face
- Link: https://huggingface.co/datasets/virattt/financial-qa-10K
- Contents: QA pairs about finance
- Best Use: Financial semantics pretraining (indirect utility)
- Not Ideal: Not transaction categorization data

## Non-Useful for SourceTax Core Categorization

These may be finance-related but are not useful for SourceTax's transaction taxonomy task:

- Loan default prediction datasets
- Credit risk scoring datasets
- Time-series price forecasting datasets
- Insurance claim records with no transaction text
- Survey datasets
- Portfolio returns and ETF history datasets

## Quick Mapping: What Each Type Helps With

| Dataset Type | Best Use | Not Useful For |
|---|---|---|
| Mitul Shah TF dataset | Vocabulary + semantic pretraining | Direct tax labels |
| BusinessTransactions | Merchant normalization | Direct tax labels |
| Kaggle credit card logs | Transaction text corpus | Direct labeled taxonomy |
| Fraud transaction logs | Edge-case NLP/features + anomaly tests | Tax label learning |
| Paysim1 | Stress tests | Tax labeling |
| Financial NER | Entity extraction training | Core taxonomy supervision |
| Financial QA | General finance NLP pretraining | Transaction categorization |
