# SourceTax Product Strategy and Roadmap Notes

This document captures the current strategic direction, architecture guidance, and execution priorities.

## 1) What Should Stay the Same

Do not overreact by replacing core architecture. The current structure is close to real fintech categorization systems.

Keep this pipeline:

transactions  
-> merchant normalization  
-> rule engine  
-> ML models  
-> confidence threshold  
-> human review

Keep the hybrid strategy:

- Rules: high-precision obvious merchants
- TF-IDF: cheap baseline classifier
- SBERT: semantic similarity for messy text
- Hybrid scoring: fallback and routing logic

Do not replace this with "just LLMs" for core categorization due to cost and reliability concerns.

## 2) Biggest Shift: Stop Searching for Perfect Public Datasets

There is no public dataset with:

- merchant text
- transaction description
- true tax category labels

for real SMB financial records at product quality.

The strategy is now to build and grow a data pipeline, not "find a magic dataset."

## 3) Build a Merchant Knowledge Base (Highest ROI)

Treat merchant intelligence as a first-class layer.

Example:

| merchant_normalized | category |
|---|---|
| home depot | Repairs & Maintenance |
| lowes | Repairs & Maintenance |
| google ads | Advertising |
| uber | Travel |
| delta airlines | Travel |

Most production systems get large gains from merchant->category priors before ML.

## 4) Expand Merchant Normalization

Double down on `merchant_normalizer.py`.

Examples that should normalize to one canonical merchant:

- HOME DEPOT #2241
- HOMEDEPOT.COM
- HD SUPPLY
- HOME DEPOT ATLANTA

Canonical target: `home depot`

Better normalization materially increases downstream categorization quality.

## 5) Generate Synthetic Training Data

Use synthetic merchant-string variants for pretraining/bootstrapping.

Start with canonical merchants, e.g.:

- home depot
- lowes
- google ads
- meta ads
- uber
- delta

Generate variants:

- HOME DEPOT #2241 ATLANTA
- LOWES #4491
- GOOGLE ADS *1234
- UBER TRIP HELP.UBER.COM

Autolabel by canonical merchant mapping and use as supplemental training data.

## 6) Grow Gold Dataset to ~1000 Rows

Current ~589 rows is a good start.  
Target next: 1000-2000 labeled transactions with balanced category coverage.

Do not optimize for volume only; optimize for coverage of critical categories.

## 7) Add Confidence Threshold Routing

Never auto-accept every model prediction.

Recommended policy:

- >=0.90: auto-accept
- 0.70-0.90: suggestion/review
- <0.70: human review required

This is better aligned with accounting/audit expectations.

## 8) Build the Feedback Loop

Core loop:

prediction  
-> user correction  
-> store correction  
-> append to training set  
-> periodic retraining

This loop is the compounding advantage over time.

## 9) Keep Taxonomy Mapped to Schedule C

Use Schedule C-aligned categories where possible:

- Advertising
- Car & Truck
- Contract Labor
- Insurance
- Legal & Professional Services
- Meals
- Office Expense
- Rent
- Repairs & Maintenance
- Supplies
- Travel
- Utilities

Tax-form alignment is a product advantage.

## 10) Product Positioning

Do not position as "QuickBooks replacement."

Position as:

AI tax categorization + deduction intelligence layer

Export destinations:

- QuickBooks
- CSV
- Schedule C-ready outputs

QuickBooks is a destination integration, not direct competition.

## 11) Frontend Focus for Demos

Priority demo UX:

uploaded transactions  
-> AI suggestions  
-> quick approve/edit grid  
-> export

Spreadsheet-like review UX is usually preferred by accountants.

## 12) Metrics to Track

Track more than accuracy:

- auto-categorization rate
- human correction rate
- unknown merchant rate
- confidence distribution

Example target profile:

- 60-70% auto-categorized
- ~30% review
- <5% wrong predictions on accepted items

## 13) Near-Term Roadmap (3-4 Weeks)

Week 1:

- build merchant knowledge base
- expand normalization rules
- generate synthetic merchant transactions

Week 2:

- grow gold dataset to ~1000
- retrain models
- evaluate hybrid pipeline

Week 3:

- implement confidence thresholding
- build review UI

Week 4:

- add correction feedback loop
- export to QuickBooks-style CSV

## 14) Current Strength

The layered architecture already matches production patterns:

- rules
- merchant normalization
- TF-IDF
- SBERT
- hybrid routing/classification

The main leverage now is data quality/coverage and UX workflow, not replacing algorithms.
