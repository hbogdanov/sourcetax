# Data Outreach Summary

## Purpose

This note documents the project's outreach and real-data acquisition effort for Weeks 12-15.

## Outreach Attempted

The project reached out to local small businesses and accounting-oriented firms to request example transaction data or lightly redacted accounting exports that could support taxonomy validation and model training.

Target profiles included:
- coffee shops
- bakeries
- small restaurants
- Midtown-area small businesses
- accounting-oriented service firms working with small-business clients

## Outcome

Most conversations ended in rejection or non-participation because the requested data touched financial records, vendor relationships, payroll context, or customer-facing business operations.

The main reasons given were:
- privacy concerns
- compliance obligations
- accounting confidentiality
- data sensitivity around vendor and banking activity

## Sample Received

One redacted and unstructured sample was received during outreach.

It was not integrated into the training pipeline because:
- fields were inconsistent
- merchant and account descriptors were not standardized
- row structure was not reliable enough for repeatable ingestion
- the sample was too sparse and irregular for dependable supervised training

## Interpretation

The outreach effort still mattered even though it produced very little usable data. It confirmed a central project constraint: obtaining real, shareable small-business financial data is substantially harder than building the pipeline itself.

## Conclusion

Real financial data access is the main bottleneck for SourceTax.

That bottleneck affects:
- gold dataset growth
- class balance
- merchant coverage
- validation on realistic business workflows

This is why the current roadmap leans on a hybrid of:
- public or open sample sources
- hand labeling
- carefully separated synthetic augmentation
- future business partnerships with clearer data-sharing boundaries
