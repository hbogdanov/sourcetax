# Phase 2: Receipt Extraction & Transaction Linking

**End State:** User uploads bank CSV + receipts (images/PDF) → auto-extract fields → link receipts to transactions → predict Schedule C categories → export QB import CSV + Schedule C totals.

Not perfect. End-to-end.

---

## Step 1: Lock Canonical Schema (MUST DO FIRST)

**Rule:** Everything transforms into canonical as early as possible. Don't refactor schema once extraction starts.

### Required Fields (audit before implementation)

Identifiers:
- `record_id` – stable UUID (not auto-increment, allows replay)
- `source` – bank|receipt|toast|quickbooks|manual
- `source_id` – row id / filename + index (for traceability)

Temporal:
- `date` – ISO 8601 string

Merchant:
- `merchant_raw` – as-written from source
- `merchant_norm` – lowercased, punctuation stripped (for matching)

Amount:
- `amount` – float, always positive
- `currency` – ISO 4217 code (default "USD")
- `direction` – "expense" | "income" (no sign ambiguity)

Categorization:
- `category_pred` – Schedule C category code from rules/ML
- `category_final` – user override if present
- `confidence` – float 0–1 (high if exact match, low if keyword/unknown)

Matching:
- `matched_transaction_id` – reference to linked bank transaction (if any)
- `match_score` – 0–1, evidence for match

Evidence:
- `evidence_keys` – list of strings (["ocr_text", "receipt_file.jpg", "bank_memo"])
- `raw_payload` – JSON blob (original OCR text, extracted fields, all heuristic outputs)

### Audit Current Schema

Check [src/sourcetax/schema.py](src/sourcetax/schema.py) against above.

Missing → Add before Phase 2.1 starts.

---

## Step 2: Receipt Ingestion + OCR Extraction (MVP)

### 2A: Receipt File Handling

Support formats:
- JPG, PNG (single image)
- PDF (single + multi-page)

Storage:
- Store original file in `data/uploads/{source}/{hash}.{ext}` (deduplicate by hash)
- Add to .gitignore: `data/uploads/`
- Extracted text + metadata → canonical record
- File reference in `evidence_keys` (e.g., "receipts/abc123.jpg")

### 2B: OCR Baseline (Fast to Implement)

Choose one (in order of simplicity):

1. **Tesseract** (local, cheap, imperfect)
   - Install: `pip install pytesseract pillow` + system tesseract
   - Fast (~50ms per receipt)
   - Works surprisingly well on receipt scans
   - Output: full text + per-word confidence (optional)

2. **EasyOCR** (better out-of-box on messy receipts)
   - Install: `pip install easyocr`
   - Slightly slower (~200ms) but handles rotations/poor quality
   - Better on slanted photos

3. **Cloud OCR** (skip for now)
   - Google Vision / Azure Computer Vision
   - Use if accuracy becomes blocking

**MVP OCR pipeline output:**
```json
{
  "full_text": "...",
  "confidence_avg": 0.87,
  "word_boxes": [optional]
}
```

### 2C: Field Extraction (Cheap Heuristics First)

Extract (with line source and confidence):

- **date** → regex: `[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}` etc.
- **total** → search lines: TOTAL, AMOUNT, BALANCE, GRANT TOTAL
- **tax** → search: TAX, GST, SALES TAX (amount after "tax" keyword)
- **tip** → search: TIP, GRATUITY
- **merchant** → top of receipt (first 1–3 non-empty lines, stripped of junk like store numbers)

**Store with confidence:**
- High (0.9+): regex hit on known format
- Medium (0.6–0.89): fuzzy keyword match
- Low (<0.6): heuristic guess or missing

**First win:** reasonable totals + date + merchant name. Don't overbuild.

---

## Step 3: Transaction Matching (Receipt ↔ Bank)

**Goal:** Auto-attach receipts to bank transactions.

### Matching Algorithm

For each receipt record:

1. **Find candidate transactions:** ±3 days from receipt date
2. **Amount similarity:**
   - Receipt total should match transaction amount ±2–10 (allow tax/tip variance)
3. **Merchant similarity:**
   - Normalize both: lowercase, strip punctuation & numbers
   - Fuzzy match (rapidfuzz) – accept >80% similarity
4. **Score** (simple):
   - date closeness: (3 - days_diff) / 3
   - amount closeness: 1 - (abs_diff / amount)
   - merchant similarity: fuzzy_score / 100
   - weighted average or pick best single factor
5. **Decision:**
   - Score >0.75 → matched, store `matched_transaction_id`
   - Else → unmatched, flag for review

**Store in canonical:**
- `matched_transaction_id` (if match found)
- `match_score`
- `evidence_keys` includes match reasoning

This alone is a killer demo.

---

## Step 4: Categorization v1 (Rules, Not ML)

**Anti-pattern:** Don't train a model until labeling loop exists.

### Rule Engines (in priority order)

1. **Merchant exact match** → use existing `data/mappings/merchant_category.csv`
   - Confidence: 0.95

2. **Merchant fuzzy match** → if merchant_norm within 80% of known merchant
   - Confidence: 0.75

3. **Keyword rules** (if merchant not in table):
   - HOME DEPOT → Supplies (66)
   - UBER / LYFT → Travel (43)
   - STARBUCKS / CAFE → Meals (32)
   - AMAZON → Supplies/Office (depends, ask user)
   - Confidence: 0.6

4. **Learned overrides table** (manual corrections persist):
   - User changes category for "LOCAL PIZZA CO" → user says it's Meals (32)
   - Next time LOCAL PIZZA CO matches → use Meals (32) automatically
   - Confidence: 0.95

### Output

- `category_pred` – best guess (from above priority)
- `category_final` – user override (if present) → always use for export
- `confidence` – trust level per above rules

### Example

```
Receipt: "STARBUCKS #1234" $5.50
  ↓ exact merchant match: NO
  ↓ fuzzy merchant match: maybe "STARBUCKS" (learned override exists?)
  ↓ keyword match: "STARBUCKS" → Meals (32)
  category_pred = "Meals (32)", confidence = 0.6
  (user can override to "Supplies" if receipts are office perks)
```

---

## Step 5: Review UI (The Non-Negotiable Piece)

Without review, you're a script. With review, you're a product.

### Phase 2 UI Requirements (Basic)

**Tech:** Streamlit (ships in a day, allows iteration)

**Main table:**
- Columns: date, merchant, amount, category, match_score, confidence, actions
- Sortable, filterable

**Filters:**
- Unmatched receipts
- Unmatched transactions
- Low confidence category (<0.7)
- By date range

**Click row → Side panel:**
- OCR text preview (truncated)
- Receipt image thumbnail
- Matched transaction details (if any)
- Category dropdown (override + save)
- "Link to transaction" action (for unmatched)
- "Mark as duplicated" or "Delete"

**Bottom metrics:**
- Total records
- Matched ✓ / Unmatched ✗
- Needs review (low conf)
- Ready to export

### Tech Stack

- **Streamlit** (MVP, fast): `pip install streamlit`
- **FastAPI + React** (later, if you need persistent state / real app)

**Recommendation:** Streamlit first. You'll ship this week if you focus.

---

## Step 6: Export Upgrades (Deterministic Outputs)

### QuickBooks Import CSV

Columns (standard format):
```
Date,Payee,Category,Amount
2025-02-22,STARBUCKS,Meals,5.50
```

Or split debit/credit:
```
Date,Payee,Category,Debit,Credit
2025-02-22,STARBUCKS,Meals,5.50,
```

**Rules:**
- Use `category_final` (user overrides) if present, else `category_pred`
- Amount: always positive (direction in column logic)
- Only direction="expense" for Schedule C exports
- Deterministic order (by date, then merchant)

### Schedule C Totals

```
Category,Code,Total,Count,Monthly_Breakdown
Meals & Entertainment,32,1234.56,42,"Jan: 123.45, Feb: 456.78, ..."
Supplies,66,789.23,18,"Jan: 234.56, ..."
```

**Extra metrics (for product dashboard):**
```
{
  "total_expenses": 2023.79,
  "unmatched_receipts": 3,
  "unmatched_transactions": 1,
  "needs_review": 5,
  "category_distribution": {...}
}
```

---

## Step 7: Dataset & Evaluation Loop (Optional But Smart)

### Create Gold Set

200 transactions (bank CSV)
100 receipts (real PDFs/JPGs)
Manually verified:
- Correct matches (receipt ↔ transaction)
- Correct categories

### Track Metrics

- **Receipt extraction:** date accuracy, total accuracy, merchant accuracy (F1)
- **Matching:** precision, recall, F1
- **Categorization:** accuracy per category, confusion matrix
- **User overrides:** how many category changes, which merchants most disputed

### Loop

Run Phase 2.4 → export metrics → find bugs → refine rules → repeat

This prevents hallucinating progress.

---

## Phase 2 Checklist (Week-by-Week Pacing)

### Phase 2.1 (Plumbing) – Week 1

- [ ] Lock canonical schema + audit fields
- [ ] Receipt ingestion: file upload → hash → storage
- [ ] OCR integration (Tesseract or EasyOCR)
- [ ] Basic field extraction (date, total, merchant)
- [ ] Store extracted data in canonical records

### Phase 2.2 (Linking) – Week 2

- [ ] Receipt ↔ bank transaction matching algorithm
- [ ] Store match_score + matched_transaction_id
- [ ] Basic UI: list unmatched receipts / unmatched transactions

### Phase 2.3 (Categorization + Review) – Week 2–3

- [ ] Build rules engine (exact → fuzzy → keyword → learned)
- [ ] Add learned overrides table to DB
- [ ] Streamlit UI: category dropdown + save
- [ ] Filter UI: low confidence, unmatched, by date

### Phase 2.4 (Exports + Demo) – Week 3–4

- [ ] Update QB CSV export (use category_final)
- [ ] Update Schedule C export (totals + monthly breakdown)
- [ ] End-to-end demo: upload CSV + receipts → exports
- [ ] Document schema + UI in README

---

## Biggest "Don't Be Dumb" Warnings

1. **Don't chase Toast/QuickBooks API integrations in Phase 2.** CSV uploads are enough. APIs are a Phase 3 or 4 problem.

2. **Don't train a big ML model until canonical schema + labeling loop exist.** Rules + learned overrides will carry you far.

3. **Don't store raw receipts in git ever again.** `data/uploads/` → .gitignore. S3 later if you scale.

4. **Don't overbuild extraction.** Heuristics + manual review beats 90% ML spend. Start simple.

5. **Don't skip the review UI.** Without it, you can't iterate or get user feedback. Ship it cheap and fast.

6. **Don't export unreviewed data.** Mark "needs_review" = true if confidence < 0.7 or unmatched. Let user decide.

---

## Success Criteria for Phase 2

- [ ] End-to-end: upload CSV + receipts → categorized transactions → QB/Schedule C exports
- [ ] Matching demo: 80%+ precision on test set of 10 manual matches
- [ ] UI: user can see low-confidence items, review, and override categories
- [ ] All canonical records have consistent field structure
- [ ] Exports are deterministic and auditable (same input → same output)
- [ ] No more refactoring of core schema

**Estimated effort:** 3–4 weeks (plumbing + UI + iteration). Depends on OCR surprises.
