#!/usr/bin/env python
"""Generate synthetic gap-fill transactions for underrepresented business categories.

Focus categories:
- COGS
- Payroll & Contractors
- Taxes & Licenses
- Insurance
- Professional Services

Approach:
- Template-driven recurring/structured payments (default and always available)
- Optional LLM text diversification, constrained to SourceTax taxonomy labels

This tool writes into staging_transactions.
"""

from __future__ import annotations

import argparse
import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sourcetax import staging, taxonomy
from sourcetax.normalization import normalize_merchant_name


@dataclass(frozen=True)
class CategoryTemplate:
    category: str
    merchants: List[str]
    amount_min: float
    amount_max: float
    cadence: str  # monthly|weekly|biweekly|quarterly|irregular
    channels: List[str]
    mcc_descriptions: List[str]
    memo_templates: List[str]


GAPFILL_TEMPLATES: List[CategoryTemplate] = [
    CategoryTemplate(
        category="COGS",
        merchants=[
            "ABC WHOLESALE SUPPLY",
            "METRO RESTAURANT DEPOT",
            "NATIONAL FOOD DISTRIBUTORS",
            "PACKAGING SOURCE CO",
        ],
        amount_min=50.0,
        amount_max=50000.0,
        cadence="irregular",
        channels=["card", "ach"],
        mcc_descriptions=["WHOLESALE SUPPLIERS", "MISCELLANEOUS GENERAL MERCHANDISE STORES"],
        memo_templates=[
            "Inventory purchase order {po_id}",
            "COGS replenishment batch {batch_id}",
            "Raw materials order {po_id}",
        ],
    ),
    CategoryTemplate(
        category="Payroll & Contractors",
        merchants=["GUSTO", "ADP PAYROLL", "CONTRACTOR PAYMENTS LLC", "PAYCHEX"],
        amount_min=200.0,
        amount_max=15000.0,
        cadence="biweekly",
        channels=["ach"],
        mcc_descriptions=["BUSINESS SERVICES NOT ELSEWHERE CLASSIFIED"],
        memo_templates=[
            "Payroll period {period_label}",
            "Contractor disbursement {period_label}",
            "Payroll run #{run_id}",
        ],
    ),
    CategoryTemplate(
        category="Taxes & Licenses",
        merchants=["IRS EFTPS", "STATE TAX AGENCY", "CITY BUSINESS LICENSE OFFICE"],
        amount_min=50.0,
        amount_max=20000.0,
        cadence="quarterly",
        channels=["ach", "wire"],
        mcc_descriptions=["TAX PAYMENTS", "GOVERNMENT SERVICES"],
        memo_templates=[
            "Quarterly estimated tax Q{quarter}",
            "Business license renewal {year}",
            "Tax payment confirmation {confirmation_id}",
        ],
    ),
    CategoryTemplate(
        category="Insurance",
        merchants=["HISCOX INSURANCE", "TRAVELERS COMMERCIAL", "STATE FARM BUSINESS"],
        amount_min=50.0,
        amount_max=2000.0,
        cadence="monthly",
        channels=["ach", "card"],
        mcc_descriptions=["INSURANCE SERVICES"],
        memo_templates=[
            "General liability premium {month}",
            "Workers comp premium {month}",
            "Commercial auto policy payment {month}",
        ],
    ),
    CategoryTemplate(
        category="Professional Services",
        merchants=["SMITH CPA GROUP", "LEGAL PARTNERS LLP", "FREELANCE DESIGN STUDIO"],
        amount_min=150.0,
        amount_max=10000.0,
        cadence="monthly",
        channels=["ach", "wire", "card"],
        mcc_descriptions=["ACCOUNTING, AUDITING, AND BOOKKEEPING SERVICES", "LEGAL SERVICES AND ATTORNEYS"],
        memo_templates=[
            "Monthly bookkeeping services {month}",
            "Legal advisory invoice {invoice_id}",
            "Professional consulting engagement {project_id}",
        ],
    ),
    # Structured recurring categories called out in the plan
    CategoryTemplate(
        category="Rent & Utilities",
        merchants=["PROPERTY MGMT GROUP", "COMCAST BUSINESS", "CITY WATER AUTHORITY", "ELECTRIC SERVICE CO"],
        amount_min=80.0,
        amount_max=6000.0,
        cadence="monthly",
        channels=["ach", "card"],
        mcc_descriptions=["UTILITY SERVICES", "TELECOMMUNICATION SERVICES"],
        memo_templates=[
            "Office rent {month}",
            "Business internet service {month}",
            "Utility bill payment {month}",
        ],
    ),
]


def _date_add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _next_date(prev: date, cadence: str, rng: random.Random) -> date:
    if cadence == "weekly":
        return prev + timedelta(days=7)
    if cadence == "biweekly":
        return prev + timedelta(days=14)
    if cadence == "monthly":
        return _date_add_months(prev, 1)
    if cadence == "quarterly":
        return _date_add_months(prev, 3)
    # irregular
    return prev + timedelta(days=rng.randint(5, 35))


def _build_template_row(
    *,
    template: CategoryTemplate,
    run_id: str,
    idx: int,
    tx_date: date,
    rng: random.Random,
) -> Dict[str, Any]:
    merchant = rng.choice(template.merchants)
    amount = -round(rng.uniform(template.amount_min, template.amount_max), 2)
    channel = rng.choice(template.channels)
    mcc_desc = rng.choice(template.mcc_descriptions) if template.mcc_descriptions else None
    month_label = tx_date.strftime("%b %Y")
    memo = rng.choice(template.memo_templates).format(
        month=month_label,
        year=tx_date.year,
        quarter=((tx_date.month - 1) // 3) + 1,
        run_id=f"{run_id}-{idx:05d}",
        period_label=f"{tx_date.isoformat()}",
        confirmation_id=f"CNF{rng.randint(100000, 999999)}",
        invoice_id=f"INV{rng.randint(1000, 9999)}",
        project_id=f"PRJ{rng.randint(100, 999)}",
        po_id=f"PO{rng.randint(10000, 99999)}",
        batch_id=f"B{rng.randint(1000, 9999)}",
    )
    merchant_norm = normalize_merchant_name(merchant, case="lower")

    return {
        "source": "synthetic_gapfill",
        "source_record_id": f"{run_id}_{idx:05d}",
        "txn_ts": tx_date.isoformat(),
        "amount": amount,
        "currency": "USD",
        "merchant_raw": merchant,
        "description": memo,
        "mcc": None,
        "mcc_description": mcc_desc,
        "category_external": template.category,
        "subcategory_external": None,
        "raw_payload_json": {
            "generator": "tools/generate_synthetic_gapfill.py",
            "generator_mode": "template",
            "target_category": template.category,
            "merchant_norm": merchant_norm,
            "channel": channel,
            "cadence": template.cadence,
            "realism_tags": ["gapfill", "synthetic", "business_category_boost"],
        },
    }


def _maybe_llm_diversify(
    rows: List[Dict[str, Any]],
    *,
    enabled: bool,
    model: str,
) -> List[Dict[str, Any]]:
    """Optional setup hook for LLM diversification.

    Currently best-effort: if OpenAI SDK/key are unavailable, it no-ops and keeps template text.
    """
    if not enabled or not rows:
        return rows
    try:
        from openai import OpenAI  # type: ignore
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return rows
        client = OpenAI(api_key=api_key)
    except Exception:
        return rows

    allowed = set(taxonomy.load_sourcetax_categories())
    out: List[Dict[str, Any]] = []
    for row in rows:
        prompt = (
            "Rewrite this merchant transaction description to be realistic for bank feeds. "
            "Keep category unchanged and return one short line only.\n"
            f"Category: {row.get('category_external')}\n"
            f"Merchant: {row.get('merchant_raw')}\n"
            f"Description: {row.get('description')}\n"
        )
        try:
            resp = client.responses.create(model=model, input=prompt, max_output_tokens=40)
            text = getattr(resp, "output_text", None) or row.get("description") or ""
            description = str(text).strip().splitlines()[0][:120]
        except Exception:
            description = row.get("description") or ""
        r = dict(row)
        category = r.get("category_external")
        if category not in allowed:
            # Enforce label-set constraint.
            r["category_external"] = "Other Expense"
        r["description"] = description
        payload = dict(r.get("raw_payload_json") or {})
        payload["generator_mode"] = "llm+template"
        payload["llm_model"] = model
        r["raw_payload_json"] = payload
        out.append(r)
    return out


def build_gapfill_rows(
    *,
    total_rows: int,
    seed: int,
    start_date: date,
    run_id: str,
    use_llm: bool,
    llm_model: str,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    allowed = set(taxonomy.load_sourcetax_categories())
    templates = [t for t in GAPFILL_TEMPLATES if t.category in allowed]
    if categories:
        selected = set(categories)
        templates = [t for t in templates if t.category in selected]
    if not templates:
        return []

    rows: List[Dict[str, Any]] = []
    date_cursor = start_date
    idx = 0
    while len(rows) < total_rows:
        for tmpl in templates:
            if len(rows) >= total_rows:
                break
            idx += 1
            tx_date = _next_date(date_cursor, tmpl.cadence, rng)
            rows.append(
                _build_template_row(
                    template=tmpl,
                    run_id=run_id,
                    idx=idx,
                    tx_date=tx_date,
                    rng=rng,
                )
            )
        date_cursor = date_cursor + timedelta(days=3)

    return _maybe_llm_diversify(rows, enabled=use_llm, model=llm_model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/staging.db")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="gapfill_v1")
    parser.add_argument("--start-date", default=date.today().isoformat())
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated SourceTax categories to include (default: template set).",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--use-llm", action="store_true", help="Enable optional LLM description diversification.")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--dry-run", action="store_true", help="Build and preview without DB writes.")
    args = parser.parse_args()

    parsed_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None

    rows = build_gapfill_rows(
        total_rows=args.rows,
        seed=args.seed,
        start_date=parsed_date,
        run_id=args.run_id,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        categories=categories,
    )

    if args.dry_run:
        print(f"Dry run complete. rows_built={len(rows)}")
        for sample in rows[:3]:
            print(sample)
        return 0

    db_path = Path(args.staging_db)
    inserted = staging.insert_staging_transactions(rows, path=db_path, batch_size=args.batch_size)
    counts = staging.get_staging_counts(path=db_path)
    print("Gapfill generation complete.")
    print(f"- rows_built: {len(rows)}")
    print(f"- rows_inserted: {inserted}")
    print(f"- staging_transactions_total: {counts['staging_transactions']}")
    print(f"- staging_receipts_total: {counts['staging_receipts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

