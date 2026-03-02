from datetime import date

from tools.generate_synthetic_gapfill import build_gapfill_rows
from sourcetax import taxonomy


def test_gapfill_rows_follow_taxonomy_and_payload_shape():
    rows = build_gapfill_rows(
        total_rows=20,
        seed=7,
        start_date=date(2026, 3, 2),
        run_id="test_gapfill",
        use_llm=False,
        llm_model="gpt-4.1-mini",
    )
    assert len(rows) == 20

    allowed = set(taxonomy.load_sourcetax_categories())
    seen = set()
    for r in rows:
        seen.add(r["category_external"])
        assert r["category_external"] in allowed
        assert r["source"] == "synthetic_gapfill"
        assert r["amount"] < 0
        assert isinstance(r["raw_payload_json"], dict)
        assert r["raw_payload_json"].get("generator_mode") in {"template", "llm+template"}
        assert r["raw_payload_json"].get("merchant_norm")

    # Ensure key undercovered categories appear in generated set.
    for must_have in {"COGS", "Payroll & Contractors", "Taxes & Licenses", "Insurance", "Professional Services"}:
        assert must_have in seen

