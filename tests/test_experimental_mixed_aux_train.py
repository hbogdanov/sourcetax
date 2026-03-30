import json
from pathlib import Path

from tools.training.experimental_mixed_aux_train import _build_synth_df, _synthetic_row_cap


def test_synthetic_row_cap_respects_requested_ratio_and_available_rows():
    assert _synthetic_row_cap(412, 0.20, 1200) == 82
    assert _synthetic_row_cap(412, 0.35, 1200) == 144
    assert _synthetic_row_cap(412, 0.35, 10) == 10
    assert _synthetic_row_cap(412, 0.0, 1200) == 0


def test_build_synth_df_uses_mapped_category_and_text(tmp_path: Path):
    jsonl_path = tmp_path / "synthetic_rows.jsonl"
    rows = [
        {
            "source": "synthetic_gapfill",
            "source_record_id": "gapfill_001",
            "merchant_raw": "PAYCHEX",
            "description": "Payroll period 2025-01-15",
            "category_mapped": "Payroll & Contractors",
        },
        {
            "source_record_id": "gapfill_002",
            "merchant_raw": "",
            "description": "",
            "category_mapped": "COGS",
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    df = _build_synth_df(jsonl_path)

    assert len(df) == 1
    assert df.iloc[0]["id"] == "synthetic_aux:gapfill_001"
    assert df.iloc[0]["category"] == "Payroll & Contractors"
    assert df.iloc[0]["source"] == "synthetic_gapfill"
    assert bool(df.iloc[0]["is_synthetic"]) is True
    assert "paychex" in df.iloc[0]["text"]
    assert "payroll period 2025-01-15" in df.iloc[0]["text"]
