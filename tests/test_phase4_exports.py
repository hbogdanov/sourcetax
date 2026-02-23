import csv
import json
from pathlib import Path

from sourcetax import exporter, storage
from sourcetax.reconciliation import export_reconciliation_reports


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_phase4_accounting_exports_and_reconciliation(tmp_path: Path):
    db_path = tmp_path / "store.db"
    out_dir = tmp_path / "outputs"
    storage.ensure_db(db_path)

    storage.insert_record(
        {
            "id": "bank_1",
            "source": "bank",
            "source_record_id": "b1",
            "transaction_date": "2026-02-20",
            "merchant_raw": "SQ *STARBUCKS COFFEE 123 SF CA",
            "merchant_norm": "starbucks",
            "amount": 4.5,
            "currency": "USD",
            "direction": "expense",
            "category_pred": "Meals & Entertainment",
            "category_final": None,
            "confidence": 0.61,
            "matched_transaction_id": None,
            "match_score": None,
            "evidence_keys": ["bank_memo"],
            "raw_payload": {"ml_prediction": "Travel", "model_version_hash": "abc123"},
        },
        path=db_path,
    )
    storage.insert_record(
        {
            "id": "receipt_1",
            "source": "receipt",
            "source_record_id": "r1",
            "transaction_date": "2026-02-20",
            "merchant_raw": "STARBUCKS",
            "merchant_norm": "starbucks",
            "amount": 4.5,
            "currency": "USD",
            "direction": "expense",
            "category_pred": "Meals & Entertainment",
            "category_final": "Meals & Entertainment",
            "confidence": 0.99,
            "matched_transaction_id": "bank_1",
            "match_score": 0.88,
            "evidence_keys": ["ocr_text", "receipt1.jpg"],
            "raw_payload": {"description": "Coffee", "label_source": "human"},
        },
        path=db_path,
    )

    bundle = exporter.export_accounting_grade_bundle(
        db_path=str(db_path),
        out_dir=str(out_dir),
        pipeline_version="phase4-test",
        run_id="phase4_test_run",
    )

    tx_rows = _read_csv(Path(bundle["transactions_enriched"]))
    assert len(tx_rows) == 2
    assert {"transaction_id", "merchant_normalized", "label_source", "pipeline_version", "run_id"} <= set(
        tx_rows[0].keys()
    )
    assert any(r["label_source"] == "human" for r in tx_rows)

    gl_rows = _read_csv(Path(bundle["gl_lines"]))
    assert len(gl_rows) == 4  # 2 lines per transaction
    assert {"account", "debit", "credit", "evidence"} <= set(gl_rows[0].keys())

    audit_lines = Path(bundle["audit_trail_jsonl"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 2
    audit_obj = json.loads(audit_lines[0])
    assert "inputs" in audit_obj and "final_decision" in audit_obj and "rule_hits" in audit_obj

    recon_outputs = export_reconciliation_reports(db_path=str(db_path), out_dir=str(out_dir / "recon"))
    summary = json.loads(Path(recon_outputs["summary_metrics"]).read_text(encoding="utf-8"))
    assert "match_rate" in summary and "avg_confidence" in summary
    assert Path(recon_outputs["low_confidence_categorizations"]).exists()
