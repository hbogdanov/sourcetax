from __future__ import annotations

from pathlib import Path

import app_review


class _FakeUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def test_receipt_csv_ingests_as_tabular_rows(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "store.db"
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_review, "DB_PATH", str(db_path))
    monkeypatch.setattr(app_review, "UPLOAD_DIR", upload_dir)

    payload = (
        "merchant,date,total,direction,receipt_file,confidence\n"
        "STARBUCKS,2026-02-01,12.86,expense,receipt_001.jpg,0.72\n"
        "HOME DEPOT,2026-02-02,188.40,expense,receipt_002.jpg,0.74\n"
    ).encode("utf-8")

    result = app_review.process_uploaded_files([_FakeUpload("receipts.csv", payload)])

    assert len(result) == 1
    assert result[0]["source_detected"] == "receipt"
    assert result[0]["rows_found"] == 2
    assert result[0]["parsed_ok"] == 2
    assert result[0]["parsed_failed"] == 0
    assert result[0]["status"] == "ok"
