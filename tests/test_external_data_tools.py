from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_external_tools_help():
    commands = [
        [sys.executable, "tools/import_hf_mitulshah.py", "--help"],
        [sys.executable, "tools/build_mitulshah_corpus.py", "--help"],
    ]
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"command failed: {' '.join(cmd)}\n{proc.stderr}"
        assert "usage" in (proc.stdout or "").lower()


def test_build_mitulshah_corpus_from_dummy_dataset(tmp_path):
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")

    dataset = datasets.Dataset.from_dict(
        {
            "transaction_description": [
                "STARBUCKS STORE 1234",
                "ADP PAYROLL SERVICE",
                "MONTHLY BANK FEE",
                "DELTA AIRLINES TICKET",
            ],
            "category": ["Food & Dining", "Income", "Financial", "Travel"],
            "country": ["US", "US", "US", "US"],
            "currency": ["USD", "USD", "USD", "USD"],
        }
    )
    mirror_dir = tmp_path / "mirror"
    datasets.DatasetDict({"train": dataset}).save_to_disk(str(mirror_dir))

    out_path = tmp_path / "corpus.parquet"
    cmd = [
        sys.executable,
        "tools/build_mitulshah_corpus.py",
        "--in-dir",
        str(mirror_dir),
        "--out-path",
        str(out_path),
        "--split",
        "train",
        "--batch-size",
        "2",
        "--min-rows",
        "1",
        "--min-labels",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    assert Path(str(out_path).replace(".parquet", ".manifest.json")).exists()
