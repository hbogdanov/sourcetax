"""Phase 4c minimal import/export adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Protocol


class ExportAdapter(Protocol):
    def export(self, rows: Iterable[Dict], out_path: str) -> str:
        ...


class CsvExportAdapter:
    """Generic CSV export adapter with stable field ordering."""

    def export(self, rows: Iterable[Dict], out_path: str) -> str:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        rows = list(rows)
        fieldnames = list(rows[0].keys()) if rows else []
        with outp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
        return str(outp)


class QboLikeExportAdapter:
    """Minimal QuickBooks-like adapter for future API integration.

    For now this writes a CSV payload compatible with a common import pattern.
    """

    def export(self, rows: Iterable[Dict], out_path: str) -> str:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Date", "Description", "Amount", "Payee", "Category", "Memo"])
            for row in rows:
                writer.writerow(
                    [
                        row.get("date") or row.get("transaction_date") or "",
                        row.get("merchant_raw") or row.get("entity") or "",
                        row.get("amount") or "",
                        row.get("merchant_raw") or row.get("entity") or "",
                        row.get("effective_category") or row.get("category") or "",
                        row.get("memo") or "",
                    ]
                )
        return str(outp)


class MockQuickBooksApi:
    """Simple file-backed mock API to preserve an integration seam."""

    def __init__(self, root: str = "outputs/mock_qbo_api"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def push_transactions(self, rows: Iterable[Dict], batch_name: str = "transactions") -> str:
        payload = {
            "batch_name": batch_name,
            "count": 0,
            "transactions": [],
        }
        for row in rows:
            payload["transactions"].append(dict(row))
        payload["count"] = len(payload["transactions"])
        out_path = self.root / f"{batch_name}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out_path)
