#!/usr/bin/env python
"""Import DC purchase card ArcGIS feed into staging_transactions.

Target mapping:
- source='dc_pcard'
- merchant_raw <- VENDOR_NAME
- amount <- -abs(TRANSACTION_AMOUNT)  (force expense convention)
- mcc_description <- MCC_DESCRIPTION
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sourcetax import staging


DEFAULT_QUERY_URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/"
    "DCGIS_DATA/Public_Service_WebMercator/MapServer/50/query"
)


def _fetch_json(url: str, params: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    with urllib.request.urlopen(full_url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _to_iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        # ArcGIS commonly returns epoch milliseconds for Date fields.
        if isinstance(value, (int, float)):
            ts = float(value) / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        s = str(value).strip()
        if not s:
            return None
        # Already iso-ish
        if "-" in s:
            return s[:10]
        return None
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        try:
            return float(str(value).replace("$", "").replace(",", "").strip())
        except Exception:
            return None


def _pick(attrs: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in attrs and attrs.get(k) is not None:
            return attrs.get(k)
    return None


def _feature_to_staging(attrs: Dict[str, Any], idx: int) -> Dict[str, Any]:
    vendor = _pick(attrs, "VENDOR_NAME", "Vendor_Name", "vendor_name")
    raw_amount = _pick(attrs, "TRANSACTION_AMOUNT", "Transaction_Amount", "transaction_amount")
    amount = _to_float(raw_amount)
    txn_date = _pick(attrs, "TRANSACTION_DATE", "Transaction_Date", "transaction_date")
    mcc_desc = _pick(attrs, "MCC_DESCRIPTION", "MCC_Description", "mcc_description")
    obj_id = _pick(attrs, "OBJECTID", "OBJECTID_1", "objectid")

    return {
        "source": "dc_pcard",
        "source_record_id": str(obj_id if obj_id is not None else f"dc_{idx}"),
        "txn_ts": _to_iso_date(txn_date),
        "amount": -abs(amount) if amount is not None else None,
        "currency": "USD",
        "merchant_raw": str(vendor).strip() if vendor is not None else None,
        "description": None,
        "mcc": None,
        "mcc_description": str(mcc_desc).strip() if mcc_desc is not None else None,
        "category_external": None,
        "subcategory_external": None,
        "raw_payload_json": {
            "dataset": "dc_arcgis_pcard",
            "country": "US",
            "raw_attributes": attrs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-db", default="data/interim/staging.db")
    parser.add_argument("--query-url", default=DEFAULT_QUERY_URL)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--where", default="1=1")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    db_path = Path(args.staging_db)
    staging.ensure_staging_db(db_path)

    offset = 0
    scanned = 0
    inserted_total = 0
    buffer: List[Dict[str, Any]] = []

    fields = ",".join(
        [
            "OBJECTID",
            "VENDOR_NAME",
            "TRANSACTION_AMOUNT",
            "TRANSACTION_DATE",
            "MCC_DESCRIPTION",
        ]
    )

    while scanned < args.max_rows:
        page_count = min(args.page_size, args.max_rows - scanned)
        params = {
            "f": "json",
            "where": args.where,
            "outFields": fields,
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": page_count,
            "orderByFields": "OBJECTID ASC",
        }
        payload = _fetch_json(args.query_url, params=params, timeout=args.timeout)
        features = payload.get("features") or []
        if not features:
            break

        for feat in features:
            attrs = feat.get("attributes") or {}
            buffer.append(_feature_to_staging(attrs, scanned))
            scanned += 1
            if len(buffer) >= args.batch_size:
                inserted_total += staging.insert_staging_transactions(
                    buffer, path=db_path, batch_size=args.batch_size
                )
                buffer = []
                if inserted_total % (args.batch_size * 5) == 0:
                    print(f"Inserted {inserted_total} rows...")
            if scanned >= args.max_rows:
                break

        offset += len(features)
        if len(features) < page_count:
            break

    if buffer:
        inserted_total += staging.insert_staging_transactions(
            buffer, path=db_path, batch_size=args.batch_size
        )

    counts = staging.get_staging_counts(db_path)
    print("Import complete.")
    print(f"- inserted_rows: {inserted_total}")
    print(f"- scanned_rows: {scanned}")
    print(f"- staging_transactions_total: {counts['staging_transactions']}")
    print(f"- staging_receipts_total: {counts['staging_receipts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


