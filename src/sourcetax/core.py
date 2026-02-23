"""
Compatibility shim for old tests.

Provides a tiny `parse_stub` that delegates to current ingestion normalization.
This keeps legacy tests working until they are rewritten to call real functions.
"""
from . import ingest


def parse_stub(row: dict, source: str = "bank"):
    """Legacy shim used by tests: normalize an input row to canonical dict.

    Returns the same dict structure as `ingest.normalize_to_canonical`.
    """
    # Support old tests that pass a free-form string like:
    # "Vendor: Coffee Shop; Date: 2026-02-21; Amount: $4.50"
    if isinstance(row, str):
        # Simple parse for legacy tests
        parts = [p.strip() for p in row.split(";") if p.strip()]
        out = {}
        for p in parts:
            if ":" in p:
                k, v = [x.strip() for x in p.split(":", 1)]
                key = k.lower()
                # Normalize amount by stripping $ and commas
                if key == "amount":
                    v = v.replace("$", "").replace(",", "").strip()
                out[key] = v
        return out

    return ingest.normalize_to_canonical(row, source)


__all__ = ["parse_stub"]
