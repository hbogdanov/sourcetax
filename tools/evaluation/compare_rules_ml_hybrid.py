#!/usr/bin/env python
"""Grouped launcher for rules vs ML vs hybrid comparisons."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_rules_ml_hybrid import run


if __name__ == "__main__":
    raise SystemExit(run())
