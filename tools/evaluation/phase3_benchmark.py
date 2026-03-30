#!/usr/bin/env python
"""Grouped launcher for phase 3 benchmarking."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
