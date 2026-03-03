#!/usr/bin/env python
"""Build a reproducibility manifest (with SHA256 hashes) for fetched datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.rglob("*") if p.is_file()])


def build_manifest(paths: List[Path]) -> Dict:
    files = []
    total_size = 0
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        size = p.stat().st_size
        total_size += size
        files.append(
            {
                "path": rel,
                "size_bytes": size,
                "sha256": sha256_file(p),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        default="data/receipts,data/forms",
        help="Comma-separated directories to include in manifest.",
    )
    parser.add_argument("--out", default="data/dataset_manifest.json")
    args = parser.parse_args()

    root_dirs = [ROOT / x.strip() for x in args.roots.split(",") if x.strip()]
    all_files: List[Path] = []
    for rd in root_dirs:
        all_files.extend(collect_files(rd))

    manifest = build_manifest(all_files)
    manifest["roots"] = [str(rd.relative_to(ROOT)).replace("\\", "/") for rd in root_dirs]

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote {outp} with {manifest['file_count']} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

