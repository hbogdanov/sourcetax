"""Fetch public datasets useful for receipts/pos/bank training.

Tries multiple sources (Hugging Face -> direct download fallbacks).
Saves samples under `data/receipts/cord`, `data/receipts/sroie`, `data/forms/funsd`.
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)


def fetch_hf_dataset(name, split="train", dest=None, sample_n=0):
    try:
        from datasets import load_dataset
    except Exception as e:
        print("datasets not installed:", e)
        return False
    # helper for saving potential image/blob fields
    from PIL import Image

    try:
        ds = load_dataset(name, split=split)
        if sample_n:
            ds = ds.select(range(min(sample_n, len(ds))))
        dest = Path(dest) if dest else OUT / name.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        images_dir = dest / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        # Save first N as json, writing images to disk when present
        for i, ex in enumerate(ds):
            out_ex = {}
            for k, v in ex.items():
                try:
                    # If v is an image (PIL.Image) save it
                    if isinstance(v, Image.Image):
                        img_path = images_dir / f"{name.replace('/', '_')}_{i}_{k}.png"
                        v.save(img_path)
                        out_ex[k] = str(img_path.name)
                    else:
                        # try to json-serialize; fallback to string
                        json.dumps(v)
                        out_ex[k] = v
                except TypeError:
                    out_ex[k] = str(v)
            with open(dest / f"{name.replace('/', '_')}_{i}.json", "w", encoding="utf-8") as f:
                json.dump(out_ex, f, ensure_ascii=False)
        print(f"Wrote {i + 1} examples to {dest} (images in {images_dir})")
        return True
    except Exception as e:
        print("HF load_dataset failed for", name, "->", e)
        return False


def main():
    print("Attempting to fetch public datasets...")

    # Try CORD from HF
    cord_dest = OUT / "receipts" / "cord"
    cord_dest.mkdir(parents=True, exist_ok=True)
    ok = fetch_hf_dataset("clovaai/cord", split="train", dest=cord_dest, sample_n=100)
    if not ok:
        print("CORD fetch failed or not available via HF here. Kept existing cord samples if any.")

    # Try SROIE - common HF ids tried
    sroie_dest = OUT / "receipts" / "sroie"
    sroie_dest.mkdir(parents=True, exist_ok=True)
    for candidate in ["sroie", "sergio/sroie", "funsd/sroie"]:
        if fetch_hf_dataset(candidate, split="train", dest=sroie_dest, sample_n=200):
            break

    # Try FUNSD
    funsd_dest = OUT / "forms" / "funsd"
    funsd_dest.mkdir(parents=True, exist_ok=True)
    for candidate in ["nielsr/funsd", "tesserae/funsd"]:
        if fetch_hf_dataset(candidate, split="train", dest=funsd_dest, sample_n=200):
            break

    print(
        "Done. Note: Kaggle-hosted retail datasets require Kaggle API credentials; see README for manual steps."
    )


if __name__ == "__main__":
    main()
