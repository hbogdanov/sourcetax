#!/usr/bin/env python
"""
Generate locked train/val/test splits from `data/gold/gold_transactions.jsonl`.

Usage:
  python tools/generate_splits.py --gold data/gold/gold_transactions.jsonl --out data/ml --seed 42

Produces CSVs: ml_train.csv, ml_val.csv, ml_test.csv and split_metadata.txt
Handles small-class counts by fallback to stratify=None when necessary.
"""
import argparse
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_gold(gold_path: Path) -> pd.DataFrame:
    rows = []
    if not gold_path.exists():
        return pd.DataFrame()
    with gold_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', default='data/gold/gold_transactions.jsonl')
    parser.add_argument('--out', default='data/ml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gold_path = Path(args.gold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_gold(gold_path)
    if df.empty:
        print('No gold records found; create labels first')
        return 1

    # Ensure required columns
    if 'category' not in df.columns:
        df['category'] = 'Unlabeled'

    # Convert category to str
    df['category'] = df['category'].astype(str)

    # If any class has fewer than 2 samples, fall back to unstratified split
    stratify = df['category'] if df['category'].value_counts().min() >= 2 else None

    # Train/Test split 85/15 then val split from train 82.35/17.65 to approximate 70/15/15
    if stratify is not None:
        X_train, X_test = train_test_split(df, test_size=0.15, random_state=args.seed, stratify=stratify)
        strat_train = X_train['category'] if X_train['category'].value_counts().min() >= 2 else None
        X_train, X_val = train_test_split(X_train, test_size=0.1765, random_state=args.seed, stratify=strat_train)
    else:
        X_train, X_test = train_test_split(df, test_size=0.15, random_state=args.seed, shuffle=True)
        X_train, X_val = train_test_split(X_train, test_size=0.1765, random_state=args.seed, shuffle=True)

    X_train.to_csv(out_dir / 'ml_train.csv', index=False)
    X_val.to_csv(out_dir / 'ml_val.csv', index=False)
    X_test.to_csv(out_dir / 'ml_test.csv', index=False)

    meta = f"seed: {args.seed}\nrecords: {len(df)}\ntrain: {len(X_train)}\nval: {len(X_val)}\ntest: {len(X_test)}\n"
    (out_dir / 'split_metadata.txt').write_text(meta)

    print('Wrote splits to', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
