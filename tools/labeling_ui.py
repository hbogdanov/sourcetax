"""
Minimal Streamlit labeling UI.

Usage:
  streamlit run tools/labeling_ui.py -- --input data/ml/selection_for_labeling.csv

This app loads a CSV of candidate rows (from `select_for_labeling.py`),
lets a reviewer set `category` and `keep` flags, and saves labeled CSVs
and appends labeled records to `data/gold/gold_transactions.jsonl`.
"""
import argparse
from pathlib import Path
import pandas as pd
import streamlit as st
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sourcetax import taxonomy


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Selection file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def append_to_gold(df: pd.DataFrame, gold_path: Path):
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            category = taxonomy.normalize_category_name(row.get("category"))
            if not category:
                continue
            rec = {
                "id": f"gold_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                "merchant_raw": row.get("merchant", ""),
                "merchant_norm": row.get("merchant", ""),
                "transaction_date": row.get("transaction_date", ""),
                "amount": row.get("amount", None),
                "category_final": category,
                "sourcetax_category_v1": category,
                "label_confidence": "medium",
                "label_notes": "",
                "source": row.get("source", "unlabeled"),
                "raw_payload": {"label_source": "human", "description": row.get("text", "")},
            }
            f.write(json.dumps(rec) + "\n")


def main():
    st.title("Labeling UI — Select & Export")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ml/selection_for_labeling.csv")
    parser.add_argument("--gold", default="data/gold/gold_transactions.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    gold_path = Path(args.gold)

    df = load_candidates(input_path)
    if df.empty:
        st.info("No candidates available. Run tools/select_for_labeling.py first.")
        return

    st.write("Loaded", len(df), "candidates")

    # Allow user to edit categories inline
    if "category" not in df.columns:
        df["category"] = ""

    edited = st.experimental_data_editor(df, num_rows="dynamic")

    if st.button("Append labeled rows to gold set"):
        labeled = edited[edited["category"].astype(bool)]
        append_to_gold(labeled, gold_path)
        st.success(f"Appended {len(labeled)} labeled records to {gold_path}")


if __name__ == "__main__":
    main()
