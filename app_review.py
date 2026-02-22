"""
SourceTax Review UI - Streamlit app for Phase 2.

Allows users to:
- View unmatched receipts
- View unmatched transactions
- Link receipts to transactions manually
- Review and override categories
- See matching scores and OCR excerpts
"""

import streamlit as st
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sourcetax import matching, storage, categorization


st.set_page_config(page_title="SourceTax Review", layout="wide")
st.title("📋 SourceTax Transaction Review")

# Database path
DB_PATH = "data/store.db"


def fetch_record(record_id: str) -> Optional[Dict]:
    """Fetch a single record by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM canonical_records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats() -> Dict:
    """Get transaction/matching stats."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM canonical_records")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM canonical_records WHERE source = 'receipt'")
    total_receipts = cur.fetchone()[0]
    
    cur.execute(
        "SELECT COUNT(*) FROM canonical_records WHERE source = 'receipt' AND matched_transaction_id IS NOT NULL"
    )
    matched_receipts = cur.fetchone()[0]
    
    cur.execute(
        "SELECT COUNT(*) FROM canonical_records WHERE source = 'receipt' AND confidence < 0.7"
    )
    low_confidence = cur.fetchone()[0]
    
    conn.close()
    
    return {
        "total": total,
        "total_receipts": total_receipts,
        "matched_receipts": matched_receipts,
        "unmatched_receipts": total_receipts - matched_receipts,
        "low_confidence": low_confidence,
    }


def display_record_detail(record: Dict, show_category_override: bool = True):
    """Display detailed view of a record."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        st.write(f"**Date:** {record.get('transaction_date', 'N/A')}")
        st.write(f"**Merchant:** {record.get('merchant_raw', 'N/A')}")
        st.write(f"**Amount:** ${record.get('amount', 0):.2f}")
        st.write(f"**Source:** {record.get('source', 'N/A')}")
        st.write(f"**Confidence:** {record.get('confidence', 0):.1%}")
        
        if record.get("matched_transaction_id"):
            st.write(f"✅ **Matched to:** {record['matched_transaction_id']}")
            st.write(f"   **Score:** {record.get('match_score', 0):.2%}")
        else:
            st.write("❌ **No match found**")
    
    with col2:
        st.subheader("Evidence & OCR")
        raw_payload = record.get("raw_payload")
        if isinstance(raw_payload, str):
            raw_payload = json.loads(raw_payload)
        
        ocr_text = raw_payload.get("ocr_text", "")
        if ocr_text:
            st.text_area("OCR Text (excerpt)", ocr_text[:500], height=150, disabled=True)
        else:
            st.info("No OCR text available")
    
    if show_category_override:
        st.subheader("Categorization")
        
        # Get prediction from rules engine
        category_pred, conf = categorization.categorize_record(record["id"], DB_PATH)
        current_final = record.get("category_final")
        
        st.write(f"**Rules Prediction:** {category_pred} ({conf:.1%} confidence)")
        
        if current_final and current_final != category_pred:
            st.write(f"**Your Override:** {current_final}")
        
        categories = [
            "Uncategorized",
            "Meals & Entertainment",
            "Travel",
            "Supplies",
            "Office Equipment",
            "Utilities",
            "Rent",
            "Insurance",
            "Taxes & Licenses",
            "Other",
        ]
        
        new_category = st.selectbox(
            "Set Category:",
            categories,
            index=categories.index(current_final) if current_final and current_final in categories else categories.index(category_pred) if category_pred in categories else 0,
            key=f"cat_{record['id']}",
        )
        
        if st.button("💾 Save Category", key=f"btn_{record['id']}"):
            categorization.save_category_override(record["id"], new_category, DB_PATH)
            st.success(f"✅ Category saved: {new_category}")
            st.rerun()


# Sidebar navigation
page = st.sidebar.radio(
    "View",
    ["Dashboard", "Unmatched Receipts", "Unmatched Transactions", "Match Receipts"],
)

# ===== DASHBOARD =====
if page == "Dashboard":
    st.header("Dashboard")
    
    stats = get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", stats["total"])
    with col2:
        st.metric("Receipts", stats["total_receipts"])
    with col3:
        st.metric("Matched", stats["matched_receipts"])
    with col4:
        st.metric("Needs Review", stats["unmatched_receipts"] + stats["low_confidence"])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Auto-match All Receipts"):
            with st.spinner("Matching receipts to transactions..."):
                count = matching.match_all_receipts(DB_PATH)
            st.success(f"Matched {count} receipts!")
            st.rerun()
    
    with col2:
        if st.button("🏷️ Auto-categorize All"):
            with st.spinner("Categorizing transactions..."):
                count = categorization.categorize_all_records(DB_PATH)
            st.success(f"Categorized {count} transactions!")
            st.rerun()

# ===== UNMATCHED RECEIPTS =====
elif page == "Unmatched Receipts":
    st.header("Unmatched Receipts")
    
    unmatched = matching.list_unmatched_receipts(DB_PATH)
    
    if not unmatched:
        st.info("✅ All receipts are matched!")
    else:
        st.write(f"Found {len(unmatched)} unmatched receipts")
        
        # Display as table
        for receipt in unmatched:
            with st.expander(
                f"📄 {receipt['transaction_date']} · {receipt['merchant_raw']} · ${receipt['amount']:.2f}"
            ):
                display_record_detail(fetch_record(receipt["id"]))

# ===== UNMATCHED TRANSACTIONS =====
elif page == "Unmatched Transactions":
    st.header("Unmatched Bank Transactions")
    
    unmatched = matching.list_unmatched_transactions(DB_PATH)
    
    if not unmatched:
        st.info("✅ All bank transactions are matched!")
    else:
        st.write(f"Found {len(unmatched)} unmatched transactions")
        
        # Display as table
        for txn in unmatched:
            with st.expander(
                f"🏦 {txn['transaction_date']} · {txn['merchant_raw']} · ${txn['amount']:.2f} ({txn['source']})"
            ):
                st.write(f"Record ID: {txn['id']}")
                st.write(f"Source: {txn['source']}")
                st.info("💡 No receipt found. You can upload and match a receipt for this transaction.")

# ===== MATCH RECEIPTS =====
elif page == "Match Receipts":
    st.header("Receipt ↔ Transaction Matching")
    
    st.write("View and approve receipt matches, or manually link receipts to transactions.")
    
    receipts_with_potential = []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT * FROM canonical_records
        WHERE source = 'receipt'
        ORDER BY transaction_date DESC
        """
    )
    for row in cur.fetchall():
        receipts_with_potential.append(dict(row))
    
    conn.close()
    
    if not receipts_with_potential:
        st.info("No receipts found. Upload receipts to get started.")
    else:
        for receipt in receipts_with_potential:
            matched_id = receipt.get("matched_transaction_id")
            match_score = receipt.get("match_score", 0)
            
            status = "✅" if matched_id else "❌"
            score_str = f"({match_score:.1%})" if matched_id else "(no match)"
            
            with st.expander(f"{status} {receipt['transaction_date']} · {receipt['merchant_raw']} {score_str}"):
                display_record_detail(receipt, show_category_override=True)
