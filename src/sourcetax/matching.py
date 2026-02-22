"""
Receipt ↔ Bank transaction matching.

Phase 2.2: Auto-link receipts to bank transactions with confidence scores.
"""

import sqlite3
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import difflib
from .normalization import normalize_merchant_name


def normalize_merchant(merchant: str) -> str:
    """Normalize merchant name for fuzzy matching.

    Delegates to the shared normalization helper to ensure consistent behavior
    across the codebase.
    """
    return normalize_merchant_name(merchant, case="lower")


def date_closeness_score(date1: str, date2: str, max_days: int = 3) -> float:
    """Score dates based on closeness (0-1). Max diff = max_days."""
    try:
        d1 = datetime.fromisoformat(date1).date()
        d2 = datetime.fromisoformat(date2).date()
        diff_days = abs((d1 - d2).days)
        if diff_days > max_days:
            return 0.0
        return 1.0 - (diff_days / max_days)
    except:
        return 0.0


def amount_closeness_score(receipt_amount: float, transaction_amount: float, tolerance: float = 10.0) -> float:
    """Score amounts based on closeness (0-1). Tolerance allows for tax/tip variance."""
    if receipt_amount is None or transaction_amount is None:
        return 0.0
    
    abs_diff = abs(receipt_amount - transaction_amount)
    if abs_diff > tolerance:
        return 0.0
    return 1.0 - (abs_diff / tolerance)


def merchant_similarity_score(merchant1: str, merchant2: str, min_ratio: float = 0.8) -> float:
    """Score merchant similarity using fuzzy matching (0-1)."""
    if not merchant1 or not merchant2:
        return 0.0
    
    norm1 = normalize_merchant(merchant1)
    norm2 = normalize_merchant(merchant2)
    
    if not norm1 or not norm2:
        return 0.0
    
    ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    
    if ratio < min_ratio:
        return 0.0
    
    return ratio


def match_receipt_to_bank(
    receipt_record: Dict,
    bank_records: List[Dict],
    date_weight: float = 0.3,
    amount_weight: float = 0.5,
    merchant_weight: float = 0.2,
    score_threshold: float = 0.75,
) -> Tuple[Optional[str], float]:
    """
    Match a single receipt to bank transactions.
    
    Returns: (matched_transaction_id, score) or (None, best_score_if_below_threshold)
    """
    if not receipt_record or not bank_records:
        return None, 0.0
    
    receipt_date = receipt_record.get("transaction_date")
    receipt_amount = receipt_record.get("amount")
    receipt_merchant = receipt_record.get("merchant_raw")
    
    best_match_id = None
    best_score = 0.0
    
    for bank_rec in bank_records:
        bank_date = bank_rec.get("transaction_date")
        bank_amount = bank_rec.get("amount")
        bank_merchant = bank_rec.get("merchant_raw")
        
        # Date score
        d_score = date_closeness_score(receipt_date, bank_date, max_days=3)
        if d_score == 0:
            continue  # Different dates, skip
        
        # Amount score (allow ±$10 for tax/tip tolerance)
        a_score = amount_closeness_score(receipt_amount, bank_amount, tolerance=10.0)
        if a_score == 0:
            continue  # Amounts too different, skip
        
        # Merchant score
        m_score = merchant_similarity_score(receipt_merchant, bank_merchant, min_ratio=0.7)
        
        # Weighted combined score
        combined = (d_score * date_weight + a_score * amount_weight + m_score * merchant_weight)
        
        if combined > best_score:
            best_score = combined
            best_match_id = bank_rec.get("id")
    
    if best_score >= score_threshold:
        return best_match_id, best_score
    
    return None, best_score


def match_all_receipts(db_path: str = "data/store.db") -> int:
    """
    Match all receipt records to bank transactions.
    Updates canonical_records with matched_transaction_id + match_score.
    
    Returns: count of newly matched receipts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Fetch all receipts
    cur.execute("""
        SELECT id, transaction_date, amount, merchant_raw
        FROM canonical_records
        WHERE source = 'receipt'
    """)
    receipts = [dict(row) for row in cur.fetchall()]
    
    # Fetch all bank transactions
    cur.execute("""
        SELECT id, transaction_date, amount, merchant_raw
        FROM canonical_records
        WHERE source IN ('bank', 'toast', 'quickbooks')
    """)
    bank_records = [dict(row) for row in cur.fetchall()]
    
    # Match each receipt
    matched_count = 0
    for receipt in receipts:
        match_id, match_score = match_receipt_to_bank(receipt, bank_records)
        
        if match_id:
            cur.execute("""
                UPDATE canonical_records
                SET matched_transaction_id = ?, match_score = ?
                WHERE id = ?
            """, (match_id, match_score, receipt["id"]))
            matched_count += 1
    
    conn.commit()
    conn.close()
    
    return matched_count


def list_unmatched_receipts(db_path: str = "data/store.db") -> List[Dict]:
    """Fetch all receipts without a matched transaction."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, transaction_date, merchant_raw, amount, confidence
        FROM canonical_records
        WHERE source = 'receipt' AND matched_transaction_id IS NULL
        ORDER BY transaction_date DESC
    """)
    
    results = [dict(row) for row in cur.fetchall()]
    conn.close()
    return results


def list_unmatched_transactions(db_path: str = "data/store.db") -> List[Dict]:
    """Fetch all bank transactions without matched receipts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, transaction_date, merchant_raw, amount, source
        FROM canonical_records
        WHERE source IN ('bank', 'toast', 'quickbooks')
        AND id NOT IN (
            SELECT matched_transaction_id FROM canonical_records
            WHERE matched_transaction_id IS NOT NULL
        )
        ORDER BY transaction_date DESC
    """)
    
    results = [dict(row) for row in cur.fetchall()]
    conn.close()
    return results
