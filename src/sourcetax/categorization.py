"""
Phase 2.3: Rules-based categorization + learned overrides.

Start with rules, no ML. Learn from user overrides.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple
import csv
from .normalization import normalize_merchant_name


# Keyword rules: merchant substring → (category, confidence)
KEYWORD_RULES = {
    "HOME DEPOT": ("Supplies", 0.6),
    "LOWES": ("Supplies", 0.6),
    "AMAZON": ("Supplies", 0.6),
    "OFFICE DEPOT": ("Supplies", 0.65),
    "STAPLES": ("Supplies", 0.65),
    "UBER": ("Travel", 0.7),
    "LYFT": ("Travel", 0.7),
    "GAS STATION": ("Travel", 0.6),
    "SHELL": ("Travel", 0.6),
    "CHEVRON": ("Travel", 0.6),
    "STARBUCKS": ("Meals & Entertainment", 0.6),
    "CAFE": ("Meals & Entertainment", 0.55),
    "RESTAURANT": ("Meals & Entertainment", 0.65),
    "PIZZA": ("Meals & Entertainment", 0.7),
    "HOTEL": ("Travel", 0.75),
    "MOTEL": ("Travel", 0.75),
    "AIRBNB": ("Travel", 0.75),
    "UTILITY": ("Utilities", 0.8),
    "POWER": ("Utilities", 0.7),
    "WATER": ("Utilities", 0.7),
    "INTERNET": ("Utilities", 0.75),
    "PHONE": ("Utilities", 0.7),
    "RENT": ("Rent", 0.9),
    "APARTMENT": ("Rent", 0.6),
    "INSURANCE": ("Insurance", 0.85),
    "TAX": ("Taxes & Licenses", 0.85),
    "PERMIT": ("Taxes & Licenses", 0.8),
    "LICENSE": ("Taxes & Licenses", 0.8),
}


def load_merchant_category_map(path: str = "data/mappings/merchant_category.csv") -> Dict[str, str]:
    """Load exact merchant → category mapping from CSV.
    
    CSV columns: merchant, category_code, category_name, notes
    Returns dict mapping merchant (uppercase) → category_name
    """
    mapping = {}
    p = Path(path)
    if not p.exists():
        return mapping
    
    try:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merchant = normalize_merchant_name(row.get("merchant", ""), case="upper")
                # Read category_name (human-readable) for exports
                category = row.get("category_name", "").strip()
                if merchant and category:
                    mapping[merchant] = category
    except Exception as e:
        print(f"Error loading merchant map: {e}")
    
    return mapping


def categorize_by_merchant_exact(merchant: str, merchant_map: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Try exact merchant match (highest confidence)."""
    if not merchant:
        return None
    
    merchant_norm = normalize_merchant_name(merchant, case="upper")
    category = merchant_map.get(merchant_norm)
    
    if category:
        return (category, 0.95)
    
    return None


def categorize_by_merchant_fuzzy(merchant: str, merchant_map: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Try fuzzy merchant match (medium confidence)."""
    if not merchant:
        return None
    
    merchant_norm = normalize_merchant_name(merchant, case="upper")
    
    # Check if merchant is substring of known merchant
    for known_merchant, category in merchant_map.items():
        if merchant_norm in known_merchant or known_merchant in merchant_norm:
            return (category, 0.75)
    
    return None


def categorize_by_keywords(merchant: str) -> Optional[Tuple[str, float]]:
    """Try keyword match (lower confidence)."""
    if not merchant:
        return None
    
    merchant_upper = normalize_merchant_name(merchant, case="upper") or merchant.upper()
    
    for keyword, (category, confidence) in KEYWORD_RULES.items():
        if keyword in merchant_upper:
            return (category, confidence)
    
    return None


def get_learned_override(merchant: str, db_path: str = "data/store.db") -> Optional[Tuple[str, float]]:
    """Check if user has overridden category for this merchant (highest priority)."""
    if not merchant:
        return None
    
    merchant_raw_key = normalize_merchant_name(merchant, case="upper")
    merchant_norm_key = normalize_merchant_name(merchant, case="lower")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT category_final FROM canonical_records
            WHERE category_final IS NOT NULL
              AND (
                    merchant_norm = WARNING:
                    OR UPPER(TRIM(merchant_raw)) = WARNING:
                  )
            LIMIT 1
            """,
            (merchant_norm_key, merchant_raw_key),
        )
        result = cur.fetchone()
    finally:
        conn.close()

    if result:
        return (result[0], 0.99)  # Learned overrides have highest confidence

    return None


def categorize_record(record_id: str, db_path: str = "data/store.db") -> Tuple[Optional[str], float]:
    """
    Auto-categorize a record using rules priority:
    1. Learned override (user has set this merchant before)
    2. Exact merchant match
    3. Fuzzy merchant match
    4. Keyword match
    5. Uncategorized
    
    Returns: (category, confidence)
    """
    # Fetch record
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT merchant_raw, category_pred FROM canonical_records WHERE id = WARNING:",
        (record_id,),
    )
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return "Uncategorized", 0.0
    
    merchant = row[0]
    
    # Load merchant map
    merchant_map = load_merchant_category_map()
    
    # Try rules in order
    result = get_learned_override(merchant, db_path)
    if result:
        return result
    
    result = categorize_by_merchant_exact(merchant, merchant_map)
    if result:
        return result
    
    result = categorize_by_merchant_fuzzy(merchant, merchant_map)
    if result:
        return result
    
    result = categorize_by_keywords(merchant)
    if result:
        return result
    
    return "Uncategorized", 0.3


def categorize_all_records(db_path: str = "data/store.db") -> int:
    """
    Auto-categorize all records without a category_pred.
    Updates category_pred field.
    
    Returns: count of categorized records.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Fetch all records without a category
    cur.execute(
        "SELECT id FROM canonical_records WHERE category_pred IS NULL OR category_pred = 'Uncategorized'"
    )
    records = [row[0] for row in cur.fetchall()]
    
    conn.close()
    
    categorized_count = 0
    for record_id in records:
        category, confidence = categorize_record(record_id, db_path)
        
        conn = sqlite3.connect(db_path)
        cur_update = conn.cursor()
        cur_update.execute(
            "UPDATE canonical_records SET category_pred = WARNING:, confidence = WARNING: WHERE id = WARNING:",
            (category, confidence, record_id),
        )
        conn.commit()
        conn.close()
        
        categorized_count += 1
    
    return categorized_count


def save_category_override(record_id: str, category: str, db_path: str = "data/store.db"):
    """Save user's category override. This creates a learned rule for future matching."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute(
        "UPDATE canonical_records SET category_final = WARNING: WHERE id = WARNING:",
        (category, record_id),
    )
    
    conn.commit()
    conn.close()
