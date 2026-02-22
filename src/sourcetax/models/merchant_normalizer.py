"""
Rule-based merchant normalization.

Cleans raw merchant strings into canonical form.
Handles:
- Whitespace, case, punctuation
- POS codes (SQ *), location codes (NYC, CA)
- Phone numbers, store IDs
- Common aliases

Quick wins:
- "SQ *STARBUCKS" → "Starbucks"
- "UBER *TRIP" → "Uber"
- "AMZN MKTP MERCHANT" → "Amazon"
"""

import re
from typing import Dict, Tuple, Optional


# Common POS/vendor prefixes to strip
JUNK_PREFIXES = {
    "SQ", "POS", "POS\\*", "SQUARE",
    "PAYPAL", "PAYPAL\\*",
    "STRIPE",
    "HELP\\.", "SUPPORT\\.",
}

# Common suffixes to strip
JUNK_SUFFIXES = {
    "CA", "NY", "TX", "FL",  # States
    "COM", "ORG", "NET",      # TLDs
    "HELP", "SUPPORT", "INFO", "PHONE", "BILLING",
}

# Merchant aliases: raw → canonical
MERCHANT_ALIASES = {
    "STARBUCKS": "Starbucks",
    "SBUX": "Starbucks",
    "AMAZON": "Amazon",
    "AMZN": "Amazon",
    "AMZN MKTP": "Amazon",
    "WHOLE FOODS": "Whole Foods",
    "WHOLEFDS": "Whole Foods",
    "TRADER JOES": "Trader Joe's",
    "TRADER JOS": "Trader Joe's",
    "COSTCO": "Costco",
    "UBER": "Uber",
    "UBER EATS": "Uber Eats",
    "LYFT": "Lyft",
    "SHELL": "Shell",
    "CHEVRON": "Chevron",
    "EXXON": "Exxon",
    "HOTEL MARRIOTT": "Marriott",
    "MARRIOTT": "Marriott",
    "HILTON": "Hilton",
    "HYATT": "Hyatt",
    "AIRBNB": "Airbnb",
    "SPOTIFY": "Spotify",
    "NETFLIX": "Netflix",
    "APPLE": "Apple",
    "GOOGLE": "Google",
    "MICROSOFT": "Microsoft",
    "ADOBE": "Adobe",
    "HOME DEPOT": "Home Depot",
    "LOWES": "Lowes",
    "BEST BUY": "Best Buy",
    "OFFICE DEPOT": "Office Depot",
    "STAPLES": "Staples",
}

# Charge amount patterns (POS codes like "1234", phone-like tokens)
CARD_AMOUNT_PATTERN = re.compile(r"[\d\s\-\(\)]{5,}")  # Card number, zip
STORE_ID_PATTERN = re.compile(r"#\d{4,}|\d{4,}[AB]")  # Store IDs like #1234 or 5678A


def clean_merchant_text(merchant_raw: str) -> str:
    """
    Clean raw merchant string.
    
    Steps:
    1. Uppercase
    2. Strip punctuation (except spaces)
    3. Remove junk tokens
    4. Remove phone/card numbers
    5. Collapse multiple spaces
    """
    if not merchant_raw:
        return ""
    
    text = merchant_raw.upper().strip()
    
    # Remove common punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove phone numbers and card patterns
    text = re.sub(CARD_AMOUNT_PATTERN, " ", text)
    text = re.sub(STORE_ID_PATTERN, " ", text)
    
    # Remove known junk
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in JUNK_PREFIXES and t not in JUNK_SUFFIXES]
    
    # Remove common location suffixes
    if tokens and tokens[-1] in {"CA", "NY", "TX", "FL", "IL", "PA", "OH"}:
        tokens = tokens[:-1]
    
    return " ".join(tokens)


def apply_aliases(clean_text: str) -> str:
    """Apply merchant aliases to standardize brand names."""
    for alias, canonical in sorted(MERCHANT_ALIASES.items(), key=lambda x: -len(x[0])):
        # Match whole alias in cleaned text
        if alias in clean_text:
            # Replace and return first match
            return canonical
    
    return clean_text


def extract_root_merchant(clean_text: str, top_n: int = 2) -> str:
    """
    Extract first N tokens as "root" merchant name.
    
    Examples:
    "STARBUCKS COFFEE 123" → "STARBUCKS COFFEE" (top_n=2)
    "UBER TRIP SFO" → "UBER TRIP"
    """
    tokens = clean_text.split()[:top_n]
    return " ".join(tokens) if tokens else "UNKNOWN"


def normalize_merchant(
    merchant_raw: str,
) -> Tuple[str, str, Optional[str]]:
    """
    Normalize a raw merchant string.
    
    Args:
        merchant_raw: Raw merchant string from transaction
    
    Returns:
        (merchant_clean, merchant_root, merchant_brand)
        
        - merchant_clean: Cleaned version of raw
        - merchant_root: First 2 tokens (category signal)
        - merchant_brand: Canonical brand if matched, else None
    """
    if not merchant_raw:
        return "", "", None
    
    # Step 1: Clean junk
    clean = clean_merchant_text(merchant_raw)
    
    # Step 2: Apply aliases
    aliased = apply_aliases(clean)
    
    # Step 3: Extract root
    root = extract_root_merchant(clean)
    
    # Check if we found an alias
    brand = None
    if aliased != clean:  # An alias was applied
        brand = aliased
    
    return clean, root, brand


def main():
    """Test merchant normalization."""
    test_cases = [
        "SQ *STARBUCKS COFFEE 123 SF CA",
        "UBER *TRIP HELP.UBER.COM",
        "PAYPAL *SPOTIFY PREMIUM",
        "AMZN MKTP MERCHANT LLC",
        "WHOLEFDS MKT 10234 NYC NY",
        "CHEVRON GAS #1234 LOS ANGELES CA",
        "SHELL OIL STATION 5678 TX",
    ]
    
    print("=" * 80)
    print("MERCHANT NORMALIZATION TEST")
    print("=" * 80)
    
    for raw in test_cases:
        clean, root, brand = normalize_merchant(raw)
        print(f"\nRaw:   {raw}")
        print(f"Clean: {clean}")
        print(f"Root:  {root}")
        print(f"Brand: {brand}")


if __name__ == "__main__":
    main()
