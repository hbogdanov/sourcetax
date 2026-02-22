"""
Receipt ingestion: OCR, field extraction, normalization.

Phase 2.1: Extract merchant, date, total, tax, tip from receipt images/PDFs.
"""

import re
from pathlib import Path
from typing import Dict, Optional
import hashlib

try:
    import pytesseract
    from PIL import Image, ImageOps, ImageEnhance
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# EasyOCR is optional and must be imported lazily inside the function
HAS_EASYOCR = False


def ocr_image_tesseract(image_path: Path) -> str:
    """Extract text from image using Tesseract.

    Applies lightweight preprocessing (grayscale, contrast) for more stable OCR.
    Requires: pip install pytesseract pillow + system tesseract.
    """
    if not HAS_TESSERACT:
        raise ImportError("pytesseract/pillow not installed. Run: pip install pytesseract pillow")

    img = Image.open(image_path)
    # Preprocess: convert to grayscale, autocontrast, enhance
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    try:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
    except Exception:
        pass

    text = pytesseract.image_to_string(img)
    return text


def ocr_image_easyocr(image_path: Path) -> str:
    """Extract text from image using EasyOCR (lazy import).

    EasyOCR/torch can be heavy; importing only when requested avoids import-time failures.
    """
    try:
        import easyocr  # type: ignore
        from PIL import Image, ImageOps
    except Exception as e:
        raise ImportError("easyocr not available. Install optional extras: pip install -e .[ocr] or pip install easyocr") from e

    # Preprocess image similarly
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(str(image_path))
    text = '\n'.join([item[1] for item in result])
    return text


def extract_ocr_text(receipt_path: Path, method: str = "tesseract") -> str:
    """Extract text from receipt image. Falls back if primary method unavailable."""
    if method == "tesseract":
        try:
            return ocr_image_tesseract(receipt_path)
        except ImportError:
            if HAS_EASYOCR:
                return ocr_image_easyocr(receipt_path)
            raise
    elif method == "easyocr":
        try:
            return ocr_image_easyocr(receipt_path)
        except ImportError:
            if HAS_TESSERACT:
                return ocr_image_tesseract(receipt_path)
            raise
    else:
        raise ValueError(f"Unknown OCR method: {method}")


def extract_date(text: str) -> Optional[str]:
    """Extract date from receipt text (YYYY-MM-DD or MM/DD/YYYY)."""
    if not text:
        return None
    
    # Try ISO format first
    match = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', text)
    if match:
        return match.group(0)
    
    # Try MM/DD/YYYY
    match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', text)
    if match:
        m, d, y = match.groups()
        # Assume 20xx if 2-digit year
        if len(y) == 2:
            y = f"20{y}"
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    
    return None


def extract_total(text: str) -> Optional[float]:
    """Extract total/amount from receipt (looks for TOTAL, AMOUNT, BALANCE, GRAND TOTAL)."""
    if not text:
        return None
    
    lines = text.split('\n')
    for line in lines:
        # Look for TOTAL, AMOUNT, BALANCE, GRAND TOTAL (case-insensitive)
        if re.search(r'\b(TOTAL|AMOUNT|BALANCE|GRAND\s+TOTAL)\b', line, re.IGNORECASE):
            # Extract number from this line or next
            match = re.search(r'\$?\s*(\d+[.,]\d{2})', line)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
    
    return None


def extract_tax(text: str) -> Optional[float]:
    """Extract tax amount from receipt."""
    if not text:
        return None
    
    lines = text.split('\n')
    for line in lines:
        if re.search(r'\b(TAX|GST|SALES\s+TAX)\b', line, re.IGNORECASE):
            match = re.search(r'\$?\s*(\d+[.,]\d{2})', line)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
    
    return None


def extract_tip(text: str) -> Optional[float]:
    """Extract tip amount from receipt."""
    if not text:
        return None
    
    lines = text.split('\n')
    for line in lines:
        if re.search(r'\b(TIP|GRATUITY)\b', line, re.IGNORECASE):
            match = re.search(r'\$?\s*(\d+[.,]\d{2})', line)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
    
    return None


def extract_merchant(text: str) -> Optional[str]:
    """Extract merchant name (top lines, biggest text, or first non-empty lines)."""
    if not text:
        return None
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    # Take first non-empty line (often merchant name)
    for line in lines[:5]:
        # Skip lines that are all numbers or too short
        if len(line) > 3 and not line.isdigit():
            return line
    
    return None


def parse_receipt_text(ocr_text: str) -> Dict[str, Optional[float | str]]:
    """Extract structured fields from OCR text."""
    return {
        "date": extract_date(ocr_text),
        "merchant": extract_merchant(ocr_text),
        "total": extract_total(ocr_text),
        "tax": extract_tax(ocr_text),
        "tip": extract_tip(ocr_text),
        "full_text": ocr_text,
    }


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file for deduplication."""
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        sha.update(f.read())
    return sha.hexdigest()


def ingest_receipt(
    receipt_path: Path,
    upload_dir: Path = Path("data/uploads"),
    ocr_method: str = "tesseract",
) -> Dict:
    """
    Ingest a receipt file:
    1. Hash for dedup
    2. Extract OCR text
    3. Parse fields (date, merchant, total, tax, tip)
    4. Store in upload_dir
    
    Returns dict suitable for canonicalization.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute hash
    file_hash = hash_file(receipt_path)
    
    # Extract OCR text
    ocr_text = extract_ocr_text(receipt_path, method=ocr_method)
    
    # Parse fields
    fields = parse_receipt_text(ocr_text)
    
    # Store original (optional, for now just hash + fields)
    # Later: copy to data/uploads/{hash}.{ext}
    
    return {
        "file_hash": file_hash,
        "original_name": receipt_path.name,
        "extracted_fields": fields,
        "ocr_full_text": ocr_text,
    }
