import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from difflib import get_close_matches

BASE = Path(__file__).resolve().parents[2] / 'data'
TAXONOMY_PATH = BASE / 'taxonomy' / 'schedule_c_taxonomy.json'
MERCHANT_MAP_PATH = BASE / 'mappings' / 'merchant_category.csv'


def load_taxonomy(path: Optional[str] = None) -> List[Dict]:
    p = Path(path) if path else TAXONOMY_PATH
    with p.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def load_merchant_map(path: Optional[str] = None) -> Dict[str, Dict]:
    p = Path(path) if path else MERCHANT_MAP_PATH
    result = {}
    with p.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            key = r['merchant'].strip().upper()
            result[key] = {
                'category_code': r['category_code'],
                'category_name': r['category_name'],
                'notes': r.get('notes', '')
            }
    return result


def merchant_lookup(merchant_name: str, merchant_map: Optional[Dict] = None) -> Optional[Dict]:
    """Lookup merchant with exact match, substring match, and fuzzy matching fallback."""
    if not merchant_name:
        return None
    if merchant_map is None:
        merchant_map = load_merchant_map()
    
    key = merchant_name.strip().upper()
    
    # 1. exact match first
    if key in merchant_map:
        return merchant_map[key]
    
    # 2. substring match (merchant_map key is contained in merchant_name)
    for mk in merchant_map.keys():
        if mk in key:
            return merchant_map[mk]
    
    # 3. fuzzy match using difflib (finds >80% similarity)
    all_keys = list(merchant_map.keys())
    fuzzy_matches = get_close_matches(key, all_keys, n=1, cutoff=0.80)
    if fuzzy_matches:
        return merchant_map[fuzzy_matches[0]]
    
    return None

if __name__ == '__main__':
    tax = load_taxonomy()
    print('Loaded taxonomy entries:', len(tax))
    mm = load_merchant_map()
    print('Loaded merchant mappings:', len(mm))
    print('Lookup COFFEE SHOP ->', merchant_lookup('Coffee Shop', mm))

