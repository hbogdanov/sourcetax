import json
from collections import Counter
from pathlib import Path

P = Path('data/combined_dataset/combined_samples.jsonl')
if not P.exists():
    print('combined_samples.jsonl not found at', P)
    raise SystemExit(1)

counts = Counter()
total = 0
with P.open(encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
            counts[obj.get('source', 'unknown')] += 1
        except Exception:
            counts['invalid'] += 1

print('total_lines:', total)
for k, v in counts.items():
    print(k, v)
