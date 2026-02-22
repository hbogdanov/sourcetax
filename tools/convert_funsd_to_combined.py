import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNSD_DIR = ROOT / 'data' / 'forms' / 'funsd'
OUT = ROOT / 'data' / 'combined_dataset' / 'combined_samples.jsonl'

def main():
    if not FUNSD_DIR.exists():
        print('FUNSD dir not found:', FUNSD_DIR)
        return
    out_lines = []
    for p in sorted(FUNSD_DIR.glob('*.json')):
        try:
            with open(p, encoding='utf-8') as f:
                data = json.load(f)
            rec = {
                'source': 'funsd',
                'image': str(p.name),
                'meta': data,
                'type': 'form'
            }
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        except Exception as e:
            print('skip', p, e)
    if not out_lines:
        print('No FUNSD json files converted')
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'a', encoding='utf-8') as f:
        for line in out_lines:
            f.write(line + '\n')
    print('Appended', len(out_lines), 'FUNSD records to', OUT)

if __name__ == '__main__':
    main()
