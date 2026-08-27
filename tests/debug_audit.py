import sys, json, requests
from pathlib import Path
sys.path.insert(0, '.')
BASE = 'http://127.0.0.1:8000'

images = [
    ('demo_data/vrsbench/vrsbench_sample_01.tif', 'Kolkata urban'),
    ('demo_data/bigearthnet/S2_multispectral_patch.tif', 'Sentinel-2 fields'),
    ('demo_data/vrsbench/vrsbench_sample_01.jpg', 'Kolkata JPEG'),
]

for img_path, label in images:
    p = Path(img_path)
    mime = 'image/jpeg' if p.suffix == '.jpg' else 'image/tiff'
    with open(p, 'rb') as f:
        up = requests.post(f'{BASE}/api/upload', files={'file': (p.name, f, mime)}).json()
    res = requests.post(f'{BASE}/api/analyze', json={
        'primary_id': up['id'],
        'query': 'Describe the land-cover and major objects visible in this image.'
    }).json()
    ans = res.get('answer')
    print(f'[{label}] Caption: {ans}')

# VQA variability check
p = Path('demo_data/bigearthnet/S2_multispectral_patch.tif')
with open(p, 'rb') as f:
    up = requests.post(f'{BASE}/api/upload', files={'file': (p.name, f, 'image/tiff')}).json()

for q in ['Is there a river?', 'Is this area urban or rural?', 'Is there water present?', 'How many buildings are there?']:
    res = requests.post(f'{BASE}/api/analyze', json={'primary_id': up['id'], 'query': q}).json()
    ans = res.get('answer')
    conf = round(res.get('confidence', 0)*100)
    print(f'VQA [{q}] -> answer:{ans} conf:{conf}%')
