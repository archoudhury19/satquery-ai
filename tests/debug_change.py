from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.planner import build_plan, understand_query
from backend.app import analyze_change, read_image, app, FILES
from fastapi.testclient import TestClient

client = TestClient(app)

p_t1 = Path('demo_data/cdvqa/cdvqa_time1.tif')
p_t2 = Path('demo_data/cdvqa/cdvqa_time2.tif')
FILES['t1'] = {'path': p_t1, 'data': read_image(p_t1), 'filename': p_t1.name}
FILES['t2'] = {'path': p_t2, 'data': read_image(p_t2), 'filename': p_t2.name}

queries = [
    'compare the changes',
    'What changed between these two dates, and where did the change occur?',
    'Has the water area grown or shrunk since last year?',
    'Compare change',
    'compare changes',
]

print('=== TESTING WITH 2 IMAGES ===')
for q in queries:
    res = client.post('/api/analyze', json={'primary_id': 't1', 'secondary_id': 't2', 'query': q}).json()
    print(f'Query: "{q}"')
    print('  Task:', res.get('task'), 'Feature:', res.get('feature'), 'Tool:', res.get('tool'))
    print('  Answer:', res.get('answer'))
    print('  Overlay URL:', res.get('overlay_url'))
    print('  Evidence:', res.get('evidence'))
    print('-'*50)

print('\n=== TESTING WITH 1 IMAGE (SECONDARY MISSING) ===')
for q in ['compare the changes', 'What changed between these two dates, and where did the change occur?']:
    res = client.post('/api/analyze', json={'primary_id': 't1', 'query': q}).json()
    print(f'Query: "{q}"')
    print('  Task:', res.get('task'), 'Answer:', res.get('answer'))
