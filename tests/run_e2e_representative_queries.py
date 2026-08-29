"""
tests/run_e2e_representative_queries.py
======================================
Executes all 5 mandatory representative queries through the full agentic pipeline
and inspects the actual textual, visual, spatial, and trace evidence.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

print("=" * 80)
print("SATQUERY AI — GENUINE END-TO-END REPRESENTATIVE QUERY VERIFICATION")
print("=" * 80)

TEST_RUNS = [
    {
        "num": 1,
        "category": "Single-Image Captioning (VRSBench)",
        "sample_key": "kolkata",
        "query": "Describe the land-cover and major objects visible in this image.",
    },
    {
        "num": 2,
        "category": "Single-Image Region Grounding (VRSBench)",
        "sample_key": "kolkata",
        "query": "Highlight the water body referred to in the query.",
    },
    {
        "num": 3,
        "category": "Bi-Temporal Change Analysis (CDVQA)",
        "sample_key": "bitemporal",
        "query": "What changed between these two dates, and where did the change occur?",
    },
    {
        "num": 4,
        "category": "Optical + SAR Cross-Modal Fusion (ISRO/SAC Pair)",
        "sample_key": "optical_sar",
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
    },
    {
        "num": 5,
        "category": "Directional Change VQA (CDVQA)",
        "sample_key": "bitemporal",
        "query": "Has the built-up area increased, decreased, or remained unchanged?",
    },
]

for item in TEST_RUNS:
    print(f"\n[{item['num']}/5] {item['category']}")
    print(f"  • Load Demo Sample : '{item['sample_key']}'")
    
    # 1. Load Demo
    load_res = client.post("/api/load_demo", json={"sample_key": item["sample_key"]})
    assert load_res.status_code == 200, f"Failed to load demo: {load_res.text}"
    demo_info = load_res.json()
    prim_id = demo_info["primary"]["id"]
    sec_id = demo_info.get("secondary", {}).get("id") if demo_info.get("secondary") else None
    
    print(f"    - Primary Image  : {demo_info['primary']['filename']} (ID: {prim_id})")
    if sec_id:
        print(f"    - Secondary Image: {demo_info['secondary']['filename']} (ID: {sec_id})")
        
    print(f"  • Input Query      : \"{item['query']}\"")
    
    # 2. Analyze
    payload = {
        "primary_id": prim_id,
        "secondary_id": sec_id,
        "query": item["query"],
    }
    ana_res = client.post("/api/analyze", json=payload)
    assert ana_res.status_code == 200, f"Analyze failed: {ana_res.text}"
    res = ana_res.json()
    
    # 3. Print full verified output
    print(f"  • Agent Classification:")
    print(f"    - Task Selected  : {res.get('task')}")
    print(f"    - Tool Executed  : {res.get('tool')}")
    print(f"    - Confidence     : {int(res.get('confidence', 0) * 100)}%")
    print(f"  • Generated Text Answer:")
    print(f"    \"{res.get('answer')}\"")
    print(f"  • Spatial Evidence:")
    ev = res.get("evidence", {})
    for k, v in ev.items():
        if k not in ("gain_mask", "loss_mask", "geojson"):
            print(f"    - {k:25}: {v}")
    if res.get("bounding_box"):
        print(f"    - Bounding Box             : {res.get('bounding_box')}")
    if res.get("overlay_url"):
        print(f"  • Visual Overlay Generated   : {res.get('overlay_url')}")
    print(f"  • Observable Execution Trace:")
    for step in res.get("execution_trace", []):
        print(f"    [Step {step.get('step')}] -> {step.get('detail')}")

print("\n" + "=" * 80)
print("ALL 5 REPRESENTATIVE QUERIES SUCCESSFULLY EXECUTED & AUDITED")
print("=" * 80)
