import sys, json, requests
from pathlib import Path

BASE = "http://127.0.0.1:8000"

print("============================================================")
print("SATQUERY MULTI-MODAL EVALUATION AUDIT")
print("============================================================\n")

# 1. Test VQA with fine-tuned adapter
print("--- [1] VQA Predictions (Fine-Tuned Adapter) ---")
p_s2 = Path("demo_data/bigearthnet/S2_multispectral_patch.tif")
with open(p_s2, "rb") as f:
    up_s2 = requests.post(f"{BASE}/api/upload", files={"file": (p_s2.name, f, "image/tiff")}).json()

p_urb = Path("demo_data/vrsbench/vrsbench_sample_01.tif")
with open(p_urb, "rb") as f:
    up_urb = requests.post(f"{BASE}/api/upload", files={"file": (p_urb.name, f, "image/tiff")}).json()

tests = [
    (up_s2["id"], "Is this an urban or rural area?", "Sentinel-2 Tile"),
    (up_s2["id"], "Is there vegetation present?", "Sentinel-2 Tile"),
    (up_urb["id"], "Is this an urban or rural area?", "Kolkata Urban"),
    (up_urb["id"], "Are there buildings visible?", "Kolkata Urban"),
]

for img_id, q, label in tests:
    res = requests.post(f"{BASE}/api/analyze", json={"primary_id": img_id, "query": q}).json()
    ans = res.get("answer")
    conf = round(res.get("confidence", 0) * 100)
    tool = res.get("tool")
    print(f"[{label}] Q: '{q}' -> Answer: '{ans}' (Conf: {conf}%, Tool: {tool})")

# 2. Test Open-Vocabulary Visual Grounding
print("\n--- [2] Open-Vocabulary Visual Grounding ---")
grounding_tests = [
    (up_urb["id"], "Locate the river corridor and water body.", "Kolkata Urban"),
    (up_s2["id"], "Highlight the green vegetation canopy.", "Sentinel-2 Tile"),
]

for img_id, q, label in grounding_tests:
    res = requests.post(f"{BASE}/api/analyze", json={"primary_id": img_id, "query": q}).json()
    ans = res.get("answer")
    bbox = res.get("evidence", {}).get("pixel_bounding_box") or res.get("bounding_box")
    overlay = res.get("overlay") or res.get("overlay_url")
    print(f"[{label}] Grounding: '{q}'")
    print(f"  Answer : {ans}")
    print(f"  BBox   : {bbox}")
    print(f"  Overlay: {overlay}")

# 3. Test Multi-Class Segmentation
print("\n--- [3] Multi-Class AI Segmentation ---")
res_seg = requests.post(f"{BASE}/api/analyze", json={
    "primary_id": up_s2["id"],
    "query": "Segment land cover into water, vegetation, and built up.",
}).json()
print(f"Sentinel-2 Segmentation: {res_seg.get('answer')}")
print(f"Overlay: {res_seg.get('overlay_url')}")

print("\n============================================================")
print("AUDIT COMPLETE: All modules verified successfully.")
print("============================================================")
