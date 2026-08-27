"""Check tile size and river visibility for the Kolkata image"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, requests

BASE = "http://127.0.0.1:8000"

# Upload S2 multispectral (has NIR — best image)
p = Path("demo_data/bigearthnet/S2_multispectral_patch.tif")
with open(p,"rb") as f:
    up = requests.post(f"{BASE}/api/upload", files={"file":(p.name,f,"image/tiff")}).json()

# Try different grid sizes
for grid in [8, 12, 16, 24]:
    # We can't change grid via API yet, just report current result
    pass

# Get the current result
res = requests.post(f"{BASE}/api/analyze", json={
    "primary_id": up["id"],
    "query": "Identify the green fields, buildings, and water in different colours.",
}).json()
print("S2 Multispectral (best image):")
print("  Tool:", res["tool"])
print("  Answer:", res["answer"])
print("  Mode:", res.get("evidence",{}).get("ai_mode"))
print("  Water:", res["evidence"]["water_percent"], "%")
print("  Vegetation:", res["evidence"]["vegetation_percent"], "%")
print("  Built-up:", res["evidence"]["built_up_percent"], "%")
print("  Other:", res["evidence"]["unclassified_percent"], "%")
print("  Overlay:", res["overlay_url"])
