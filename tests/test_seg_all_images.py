from __future__ import annotations
import sys, requests, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"

print("=== Segmentation on ALL benchmark images ===\n")
images = {
    "Kolkata RGB (3-band optical)":        "demo_data/vrsbench/vrsbench_sample_01.tif",
    "Sentinel-2 Multispectral (4-band T1)": "demo_data/bigearthnet/S2_multispectral_patch.tif",
    "ISRO Optical (Cartosat)":             "demo_data/isro_sac/cartosat_optical_coregistered.tif",
    "Kolkata JPEG":                        "demo_data/vrsbench/vrsbench_sample_01.jpg",
}

for label, img_path in images.items():
    p = Path(img_path)
    mime = "image/jpeg" if p.suffix == ".jpg" else "image/tiff"
    with open(p, "rb") as f:
        up = requests.post(f"{BASE}/api/upload", files={"file": (p.name, f, mime)}).json()

    res = requests.post(f"{BASE}/api/analyze", json={
        "primary_id": up["id"],
        "query": "Identify the green fields, buildings, and water in different colours.",
    }).json()

    ev = res.get("evidence", {})
    print(f"[{label}]")
    print(f"  Tool   : {res.get('tool')}")
    print(f"  Answer : {res.get('answer')}")
    print(f"  Water  : {ev.get('water_percent')}%")
    print(f"  Veg    : {ev.get('vegetation_percent')}%")
    print(f"  Built  : {ev.get('built_up_percent')}%")
    print(f"  Other  : {ev.get('unclassified_percent')}%")
    print()
