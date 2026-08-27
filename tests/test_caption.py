import requests
from pathlib import Path

BASE = "http://127.0.0.1:8000"

images = [
    ("demo_data/vrsbench/vrsbench_sample_01.tif", "Kolkata High-Res Optical"),
    ("demo_data/bigearthnet/S2_multispectral_patch.tif", "Sentinel-2 Multispectral"),
]

print("============================================================")
print("TESTING ADVANCED REMOTE-SENSING SCENE CAPTIONING")
print("============================================================\n")

for img_path, label in images:
    p = Path(img_path)
    with open(p, "rb") as f:
        up = requests.post(f"{BASE}/api/upload", files={"file": (p.name, f, "image/tiff")}).json()

    res = requests.post(f"{BASE}/api/analyze", json={
        "primary_id": up["id"],
        "query": "Describe the land-cover and major objects visible in this image.",
    }).json()

    print(f"[{label}]")
    print(f"  Tool       : {res.get('tool')}")
    print(f"  Caption    : {res.get('answer')}")
    print(f"  Confidence : {round(res.get('confidence', 0) * 100)}%")
    print(f"  Overlay    : {res.get('overlay_url')}")
    print()
