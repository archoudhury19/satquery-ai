"""
Fetch real Mississippi River Delta or Ganges Delta Sentinel-2 scene.
"""
import sys
import json
import urllib.request
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT_DIR = Path(__file__).resolve().parent.parent / "demo_data" / "edge_cases"
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = ["red", "green", "blue", "nir"]

payload = json.dumps({
    "collections": ["sentinel-2-l2a"],
    "bbox": [-89.5, 29.2, -89.1, 29.6],  # Mississippi Delta / Gulf Coast
    "datetime": "2024-05-01T00:00:00Z/2024-08-31T23:59:59Z",
    "limit": 5,
}).encode()
req = urllib.request.Request(STAC_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())
feats = data.get("features", [])
feats.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))
item = feats[0]
print("Found item:", item.get("id"))

out_path = OUT_DIR / "ec_river_delta.tif"
assets = item.get("assets", {})
arrays = []
meta = None
transform = None

with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
    red_href = assets["red"]["href"]
    with rasterio.open(red_href) as src:
        # Read a window that has non-zero values
        for test_y in [src.height // 2, src.height // 3, 2000, 3000]:
            for test_x in [src.width // 2, src.width // 3, 2000, 3000]:
                test_win = Window(test_x, test_y, 512, 512)
                sample = src.read(1, window=test_win)
                if sample.max() > 100:
                    win = test_win
                    print(f"Selected valid non-empty window at col={test_x}, row={test_y}, max={sample.max()}")
                    break
            if 'win' in locals():
                break

    for b in BANDS:
        href = assets[b]["href"]
        with rasterio.open(href) as src:
            data = src.read(1, window=win, resampling=Resampling.nearest)
            if meta is None:
                meta = src.meta.copy()
                transform = src.window_transform(win)
            arrays.append(data)
            print(f"  {b}: shape={data.shape}, min={data.min()}, max={data.max()}")

meta.update({
    "count": len(arrays),
    "width": 512,
    "height": 512,
    "transform": transform,
    "compress": "deflate",
})
with rasterio.open(out_path, "w", **meta) as dst:
    for i, arr in enumerate(arrays, start=1):
        dst.write(arr, i)
print(f"SAVED -> {out_path.name} ({out_path.stat().st_size // 1024} KB)")
