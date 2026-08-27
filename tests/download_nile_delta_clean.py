"""
Download authentic Nile Delta Rosetta River mouth (S2A_36RTV_20240729_1_L2A).
"""
import rasterio
from rasterio.windows import Window
from pathlib import Path
from rasterio.enums import Resampling

OUT_DIR = Path("demo_data/edge_cases")
out_path = OUT_DIR / "ec_river_delta.tif"

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = ["red", "green", "blue", "nir"]

import urllib.request, json
payload = json.dumps({
    "collections": ["sentinel-2-l2a"],
    "bbox": [30.3, 31.3, 30.6, 31.6],
    "datetime": "2024-07-01T00:00:00Z/2024-08-31T23:59:59Z",
    "limit": 5
}).encode()
req = urllib.request.Request(STAC_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())
feats = data.get("features", [])
feats.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))
item = feats[0]
print(f"Downloading Nile Delta scene: {item.get('id')} (Cloud: {item.get('properties',{}).get('eo:cloud_cover')}%)")

assets = item.get("assets", {})
arrays = []
meta = None
transform = None

# Center on Rosetta mouth: col=5484, row=1768
win = Window(5484 - 256, 1768 - 256, 512, 512)

with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
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
    "driver": "GTiff",
})
with rasterio.open(out_path, "w", **meta) as dst:
    for i, arr in enumerate(arrays, start=1):
        dst.write(arr, i)

print(f"SUCCESS: Written authentic Nile Delta GeoTIFF -> {out_path} ({out_path.stat().st_size // 1024} KB)")
