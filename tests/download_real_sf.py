"""
download_real_san_francisco.py
==============================
Downloads an authentic, real-life Sentinel-2 L2A multi-spectral scene of
San Francisco Bay (Downtown SF, Golden Gate Park, Bay Bridge, and SF Bay water).
Data Source: ESA Copernicus / AWS Open Data STAC (sentinel-cogs)
"""
import json
import urllib.request
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from pyproj import Transformer

OUT_DIR = Path(__file__).resolve().parent.parent / "demo_data" / "real_world_satellite"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "real_san_francisco_optical.tif"

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = ["red", "green", "blue", "nir"]

print("Searching authentic Sentinel-2 L2A scene for San Francisco Bay (37.77 N, -122.42 W) ...")
payload = json.dumps({
    "collections": ["sentinel-2-l2a"],
    "bbox": [-122.52, 37.70, -122.35, 37.82],  # San Francisco Peninsula & Bay
    "datetime": "2024-05-01T00:00:00Z/2024-08-31T23:59:59Z",
    "limit": 5,
}).encode()

req = urllib.request.Request(STAC_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())

feats = data.get("features", [])
feats.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))
item = feats[0]
print(f"Found SF Sentinel-2 Scene: {item.get('id')} (Cloud: {item.get('properties',{}).get('eo:cloud_cover')}%)")

assets = item.get("assets", {})
arrays = []
meta = None
transform = None

with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
    red_href = assets["red"]["href"]
    with rasterio.open(red_href) as src:
        trans = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        # Center on Downtown San Francisco (Market St / Embarcadero / Bay)
        px_x, px_y = trans.transform(-122.4194, 37.7749)
        row, col = src.index(px_x, px_y)
        print(f"Downtown SF in {item.get('id')}: col={col}, row={row}")
        
        # 512x512 window capturing Downtown SF + Bay water + waterfront
        win = Window(col - 256, row - 256, 512, 512)

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

print(f"SUCCESS: Written authentic San Francisco Bay GeoTIFF -> {out_path} ({out_path.stat().st_size // 1024} KB)")
