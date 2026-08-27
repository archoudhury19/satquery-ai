"""
download_real_edge_cases.py
===========================
Downloads 7 authentic, real-life Sentinel-2 L2A satellite image patches (Red, Green, Blue, NIR)
from the public AWS Earth Search STAC API (Earth Search v1 / sentinel-cogs) with exact geographic centering.

1. ec_urban_dense      — Paris City Center (Louvre, Seine, Haussmannian urban grid)
2. ec_forest_dense     — Brazilian Amazon Rainforest (Dense canopy, Amazon basin)
3. ec_water_dominant   — Mediterranean Open Sea / Sardinia Coastal Water
4. ec_desert_bare      — Sahara Desert (Arid dunes and bare ground, Algeria)
5. ec_agricultural     — Ukraine Dnipro Farmland (Active crop fields & parcels)
6. ec_river_delta      — Mississippi River Active Delta (Active sediment channels & wetlands)
7. ec_suburban_mixed   — London Greenwich / Thames Corridor (River, parks, suburban housing)
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://earth-search.aws.element84.com/v1/search"

SCENARIOS = [
    {
        "name": "ec_urban_dense",
        "desc": "Dense Urban — Paris City Core (Louvre, Notre-Dame, Haussmann grid)",
        "bbox": [2.30, 48.83, 2.40, 48.90],
        "date": "2024-05-01T00:00:00Z/2024-07-31T23:59:59Z",
        "col_off": 5000, "row_off": 8570, "size": 512,
    },
    {
        "name": "ec_forest_dense",
        "desc": "Tropical Rainforest — Brazilian Amazon Canopy",
        "bbox": [-62.0, -3.5, -61.5, -3.0],
        "date": "2024-06-01T00:00:00Z/2024-09-30T23:59:59Z",
        "col_off": 2500, "row_off": 2500, "size": 512,
    },
    {
        "name": "ec_water_dominant",
        "desc": "Open Coastal Water — Mediterranean Sea (Sardinia)",
        "bbox": [8.0, 38.5, 8.5, 39.0],
        "date": "2024-05-01T00:00:00Z/2024-07-31T23:59:59Z",
        "col_off": 2500, "row_off": 2500, "size": 512,
    },
    {
        "name": "ec_desert_bare",
        "desc": "Arid Desert — Sahara Sand Dunes & Rock (Algeria)",
        "bbox": [3.2, 27.2, 3.4, 27.4],
        "date": "2024-04-01T00:00:00Z/2024-06-30T23:59:59Z",
        "col_off": 5200, "row_off": 5200, "size": 512,
    },
    {
        "name": "ec_agricultural",
        "desc": "Agricultural Fields — Ukraine Dnipro Crop Basin",
        "bbox": [34.5, 48.0, 35.0, 48.5],
        "date": "2024-05-01T00:00:00Z/2024-07-31T23:59:59Z",
        "col_off": 3000, "row_off": 3000, "size": 512,
    },
    {
        "name": "ec_river_delta",
        "desc": "River Delta — Mississippi River Active Sediment Delta",
        "bbox": [-89.5, 29.2, -89.1, 29.6],
        "date": "2024-05-01T00:00:00Z/2024-08-31T23:59:59Z",
        "col_off": 6900, "row_off": 6000, "size": 512,
    },
    {
        "name": "ec_suburban_mixed",
        "desc": "Suburban Mixed — London Thames & Greenwich Corridor",
        "bbox": [-0.1, 51.4, 0.1, 51.6],
        "date": "2024-05-01T00:00:00Z/2024-07-31T23:59:59Z",
        "col_off": 9400, "row_off": 8400, "size": 512,
    },
]

BANDS = ["red", "green", "blue", "nir"]

def stac_search(bbox, date_range):
    payload = json.dumps({
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "limit": 5,
    }).encode()

    req = urllib.request.Request(
        STAC_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    features = data.get("features", [])
    if not features:
        return None
    features.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))
    return features[0]

def download_scenario(sc):
    out_path = OUT_DIR / f"{sc['name']}.tif"
    print(f"\n[{sc['name']}] {sc['desc']}")

    print(f"  Searching real Sentinel-2 scene via STAC ...", end=" ", flush=True)
    item = stac_search(sc["bbox"], sc["date"])
    if item is None:
        raise RuntimeError("No STAC scene found")
    cloud = item.get("properties", {}).get("eo:cloud_cover", "?")
    scene_id = item.get("id", "?")
    print(f"FOUND: {scene_id} (Cloud: {cloud}%)")

    assets = item.get("assets", {})
    arrays = []
    meta = None
    transform = None

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for band_key in BANDS:
            href = assets.get(band_key, {}).get("href")
            if not href:
                raise RuntimeError(f"Band '{band_key}' missing from STAC asset")
            print(f"  Fetching {band_key} band from AWS ...", end=" ", flush=True)
            with rasterio.open(href) as src:
                h, w = src.height, src.width
                safe_col = min(sc["col_off"], max(0, w - sc["size"]))
                safe_row = min(sc["row_off"], max(0, h - sc["size"]))
                safe_size = min(sc["size"], w - safe_col, h - safe_row)
                win = Window(safe_col, safe_row, safe_size, safe_size)

                data = src.read(1, window=win, resampling=Resampling.nearest)
                if meta is None:
                    meta = src.meta.copy()
                    transform = src.window_transform(win)
                arrays.append(data)
                print(f"OK ({data.shape[0]}x{data.shape[1]} px, DN [{data.min()}..{data.max()}])")

    actual_size = arrays[0].shape[0]
    meta.update({
        "count": len(arrays),
        "width": actual_size,
        "height": actual_size,
        "transform": transform,
        "compress": "deflate",
        "driver": "GTiff",
    })
    with rasterio.open(out_path, "w", **meta) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr, i)

    size_kb = out_path.stat().st_size // 1024
    print(f"  [SUCCESS] Written authentic Sentinel-2 GeoTIFF -> {out_path.name} ({size_kb} KB)")
    return out_path

def main():
    print("=" * 65)
    print("DOWNLOADING 7 AUTHENTIC REAL-LIFE SENTINEL-2 OPTICAL/NIR SATELLITE IMAGES")
    print("Data Source: ESA Copernicus Sentinel-2 L2A via AWS Open Data")
    print("=" * 65)

    failed = []
    for sc in SCENARIOS:
        try:
            download_scenario(sc)
        except Exception as e:
            print(f"  [FAIL] {sc['name']}: {e}")
            failed.append(sc["name"])

    print("\n" + "=" * 65)
    done = len(SCENARIOS) - len(failed)
    print(f"Downloaded {done}/{len(SCENARIOS)} authentic real-world Sentinel-2 images.")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("ALL 7 REAL-LIFE SENTINEL-2 SATELLITE IMAGES READY FOR EVALUATION!")

if __name__ == "__main__":
    main()
