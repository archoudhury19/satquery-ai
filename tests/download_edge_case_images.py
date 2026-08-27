"""
tests/download_edge_case_images.py
===================================
Download diverse real-world satellite imagery covering all edge cases:
1. Dense urban core (high built-up, low veg)
2. Dense forest / jungle (high veg, low built-up, no water)
3. Open water / ocean coastline (dominant water)
4. Arid desert / barren (high bare, low everything)
5. Agricultural patchwork (mixed veg + bare + rural built-up)
6. River delta / wetland (mixed water + veg)
7. Mixed suburban (moderate built-up + veg + water)
8. SAR-only backscatter tile

All sources are open-access COG / GeoTIFF URLs.
"""

import json
import requests
import rasterio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "demo_data" / "edge_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Real public COG satellite imagery covering diverse edge cases
EDGE_CASE_CATALOG = [
    {
        "id": "sentinel2_urban_dense",
        "title": "Sentinel-2 TCI — Dense Urban (Paris, France)",
        "scene": "Dense city centre, high built-up, minimal vegetation",
        "expected": {"water": "<5%", "vegetation": "<15%", "built_up": ">60%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/31/U/DQ/2020/6/S2B_31UDQ_20200601_0_L2A/TCI.tif",
        "filename": "ec_urban_paris_s2_tci.tif",
    },
    {
        "id": "sentinel2_agricultural",
        "title": "Sentinel-2 TCI — Agricultural Patchwork (Ukraine Farmland)",
        "scene": "Open farmland, crop parcels, rural settlements, mixed bare/veg",
        "expected": {"water": "<8%", "vegetation": "30-60%", "built_up": "<15%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/U/YA/2020/7/S2B_36UYA_20200701_0_L2A/TCI.tif",
        "filename": "ec_agricultural_ukraine_s2_tci.tif",
    },
    {
        "id": "sentinel2_coastal_water",
        "title": "Sentinel-2 TCI — Coastal / Ocean (Mediterranean)",
        "scene": "Dominant water body, coastal strip, port infrastructure",
        "expected": {"water": ">50%", "vegetation": "<20%", "built_up": "<20%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/33/T/WF/2020/8/S2B_33TWF_20200801_0_L2A/TCI.tif",
        "filename": "ec_coastal_water_mediterranean_s2_tci.tif",
    },
    {
        "id": "sentinel2_forest",
        "title": "Sentinel-2 TCI — Dense Forest (Amazon Basin, Brazil)",
        "scene": "Tropical rainforest, continuous canopy, minimal urban",
        "expected": {"water": "<10%", "vegetation": ">70%", "built_up": "<5%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/21/M/YT/2020/7/S2B_21MYT_20200701_0_L2A/TCI.tif",
        "filename": "ec_forest_amazon_s2_tci.tif",
    },
    {
        "id": "sentinel2_desert_arid",
        "title": "Sentinel-2 TCI — Arid Desert (Sahara, Algeria)",
        "scene": "Bare rocky desert, no vegetation, no water, sand dunes",
        "expected": {"water": "<2%", "vegetation": "<5%", "built_up": "<5%", "bare": ">85%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/R/KS/2020/7/S2B_32RKS_20200701_0_L2A/TCI.tif",
        "filename": "ec_desert_sahara_s2_tci.tif",
    },
    {
        "id": "sentinel2_river_delta",
        "title": "Sentinel-2 TCI — River Delta / Wetland (Nile Delta, Egypt)",
        "scene": "River channels, delta floodplain, irrigated agriculture and water",
        "expected": {"water": "15-40%", "vegetation": "20-50%", "built_up": "<20%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/R/VU/2020/6/S2B_36RVU_20200601_0_L2A/TCI.tif",
        "filename": "ec_river_delta_nile_s2_tci.tif",
    },
    {
        "id": "sentinel2_suburban_mixed",
        "title": "Sentinel-2 TCI — Mixed Suburban (London Outskirts, UK)",
        "scene": "Suburban housing, green parks, rivers, mixed land use",
        "expected": {"water": "5-15%", "vegetation": "20-45%", "built_up": "25-50%"},
        "url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/30/U/YC/2020/6/S2B_30UYC_20200601_0_L2A/TCI.tif",
        "filename": "ec_suburban_london_s2_tci.tif",
    },
]


def download_with_range_request(url: str, dest: Path, max_bytes: int = 3_000_000) -> bool:
    """
    Download first N bytes of a COG file using Range request.
    COG files are structured so the header + overview tiles come first.
    """
    try:
        headers = {"Range": f"bytes=0-{max_bytes}"}
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        if r.status_code in (200, 206):
            dest.write_bytes(r.content)
            return True
        return False
    except Exception:
        return False


def run_downloads():
    print("=" * 60)
    print("DOWNLOADING REAL-WORLD EDGE CASE SATELLITE IMAGERY")
    print("=" * 60)
    print()

    manifest = []
    for item in EDGE_CASE_CATALOG:
        dest = OUT_DIR / item["filename"]
        print(f"[*] {item['title']}")
        print(f"    Scene    : {item['scene']}")
        print(f"    Expected : {item['expected']}")

        # Try full download first, fallback to Range request
        success = False
        try:
            r = requests.get(item["url"], timeout=25, stream=True)
            if r.status_code == 200:
                total = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        total += len(chunk)
                        if total >= 2_500_000:  # cap at 2.5MB for speed
                            break
                success = True
        except Exception:
            success = download_with_range_request(item["url"], dest)

        if success and dest.exists() and dest.stat().st_size > 1000:
            try:
                with rasterio.open(dest) as src:
                    specs = f"{src.width}x{src.height}px, {src.count} bands, CRS:{src.crs}"
                print(f"    [+] Saved : {dest.name} ({dest.stat().st_size//1024}KB) — {specs}")
                rec = dict(item)
                rec["path"] = str(dest)
                rec["file_size_kb"] = dest.stat().st_size // 1024
                manifest.append(rec)
            except Exception as e:
                print(f"    [-] Rasterio read error: {e}")
        else:
            print(f"    [-] Download failed or file too small")
        print()

    manifest_path = OUT_DIR / "edge_case_manifest.json"
    manifest_path.write_text(json.dumps({"edge_cases": manifest}, indent=2))
    print(f"[+] Manifest saved to: {manifest_path}")
    print(f"[+] Downloaded {len(manifest)}/{len(EDGE_CASE_CATALOG)} edge case images\n")
    return manifest


if __name__ == "__main__":
    run_downloads()
