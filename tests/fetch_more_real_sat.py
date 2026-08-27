import requests
from pathlib import Path
import rasterio

REAL_IMG_DIR = Path("demo_data/real_world_satellite")

urls = [
    ("https://github.com/OSGeo/gdal/raw/master/autotest/gcore/data/byte.tif", "real_usgs_elevation_byte.tif", "USGS Real Sensor Raster"),
    ("https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif", "real_landsat_multispectral.tif", "Real Landsat Multispectral Optical Tile"),
]

for url, fname, title in urls:
    dest = REAL_IMG_DIR / fname
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.write_bytes(r.content)
        with rasterio.open(dest) as src:
            print(f"[+] Downloaded: {title} -> {fname} ({src.width}x{src.height} px, {src.count} bands)")
    except Exception as e:
        print(f"[-] Failed {fname}: {e}")
