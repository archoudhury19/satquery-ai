import rasterio
from pathlib import Path

images = [
    ("Optical Urban (Kolkata)", "demo_data/01_kolkata_optical_georef.tif"),
    ("SAR Sentinel-1 (Kolkata)", "demo_data/02_kolkata_sar_sentinel1.tif"),
    ("Multispectral Sentinel-2 T1", "demo_data/03_sentinel2_multispectral_t1.tif"),
    ("Multispectral Sentinel-2 T2", "demo_data/04_sentinel2_multispectral_t2.tif"),
    ("ISRO Cartosat Optical", "demo_data/isro_sac/cartosat_optical_coregistered.tif"),
    ("ISRO RISAT SAR", "demo_data/isro_sac/risat_sar_coregistered.tif"),
    ("BigEarthNet Sentinel-2", "demo_data/bigearthnet/S2_multispectral_patch.tif"),
    ("BigEarthNet Sentinel-1 SAR", "demo_data/bigearthnet/S1_sar_patch.tif"),
]

print("============================================================")
print("TEST SATELLITE IMAGERY CATALOG")
print("============================================================\n")

for label, p_str in images:
    p = Path(p_str)
    if not p.exists():
        print(f"[-] {label}: File not found ({p_str})")
        continue
    with rasterio.open(p) as src:
        crs = src.crs.to_string() if src.crs else "Non-georeferenced"
        bounds = src.bounds
        descriptions = src.descriptions
        print(f"[+] {label}")
        print(f"    Path        : {p_str}")
        print(f"    Dimensions  : {src.width} x {src.height} px, {src.count} band(s)")
        print(f"    CRS         : {crs}")
        print(f"    Band Names  : {descriptions}")
        print(f"    File Size   : {p.stat().st_size / 1024:.1f} KB")
        print()
