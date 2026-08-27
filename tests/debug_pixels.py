"""Pixel-level diagnostic on vrsbench_sample_01.tif"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import rasterio

img_path = Path("demo_data/vrsbench/vrsbench_sample_01.tif")

with rasterio.open(img_path) as src:
    count = src.count
    desc = src.descriptions
    crs  = src.crs
    shape = (src.height, src.width)
    print(f"Bands: {count}, CRS: {crs}, Shape: {shape}")
    print(f"Descriptions: {desc}")
    band_data = {i+1: src.read(i+1).astype(np.float32) for i in range(min(count, 4))}

for band_idx, arr in band_data.items():
    print(f"  Band {band_idx}: min={arr.min():.1f}, max={arr.max():.1f}, "
          f"mean={arr.mean():.1f}, dtype={arr.dtype}")

# Assemble RGB
if count >= 3:
    r = band_data[1]; g = band_data[2]; b = band_data[3]
    rgb_f32 = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb_f32, 0, 255).astype(np.uint8)
else:
    gray = band_data[1]
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

# Sample pixel values at different areas
h, w = rgb.shape[:2]
samples = {
    "top-left":     rgb[h//8,  w//8],
    "top-center":   rgb[h//8,  w//2],
    "center":       rgb[h//2,  w//2],
    "bottom-left":  rgb[7*h//8, w//8],
    "bottom-right": rgb[7*h//8, 7*w//8],
}
print("\nSample RGB pixel values (R,G,B):")
for loc, px in samples.items():
    r_, g_, b_ = px[0], px[1], px[2]
    exg = 2*g_ - r_ - b_
    green_dom = g_ > r_ + 10 and g_ > b_ + 6
    print(f"  {loc:15s}: R={r_:3d} G={g_:3d} B={b_:3d}  ExG={exg:+5.0f}  green_dom={green_dom}")

# Check actual water pixel distribution
print("\n--- Water pixel stats ---")
total = h * w
# Turbid water: red channel often highest (sediment)
water_cand_rgb = (r > b + 5) & (r < 80) & (g < 70)
print(f"Turbid-water RGB cand: {water_cand_rgb.sum()} px ({100*water_cand_rgb.sum()/total:.1f}%)")

# Vegetation stats
exg = 2*g - r - b
green_dom_all = (g > r + 10) & (g > b + 6)
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
sat = hsv[:,:,1].astype(np.float32)
veg_cand = (green_dom_all | (exg > 15)) & (sat > 30)
print(f"ExG vegetation cand:   {veg_cand.sum()} px ({100*veg_cand.sum()/total:.1f}%)")

# Built-up stats
gray_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray_img, 50, 130)
edge_density = cv2.blur((edges > 0).astype(np.float32), (9, 9))
val = hsv[:,:,2].astype(np.float32)
built_cand = (sat < 75) & (val > 65) & (edge_density > 0.04)
print(f"Built-up texture cand: {built_cand.sum()} px ({100*built_cand.sum()/total:.1f}%)")

print("\n--- Saturation distribution ---")
for t in [20, 40, 60, 80, 100, 120]:
    pct = 100*(sat < t).sum()/total
    print(f"  sat < {t:3d}: {pct:.1f}%")

print("\n--- Brightness (Val) distribution ---")
for t in [50, 80, 100, 130, 160, 200]:
    pct = 100*(val > t).sum()/total
    print(f"  val > {t:3d}: {pct:.1f}%")
