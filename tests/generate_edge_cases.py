"""
tests/generate_edge_cases.py
==============================
Generates realistic synthetic satellite GeoTIFFs for all edge-case scenarios.
Each image is created from known spectral signatures for the land-cover class.

WHY SYNTHETIC IS VALID:
- The segmenter uses CLIP + spectral physics (RGB statistics, HSV, ExG)
- A dense urban tile made from REAL concrete/asphalt RGB values IS a real test
- Ground truth is known precisely, so failures are unambiguous
- 7 scenarios × deterministic seeds = reproducible regression tests
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "demo_data" / "edge_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_tif(arr: np.ndarray, path: Path, crs_epsg: int = 4326, bbox=(-10, 40, 10, 60)):
    """Save H×W×C uint8 array as 3-band GeoTIFF."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    h, w = arr.shape[:2]
    transform = from_bounds(*bbox, width=w, height=h)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=3,
        dtype=rasterio.uint8,
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)


def _noise(shape, scale=8, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# ─── 1. DENSE URBAN CORE ───────────────────────────────────────────────────
def make_urban_dense(path: Path, size: int = 256):
    """
    High built-up (>70%), low veg (<10%), minimal water.
    Spectral signature: grey concrete (R≈G≈B≈130-180), high edge density from buildings.
    """
    rng = np.random.default_rng(1)
    img = np.zeros((size, size, 3), np.float32)

    # Base: grey urban fabric
    base_grey = rng.integers(110, 175, (size, size), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = base_grey + _noise((size, size), 10, rng)

    # Add road grid (dark grey, ~90)
    for i in range(0, size, 28):
        r_end = min(i + 5, size)
        c_end = min(i + 5, size)
        h_r = r_end - i
        img[i:r_end, :, :] = 85 + _noise((h_r, size, 3), 5, rng)
        img[:, i:c_end, :] = 85 + _noise((size, h_r, 3), 5, rng)

    # Add building blocks (varied grey, 130-200)
    for y in range(0, size, 28):
        for x in range(0, size, 28):
            bh, bw = int(rng.integers(12, 22)), int(rng.integers(12, 22))
            shade = int(rng.integers(120, 200))
            y1, y2 = y + 5, min(y + 5 + bh, size)
            x1, x2 = x + 5, min(x + 5 + bw, size)
            if y2 > y1 and x2 > x1:
                img[y1:y2, x1:x2, :] = shade + _noise((y2-y1, x2-x1, 3), 12, rng)

    # Small park patch (green, ~10% area)
    img[10:45, 10:55, 0] = 60 + _noise((35, 45), 6, rng)   # R low
    img[10:45, 10:55, 1] = 110 + _noise((35, 45), 8, rng)  # G high
    img[10:45, 10:55, 2] = 50 + _noise((35, 45), 6, rng)   # B low

    # Tiny water body (~3%)
    img[200:215, 200:250, 0] = 40 + _noise((15, 50), 4, rng)  # R very low
    img[200:215, 200:250, 1] = 65 + _noise((15, 50), 5, rng)  # G low
    img[200:215, 200:250, 2] = 100 + _noise((15, 50), 8, rng)  # B dominant

    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (-0.2, 51.4, 0.2, 51.6))
    return {"expected_built_up_pct_min": 60, "expected_veg_pct_max": 15}


# ─── 2. DENSE TROPICAL FOREST ──────────────────────────────────────────────
def make_forest_dense(path: Path, size: int = 256):
    """
    High vegetation (>75%), no water, no built-up.
    Spectral: dark/medium green dominant (G >> R > B), high ExG.
    """
    rng = np.random.default_rng(2)
    img = np.zeros((size, size, 3), np.float32)
    # Dark forest canopy: G channel dominant, low R and B
    img[:, :, 0] = 45 + _noise((size, size), 10, rng)    # R
    img[:, :, 1] = 90 + _noise((size, size), 15, rng)    # G (dominant)
    img[:, :, 2] = 30 + _noise((size, size), 8, rng)     # B
    # Lighter canopy variation (sunlit gaps)
    for _ in range(20):
        y, x = rng.integers(0, size-20, 2)
        h, w = rng.integers(10, 30, 2)
        img[y:y+h, x:x+w, 0] += rng.integers(10, 25)
        img[y:y+h, x:x+w, 1] += rng.integers(15, 35)
        img[y:y+h, x:x+w, 2] += rng.integers(5, 15)
    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (-62, -5, -58, -2))
    return {"expected_veg_pct_min": 70, "expected_built_up_pct_max": 5}


# ─── 3. OPEN WATER / OCEAN ─────────────────────────────────────────────────
def make_water_dominant(path: Path, size: int = 256):
    """
    Dominant water (>80%), small coastal strip at edge.
    Spectral: blue dominant (B >> R, G), low total reflectance.
    """
    rng = np.random.default_rng(3)
    img = np.zeros((size, size, 3), np.float32)
    # Deep water
    img[:, :, 0] = 22 + _noise((size, size), 5, rng)   # R very low
    img[:, :, 1] = 55 + _noise((size, size), 8, rng)   # G medium-low
    img[:, :, 2] = 95 + _noise((size, size), 12, rng)  # B dominant
    # Shallow/turbid water strip
    img[:, :size//6, 0] = 55 + _noise((size, size//6), 8, rng)
    img[:, :size//6, 1] = 75 + _noise((size, size//6), 10, rng)
    img[:, :size//6, 2] = 80 + _noise((size, size//6), 10, rng)
    # Thin coastal land (rightmost 10%)
    land_w = size // 10
    img[:, -land_w:, 0] = 120 + _noise((size, land_w), 12, rng)
    img[:, -land_w:, 1] = 110 + _noise((size, land_w), 10, rng)
    img[:, -land_w:, 2] = 85 + _noise((size, land_w), 10, rng)
    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (5, 43, 7, 44))
    return {"expected_water_pct_min": 70, "expected_veg_pct_max": 10}


# ─── 4. ARID DESERT / BARE ─────────────────────────────────────────────────
def make_desert_bare(path: Path, size: int = 256):
    """
    Dominant bare ground (>85%), no vegetation, no water.
    Spectral: sandy/brown (R>G>B), all channels moderate-high.
    """
    rng = np.random.default_rng(4)
    img = np.zeros((size, size, 3), np.float32)
    img[:, :, 0] = 175 + _noise((size, size), 18, rng)  # R high (sand/rock)
    img[:, :, 1] = 150 + _noise((size, size), 15, rng)  # G medium
    img[:, :, 2] = 100 + _noise((size, size), 12, rng)  # B lower
    # Rock outcrops (darker, more contrast)
    for _ in range(8):
        y, x = rng.integers(0, size-30, 2)
        h, w = rng.integers(15, 40, 2)
        img[y:y+h, x:x+w, 0] = 140 + _noise((h, w), 15, rng)
        img[y:y+h, x:x+w, 1] = 120 + _noise((h, w), 12, rng)
        img[y:y+h, x:x+w, 2] = 90 + _noise((h, w), 10, rng)
    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (5, 23, 7, 25))
    return {"expected_bare_pct_min": 70, "expected_veg_pct_max": 5, "expected_water_pct_max": 2}


# ─── 5. AGRICULTURAL PATCHWORK ─────────────────────────────────────────────
def make_agricultural(path: Path, size: int = 256):
    """
    Mixed farmland: alternating crop fields (green), fallow (bare), small settlement.
    Edge case: field parcel edges should NOT be misclassified as built-up.
    """
    rng = np.random.default_rng(5)
    img = np.zeros((size, size, 3), np.float32)
    # Base: fallow/bare soil
    img[:, :, 0] = 150 + _noise((size, size), 12, rng)
    img[:, :, 1] = 130 + _noise((size, size), 10, rng)
    img[:, :, 2] = 90 + _noise((size, size), 10, rng)
    # Green crop parcels in checkerboard
    parcel = 30
    for py in range(0, size, parcel):
        for px in range(0, size, parcel):
            if (py // parcel + px // parcel) % 2 == 0:
                py2, px2 = min(py + parcel, size), min(px + parcel, size)
                h_p, w_p = py2 - py, px2 - px
                img[py:py2, px:px2, 0] = 50 + _noise((h_p, w_p), 8, rng)
                img[py:py2, px:px2, 1] = 110 + _noise((h_p, w_p), 12, rng)
                img[py:py2, px:px2, 2] = 40 + _noise((h_p, w_p), 6, rng)
    # Small rural settlement (5% area, top-right)
    img[5:40, -55:-5, :] = 140 + _noise((35, 50, 3), 15, rng)
    img[5:40, -55:-5, 1] -= 20  # slightly less green than veg
    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (30, 49, 32, 51))
    return {"expected_built_up_pct_max": 12, "expected_veg_pct_min": 25}


# ─── 6. RIVER DELTA / WETLAND ──────────────────────────────────────────────
def make_river_delta(path: Path, size: int = 256):
    """
    Mixed water channels + marsh vegetation + delta fans.
    Edge case: turbid sediment water (brownish) should be classified as water, not bare.
    """
    rng = np.random.default_rng(6)
    img = np.zeros((size, size, 3), np.float32)
    # Background: marsh vegetation
    img[:, :, 0] = 55 + _noise((size, size), 10, rng)
    img[:, :, 1] = 100 + _noise((size, size), 15, rng)
    img[:, :, 2] = 40 + _noise((size, size), 8, rng)
    # River channels (clear water — blue dominant)
    for channel_y in [60, 130, 195]:
        w = rng.integers(12, 22)
        img[channel_y:channel_y+w, :, 0] = 35 + _noise((w, size), 6, rng)
        img[channel_y:channel_y+w, :, 1] = 65 + _noise((w, size), 8, rng)
        img[channel_y:channel_y+w, :, 2] = 105 + _noise((w, size), 12, rng)
    # Turbid/sediment water (brownish — edge case)
    img[90:130, 50:180, 0] = 105 + _noise((40, 130), 10, rng)  # R elevated (sediment)
    img[90:130, 50:180, 1] = 95 + _noise((40, 130), 8, rng)
    img[90:130, 50:180, 2] = 80 + _noise((40, 130), 8, rng)
    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (31, 30, 32, 31))
    return {"expected_water_pct_min": 20, "expected_veg_pct_min": 20}


# ─── 7. SUBURBAN MIXED ─────────────────────────────────────────────────────
def make_suburban_mixed(path: Path, size: int = 256):
    """
    Mixed suburban: housing blocks + tree-lined streets + river/lake.
    Edge case: vegetation inside urban area should not all collapse to built-up.
    """
    rng = np.random.default_rng(7)
    img = np.zeros((size, size, 3), np.float32)
    # Base: lighter urban/suburban grey
    img[:, :, 0] = 145 + _noise((size, size), 12, rng)
    img[:, :, 1] = 140 + _noise((size, size), 10, rng)
    img[:, :, 2] = 120 + _noise((size, size), 10, rng)
    # Green residential gardens & parks (~35%)
    for _ in range(18):
        y, x = rng.integers(0, size-25, 2)
        h, w = rng.integers(10, 30, 2)
        img[y:y+h, x:x+w, 0] = 55 + _noise((h, w), 8, rng)
        img[y:y+h, x:x+w, 1] = 110 + _noise((h, w), 12, rng)
        img[y:y+h, x:x+w, 2] = 40 + _noise((h, w), 6, rng)
    # River (~10%)
    img[110:135, :, 0] = 30 + _noise((25, size), 6, rng)
    img[110:135, :, 1] = 70 + _noise((25, size), 8, rng)
    img[110:135, :, 2] = 120 + _noise((25, size), 12, rng)
    _save_tif(np.clip(img, 0, 255).astype(np.uint8), path, 4326, (-0.4, 51.3, 0.1, 51.6))
    return {"expected_veg_pct_min": 20, "expected_water_pct_min": 5, "expected_built_up_pct_min": 20}


SCENARIOS = [
    ("ec_urban_dense.tif",       make_urban_dense,    "Dense Urban Core",            "High built-up, low veg, minimal water"),
    ("ec_forest_dense.tif",      make_forest_dense,   "Dense Tropical Forest",       "High veg, no water, no built-up"),
    ("ec_water_dominant.tif",    make_water_dominant, "Open Water / Coastal",        "Dominant water, thin coastal strip"),
    ("ec_desert_bare.tif",       make_desert_bare,    "Arid Desert / Bare Ground",   "High bare, no veg, no water"),
    ("ec_agricultural.tif",      make_agricultural,   "Agricultural Patchwork",      "Mixed crop+fallow+rural settlement — field edge false-built-up test"),
    ("ec_river_delta.tif",       make_river_delta,    "River Delta / Wetland",       "Turbid sediment water classification edge case"),
    ("ec_suburban_mixed.tif",    make_suburban_mixed, "Mixed Suburban",              "Green gardens + river inside urban area"),
]


def main():
    print("=" * 60)
    print("GENERATING EDGE CASE SATELLITE TILES")
    print("=" * 60)
    results = []
    for fname, fn, title, description in SCENARIOS:
        path = OUT_DIR / fname
        meta = fn(path)
        size_kb = path.stat().st_size // 1024
        print(f"[+] {title}")
        print(f"    Desc     : {description}")
        print(f"    File     : {path.name} ({size_kb}KB)")
        print(f"    Expected : {meta}")
        print()
        results.append({"id": fname.replace(".tif", ""), "title": title, "description": description,
                        "path": str(path), "expected": meta})

    import json
    manifest = OUT_DIR / "edge_case_manifest.json"
    manifest.write_text(json.dumps({"edge_cases": results}, indent=2))
    print(f"[+] Manifest: {manifest}")
    print(f"[+] {len(results)} edge case tiles ready\n")
    return results


if __name__ == "__main__":
    main()
