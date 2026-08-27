from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio


# ============================================================
# SPECTRAL BAND RESOLUTION
# ============================================================

def resolve_spectral_bands(
    descriptions: List[Optional[str]],
    count: int,
) -> Dict[str, int]:
    """
    Auto-resolve band indices (1-indexed) for remote-sensing rasters.

    Supports:
    - Sentinel-2 naming conventions ('B03', 'B3', 'Green', 'B08', 'B8', 'NIR', 'B04', 'B4', 'Red', 'B02', 'B2', 'Blue')
    - Fallback to standard 4-band Sentinel-2 RGB+NIR composite (1: Red, 2: Green, 3: Blue, 4: NIR)
    """
    desc_clean = [
        str(d or "").lower().strip().replace("_", " ").replace("-", " ")
        for d in descriptions
    ]

    def _find_idx(keys: List[str]) -> Optional[int]:
        for i, text in enumerate(desc_clean, start=1):
            if any(k in text for k in keys):
                return i
        return None

    green = _find_idx(["b03", "b3", "green", "grn"])
    nir = _find_idx(["b08", "b8", "nir", "near infrared", "near_infrared", "b8a", "b08a"])
    red = _find_idx(["b04", "b4", "red"])
    blue = _find_idx(["b02", "b2", "blue"])
    swir = _find_idx(["b11", "b12", "swir", "swir1", "swir2"])

    # Standard fallback if 4-band raster without explicit labels
    if count >= 4 and (green is None or nir is None):
        red = red or 1
        green = green or 2
        blue = blue or 3
        nir = nir or 4

    bands: Dict[str, int] = {}
    if green is not None:
        bands["green"] = green
    if nir is not None:
        bands["nir"] = nir
    if red is not None:
        bands["red"] = red
    if blue is not None:
        bands["blue"] = blue
    if swir is not None:
        bands["swir"] = swir

    return bands


# ============================================================
# SPECTRAL INDICES & THRESHOLDING
# ============================================================

def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
) -> np.ndarray:
    """
    McFeeters (1996) Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)
    """
    denom = green + nir
    return np.divide(
        green - nir,
        denom + 1e-6,
        out=np.zeros_like(green, dtype=np.float32),
        where=np.isfinite(denom) & (denom > 0),
    )


def calculate_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    denom = nir + red
    return np.divide(
        nir - red,
        denom + 1e-6,
        out=np.zeros_like(nir, dtype=np.float32),
        where=np.isfinite(denom) & (denom > 0),
    )


def compute_otsu_ndwi_threshold(
    ndwi: np.ndarray,
    finite_mask: np.ndarray,
) -> Optional[float]:
    """
    Calculate Otsu dynamic threshold for NDWI if bimodal distribution exists.
    Maps NDWI [-1.0, 1.0] to [0, 255].
    """
    valid = ndwi[finite_mask]
    if valid.size < 100:
        return None

    # Map [-1.0, 1.0] to [0, 255]
    scaled = np.clip((valid + 1.0) * 127.5, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(
        scaled,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_thresh = (float(otsu_val) / 127.5) - 1.0
    return float(otsu_thresh)


# ============================================================
# MULTISPECTRAL WATER DETECTION
# ============================================================

def detect_spectral_water(
    path: Path,
    data: Dict[str, Any],
    bands: Dict[str, int],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Remote-sensing water detector using real multispectral bands.

    Key physical principles:
    1. NDWI (McFeeters): Water reflects green light and absorbs NIR.
    2. NDVI Gating: Vegetation has high NIR/Red reflectance (NDVI > 0.2),
       while water has negative or very low NDVI.
    3. NIR Absorption Gating: Water absorbs almost all NIR radiation.
    4. Adaptive Otsu/Zero-crossing: Replaces scene-percentile distortion
       with physically meaningful zero-crossing (NDWI > 0.0) or Otsu threshold.
    """
    with rasterio.open(path) as src:
        green = src.read(bands["green"]).astype(np.float32)
        nir = src.read(bands["nir"]).astype(np.float32)
        red = (
            src.read(bands["red"]).astype(np.float32)
            if "red" in bands
            else None
        )

    finite = np.isfinite(green) & np.isfinite(nir)
    if red is not None:
        finite &= np.isfinite(red)

    ndwi = calculate_ndwi(green, nir)
    valid_ndwi = ndwi[finite]

    if valid_ndwi.size == 0:
        return np.zeros_like(ndwi, dtype=np.uint8), "Multispectral NDWI (Empty)", 0.0, {}

    ndwi_min = float(np.min(valid_ndwi))
    ndwi_max = float(np.max(valid_ndwi))
    ndwi_mean = float(np.mean(valid_ndwi))
    ndwi_median = float(np.median(valid_ndwi))

    # Otsu thresholding evaluation
    otsu_t = compute_otsu_ndwi_threshold(ndwi, finite)

    # Physical baseline for McFeeters NDWI is 0.0.
    # If Otsu finds a clean separation in the reasonable [0.0, 0.25] interval, use it;
    # otherwise, default to standard physical zero-crossing (0.0).
    if otsu_t is not None and 0.0 <= otsu_t <= 0.25:
        threshold = float(otsu_t)
        method_desc = f"Multispectral NDWI (Otsu T={threshold:.3f}) + NDVI/NIR Gating"
    else:
        threshold = 0.0
        method_desc = "Multispectral NDWI (Zero-Crossing T=0.00) + NDVI/NIR Gating"

    candidate = finite & (ndwi > threshold)

    ndvi_stats: Dict[str, float] = {}
    if red is not None:
        ndvi = calculate_ndvi(nir, red)
        valid_ndvi = ndvi[finite]
        if valid_ndvi.size > 0:
            ndvi_stats = {
                "min": float(np.min(valid_ndvi)),
                "max": float(np.max(valid_ndvi)),
                "mean": float(np.mean(valid_ndvi)),
                "median": float(np.median(valid_ndvi)),
            }
        # Vegetation suppression: water has low/negative NDVI and NDWI > NDVI
        candidate &= (ndvi < 0.15) & (ndwi > ndvi)

    # NIR absorption suppression: reject excessively high NIR reflectance
    nir_p90 = float(np.percentile(nir[finite], 90))
    candidate &= (nir <= max(nir_p90, 1.0))

    # Morphological refinement
    mask = (candidate.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 5), np.uint8))

    # Connected component filtering to remove tiny speckles
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask)
    min_area = max(30, int(mask.size * 0.00004))

    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_id] = 255

    active_pixels = int((cleaned > 0).sum())
    water_pct = 100.0 * active_pixels / float(max(cleaned.size, 1))

    diagnostics = {
        "mode": "multispectral",
        "bands": bands,
        "ndwi_stats": {
            "min": round(ndwi_min, 4),
            "max": round(ndwi_max, 4),
            "mean": round(ndwi_mean, 4),
            "median": round(ndwi_median, 4),
        },
        "ndvi_stats": {k: round(v, 4) for k, v in ndvi_stats.items()},
        "threshold_used": round(threshold, 4),
        "otsu_threshold": round(otsu_t, 4) if otsu_t is not None else None,
        "water_pixels": active_pixels,
        "water_percent": round(water_pct, 2),
    }

    return cleaned, method_desc, 0.92, diagnostics


# ============================================================
# RGB WATER DETECTION FALLBACK
# ============================================================

def detect_rgb_water(
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Heuristic RGB water baseline when multispectral Near-Infrared
    is genuinely unavailable (e.g. 3-channel optical composites, JPEGs, PNGs).
    """
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB water detection requires a 3-channel image.")

    rgb = np.nan_to_num(rgb[:, :, :3], nan=0.0, posinf=255.0, neginf=0.0)
    max_val = float(np.max(rgb)) if rgb.size else 0.0
    min_val = float(np.min(rgb)) if rgb.size else 0.0

    if 0.0 <= min_val and max_val <= 1.5:
        rgb = rgb * 255.0
    elif max_val > 255.0:
        rgb = (rgb / max_val) * 255.0

    rgb = np.clip(rgb, 0.0, 255.0).astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    gray = (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.float32)

    p20, p50, p90 = np.percentile(gray, [20, 50, 90])

    local_mean = cv2.blur(gray, (11, 11))
    local_sq_mean = cv2.blur(gray * gray, (11, 11))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    edges = cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 45, 120)
    edge_density = cv2.blur((edges > 0).astype(np.float32), (11, 11))

    texture_limit = max(7.0, float(np.percentile(local_std, 55)))
    smooth = local_std <= texture_limit
    low_edge = edge_density <= 0.10

    # Chromatic blue/cyan
    blue_dominant = (blue > red + 5.0) & (blue >= green - 12.0)
    blue_green = (((blue + green) * 0.5) > (red + 2.0)) & (saturation < 190.0)

    # Turbid / pale river water (grayish/brownish water bodies)
    neutral_water = (
        (saturation < 100.0)
        & (value > max(25.0, float(p20 - 15.0)))
        & (value < min(250.0, float(p90 + 35.0)))
        & (green <= red + 25.0)
        & (blue <= red + 30.0)
    )

    # Silt / sediment-laden river water (brownish/turbid river channels)
    turbid_river_cand = (
        (local_std <= max(texture_limit * 1.5, 14.0))
        & (edge_density <= 0.14)
        & (value > 12.0)
        & (value < min(130.0, float(p90 + 20.0)))
        & (saturation < 170.0)
        & (red >= blue - 8.0)
        & (green <= red * 1.25 + 5.0)
    )

    # Vegetation suppression
    vegetation_like = (green > red + 10.0) & (green > blue + 6.0) & (saturation > 45.0)

    # Urban texture suppression
    urban_texture = (edge_density > 0.14) & (saturation < 95.0) & (gray > p50)

    chromatic_cand = (blue_dominant | blue_green) & (smooth | low_edge)
    pale_river_cand = neutral_water & (local_std <= max(texture_limit * 1.25, 12.0)) & (edge_density <= 0.12) & ~vegetation_like
    dark_smooth_cand = (gray < p50) & (saturation < 125.0) & (smooth | low_edge) & (blue_dominant | blue_green | neutral_water | (value > 10.0))

    candidate = (
        np.asarray(chromatic_cand, dtype=bool)
        | np.asarray(pale_river_cand, dtype=bool)
        | np.asarray(turbid_river_cand, dtype=bool)
        | np.asarray(dark_smooth_cand, dtype=bool)
    )
    candidate &= ~np.asarray(vegetation_like, dtype=bool)
    candidate &= ~np.asarray(urban_texture, dtype=bool)
    candidate &= ~((gray > p90) & (saturation < 55.0) & (edge_density > 0.08))

    mask = np.asarray(candidate, dtype=np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask)
    min_area = max(40, int(mask.size * 0.00005))

    components = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label_id))

    components.sort(reverse=True)
    for _, label_id in components[:12]:
        cleaned[labels == label_id] = 255

    active_pixels = int((cleaned > 0).sum())
    water_pct = 100.0 * active_pixels / float(max(cleaned.size, 1))

    diagnostics = {
        "mode": "rgb_fallback",
        "bands": {"red": 1, "green": 2, "blue": 3},
        "water_pixels": active_pixels,
        "water_percent": round(water_pct, 2),
    }

    return cleaned, "Adaptive RGB water baseline (spectral + pale-river + texture)", 0.78, diagnostics


# ============================================================
# UNIFIED DISPATCHER
# ============================================================

def detect_remote_sensing_water(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Unified Remote-Sensing Water Detector.

    1. Checks if the file is a GeoTIFF with resolvable multispectral bands
       (Green + NIR). If available, runs the physical multispectral detector.
    2. Otherwise, falls back to the robust RGB detector.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            with rasterio.open(path) as src:
                bands = resolve_spectral_bands(src.descriptions, src.count)
                if "green" in bands and "nir" in bands:
                    return detect_spectral_water(path, data, bands)
        except Exception as exc:
            print(f"[WATER_DETECTOR] Spectral detection failed, falling back to RGB: {exc}")

    return detect_rgb_water(data)
