from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from geospatial.water_detector import resolve_spectral_bands


def detect_spectral_vegetation(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Multispectral vegetation detection using Normalized Difference
    Vegetation Index (NDVI) with Sentinel-2 Near-Infrared (B08) and Red (B04).

    NDVI = (NIR - Red) / (NIR + Red + 1e-6)
    """
    import rasterio

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions or [])
        count = src.count

        bands = resolve_spectral_bands(descriptions, count)
        if "nir" not in bands or "red" not in bands:
            return detect_rgb_vegetation(data)

        nir = src.read(bands["nir"]).astype(np.float32)
        red = src.read(bands["red"]).astype(np.float32)

    denom = nir + red + 1e-6
    ndvi = (nir - red) / denom
    ndvi = np.nan_to_num(ndvi, nan=-1.0, posinf=1.0, neginf=-1.0)

    # Standard remote sensing vegetation threshold (NDVI >= 0.20 for healthy vegetation)
    veg_mask = (ndvi >= 0.20).astype(np.uint8) * 255
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    active_pixels = int((veg_mask > 0).sum())
    veg_pct = 100.0 * active_pixels / float(max(veg_mask.size, 1))

    diagnostics = {
        "mode": "multispectral",
        "bands": bands,
        "ndvi_mean": round(float(np.mean(ndvi)), 4),
        "ndvi_max": round(float(np.max(ndvi)), 4),
        "threshold_used": 0.20,
        "vegetation_pixels": active_pixels,
        "vegetation_percent": round(veg_pct, 2),
    }

    method = "Multispectral NDVI (NIR/Red, threshold >= 0.20)"
    return veg_mask, method, 0.90, diagnostics


def detect_rgb_vegetation(
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    RGB vegetation fallback using Excess Green Index (ExG = 2G - R - B)
    computed on float32 (avoiding uint8 overflow) and HSV green-hue gate.

    Vegetation pixels are expected to:
    - Have greenish hue (HSV H roughly 25–85 degrees = opencv H 13–42)
    - ExG > 20 on [0-255] float32 range
    - Saturation > 35 (not achromatic / grey surfaces)
    - Green channel actually dominant over Red AND Blue
    """
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB vegetation detection requires a 3-channel image.")

    # Normalise to [0, 255] range regardless of input scale
    maxv = float(np.max(rgb)) if rgb.size else 1.0
    if maxv <= 1.5:
        rgb = rgb * 255.0
    elif maxv > 255.0:
        rgb = (rgb / maxv) * 255.0
    rgb = np.clip(rgb, 0.0, 255.0)

    red_f   = rgb[:, :, 0]
    green_f = rgb[:, :, 1]
    blue_f  = rgb[:, :, 2]

    # ExG on float32 — no overflow
    exg = 2.0 * green_f - red_f - blue_f

    # HSV for hue and saturation gating
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)   # OpenCV: [0, 179]
    sat = hsv[:, :, 1].astype(np.float32)   # [0, 255]

    # Green hue band: 25-85 degrees → OpenCV H 13-42
    green_hue = (hue >= 13) & (hue <= 42)

    # All conditions must hold simultaneously
    veg_mask = (
        green_hue
        & (exg > 20.0)
        & (sat > 35.0)
        & (green_f > red_f + 8.0)
        & (green_f > blue_f + 5.0)
    )

    veg_uint8 = veg_mask.astype(np.uint8) * 255
    veg_uint8 = cv2.morphologyEx(veg_uint8, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    veg_uint8 = cv2.morphologyEx(veg_uint8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    active_pixels = int((veg_uint8 > 0).sum())
    total_pixels  = int(veg_uint8.size)
    veg_pct = 100.0 * active_pixels / max(total_pixels, 1)

    diagnostics = {
        "mode": "rgb_fallback_fixed",
        "vegetation_pixels": active_pixels,
        "vegetation_percent": round(veg_pct, 2),
        "exg_mean": round(float(np.mean(exg)), 2),
    }

    method = "RGB Excess Green (float32) + HSV Green-Hue Gate"
    return veg_uint8, method, 0.72, diagnostics


def detect_remote_sensing_vegetation(
    path: Optional[Path],
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Unified remote-sensing vegetation detection dispatcher.
    """
    if path and path.is_file() and path.suffix.lower() in [".tif", ".tiff"]:
        try:
            return detect_spectral_vegetation(path, data)
        except Exception:
            pass

    return detect_rgb_vegetation(data)
