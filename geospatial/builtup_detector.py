from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from geospatial.water_detector import resolve_spectral_bands


def detect_spectral_builtup(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Multispectral built-up detection using NDBI / SWIR / NIR when available,
    or high-resolution edge-density texture on optical bands.
    """
    import rasterio

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions or [])
        count = src.count

        bands = resolve_spectral_bands(descriptions, count)

        # If SWIR and NIR are both available, compute NDBI = (SWIR - NIR) / (SWIR + NIR)
        if "swir" in bands and "nir" in bands:
            swir = src.read(bands["swir"]).astype(np.float32)
            nir = src.read(bands["nir"]).astype(np.float32)
            ndbi = (swir - nir) / (swir + nir + 1e-6)
            builtup_mask = (ndbi > 0.0).astype(np.uint8) * 255
            builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            active_pixels = int((builtup_mask > 0).sum())
            builtup_pct = 100.0 * active_pixels / float(max(builtup_mask.size, 1))
            diagnostics = {
                "mode": "multispectral_ndbi",
                "bands": bands,
                "builtup_pixels": active_pixels,
                "builtup_percent": round(builtup_pct, 2),
            }
            return builtup_mask, "Multispectral NDBI (SWIR/NIR)", 0.88, diagnostics

    return detect_rgb_builtup(data)


def detect_rgb_builtup(
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    RGB built-up and settlement detection based on edge density,
    low-to-moderate saturation, and urban structural texture.
    """
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB built-up detection requires a 3-channel image.")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 130)
    edge_density = cv2.blur((edges > 0).astype(np.float32), (9, 9))

    # Built-up structures have high edge density, neutral color (low saturation), and moderate-high reflectance
    urban_cand = (sat < 75.0) & (val > 65.0) & (edge_density > 0.04)
    builtup_mask = (urban_cand.astype(np.uint8)) * 255
    builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    active_pixels = int((builtup_mask > 0).sum())
    builtup_pct = 100.0 * active_pixels / float(max(builtup_mask.size, 1))

    diagnostics = {
        "mode": "rgb_texture",
        "builtup_pixels": active_pixels,
        "builtup_percent": round(builtup_pct, 2),
    }

    method = "Urban Structural Texture & Edge Density Baseline"
    return builtup_mask, method, 0.78, diagnostics


def detect_remote_sensing_builtup(
    path: Optional[Path],
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Unified remote-sensing built-up detection dispatcher.
    """
    if path and path.is_file() and path.suffix.lower() in [".tif", ".tiff"]:
        try:
            return detect_spectral_builtup(path, data)
        except Exception:
            pass

    return detect_rgb_builtup(data)
