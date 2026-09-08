from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


from geospatial.fusion import apply_speckle_filter


def calibrate_sar_db(
    raw_band: np.ndarray,
    calibration_factor_db: float = 0.0,
    apply_lee: bool = True,
) -> np.ndarray:
    """
    Convert raw Sentinel-1 / RISAT SAR DN values to calibrated backscatter (dB) scale:
    sigma^0_dB = 10 * log10(DN^2 + eps) - calibration_factor_db
    Optionally applies an adaptive Lee speckle filter.
    """
    arr = np.nan_to_num(raw_band.astype(np.float32), nan=1e-5, posinf=1e4, neginf=1e-5)
    arr = np.maximum(arr, 1e-5)
    if apply_lee:
        arr = apply_speckle_filter(arr, method="lee", window_size=5)
        arr = np.maximum(arr, 1e-5)
    db = 10.0 * np.log10(np.square(arr) + 1e-6) - calibration_factor_db
    return np.clip(db, -50.0, 20.0)


def detect_sar_water_backscatter(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Detect water bodies from Synthetic Aperture Radar (SAR) imagery.
    Water behaves as a specular reflector in radar wavelengths,
    returning very low backscatter (typically < -15 dB to -18 dB).
    """
    import rasterio

    with rasterio.open(path) as src:
        band1 = src.read(1).astype(np.float32)

    db = calibrate_sar_db(band1)

    # Filter out nodata margins / zeros
    valid = db[band1 > 1e-4]
    if len(valid) == 0:
        valid = db.flatten()

    p10 = float(np.percentile(valid, 10))
    p25 = float(np.percentile(valid, 25))
    threshold = min(-12.0, max(-26.0, p10 + 2.0))

    water_mask = (db < threshold).astype(np.uint8) * 255
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    active_pixels = int((water_mask > 0).sum())
    water_pct = 100.0 * active_pixels / float(max(water_mask.size, 1))

    diagnostics = {
        "mode": "sar_backscatter_water",
        "threshold_db": round(threshold, 2),
        "db_min": round(float(np.min(db)), 2),
        "db_max": round(float(np.max(db)), 2),
        "water_pixels": active_pixels,
        "water_percent": round(water_pct, 2),
    }

    method = f"SAR Low-Backscatter Water Extraction (Threshold: {threshold:.1f} dB)"
    return water_mask, method, 0.85, diagnostics


def detect_sar_builtup_backscatter(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """
    Detect built-up structures from SAR imagery.
    Urban settlements produce double-bounce corner reflection,
    yielding high backscatter (typically > -8 dB to -10 dB).
    """
    import rasterio

    with rasterio.open(path) as src:
        band1 = src.read(1).astype(np.float32)

    db = calibrate_sar_db(band1)
    valid = db[band1 > 1e-4]
    if len(valid) == 0:
        valid = db.flatten()

    p75 = float(np.percentile(valid, 75))
    p90 = float(np.percentile(valid, 90))
    threshold = max(-10.0, min(-4.0, p75))

    urban_mask = (db > threshold).astype(np.uint8) * 255
    urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    active_pixels = int((urban_mask > 0).sum())
    urban_pct = 100.0 * active_pixels / float(max(urban_mask.size, 1))

    diagnostics = {
        "mode": "sar_backscatter_builtup",
        "threshold_db": round(threshold, 2),
        "builtup_pixels": active_pixels,
        "builtup_percent": round(urban_pct, 2),
    }

    method = f"SAR High-Backscatter Double-Bounce Urban Extraction (Threshold: {threshold:.1f} dB)"
    return urban_mask, method, 0.82, diagnostics
