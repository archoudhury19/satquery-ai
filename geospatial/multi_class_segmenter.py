"""
geospatial/multi_class_segmenter.py
=====================================
Multi-class land-cover segmentation with two modes:

  1. AI mode (default)  — GeoRSCLIP zero-shot patch classification.
     Uses the already-loaded GeoRSCLIP ViT-B/32 to classify each image
     tile against text prompts for water / vegetation / built-up / bare.

  2. Spectral fallback — Physics-based NDWI / NDVI / NDBI / RGB heuristics
     when the VLM model is not available.

Colour convention (RGB):
  - Water      : deep blue    ( 30, 100, 200 )
  - Vegetation : bright green ( 50, 200,  70 )
  - Built-up   : coral orange (220,  90,  40 )
  - Bare/Other : tan          (160, 130,  90 ) — shown faintly in AI mode

Priority order in spectral mode:
  water > vegetation > built-up
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from geospatial.water_detector import detect_remote_sensing_water, resolve_spectral_bands
from geospatial.vegetation_detector import detect_spectral_vegetation, detect_rgb_vegetation
from geospatial.builtup_detector import detect_spectral_builtup, detect_rgb_builtup


# ------------------------------------------------------------------ colours
COLOUR_WATER  = np.array([   0, 160, 255 ], dtype=np.uint8)   # Azure blue
COLOUR_VEG    = np.array([  34, 197,  94 ], dtype=np.uint8)   # Vibrant green
COLOUR_BUILT  = np.array([ 220,  38,  38 ], dtype=np.uint8)   # Brick/crimson red
COLOUR_DESERT = np.array([ 234, 179,   8 ], dtype=np.uint8)   # Golden sand / yellow-ochre
ALPHA         = 0.55                                            # overlay opacity


def segment_land_cover(
    path: Path,
    data: Dict[str, Any],
    vlm: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Multi-class land-cover segmentation.

    Uses physics-based spectral indices (NDWI / NDVI / NDBI / Sand spectral heuristics)
    for pixel-level classification.

    Returns
    -------
    overlay_rgb : H x W x 3 uint8 array (RGB, suitable for cv2.imwrite)
    stats       : dict with per-class pixel counts and percentages
    """

    # ---------------------------------------------------------------- water
    water_mask, water_method, _, water_diag = detect_remote_sensing_water(path, data)
    water_binary = (water_mask > 0)

    # ----------------------------------------------------------- vegetation
    try:
        veg_mask, veg_method, _, veg_diag = detect_spectral_vegetation(path, data)
    except Exception:
        veg_mask, veg_method, _, veg_diag = detect_rgb_vegetation(data)
    veg_binary = (veg_mask > 0)

    # ------------------------------------------------------------ built-up
    try:
        built_mask, built_method, _, built_diag = detect_spectral_builtup(path, data)
    except Exception:
        built_mask, built_method, _, built_diag = detect_rgb_builtup(data)
    built_binary = (built_mask > 0)

    # --------------------------------------------------------- base RGB
    rgb = np.asarray(data["rgb"], dtype=np.uint8).copy()
    if rgb.ndim == 2:                          # grayscale → replicate
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[2] > 3:
        rgb = rgb[..., :3]                     # drop alpha if present

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    # ----------------------------------------------------------- desert / sand
    # Desert / Sand Dunes: warm golden-yellow reflectance (high Red & Green, low Blue)
    desert_binary = (r > 95) & (g > 75) & (r >= b + 6) & (~water_binary) & (~veg_binary)

    # ----------------------------------------- priority: water > veg > built > desert
    veg_binary    = veg_binary    & ~water_binary
    built_binary  = built_binary  & ~water_binary & ~veg_binary & ~desert_binary
    desert_binary = desert_binary & ~water_binary & ~veg_binary

    overlay = rgb.copy().astype(np.float32)

    def _apply_colour(mask_bool: np.ndarray, colour: np.ndarray) -> None:
        """Paint mask pixels with colour blended on top of overlay."""
        overlay[mask_bool] = (
            ALPHA * colour.astype(np.float32)
            + (1.0 - ALPHA) * overlay[mask_bool]
        )

    _apply_colour(desert_binary, COLOUR_DESERT)
    _apply_colour(built_binary,  COLOUR_BUILT)
    _apply_colour(veg_binary,    COLOUR_VEG)
    _apply_colour(water_binary,  COLOUR_WATER)

    overlay_rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------- legend bar
    overlay_rgb = _add_legend(overlay_rgb)

    # ---------------------------------------------------------------- stats
    total = float(rgb.shape[0] * rgb.shape[1])
    desert_count = int(desert_binary.sum())
    stats = {
        "total_pixels": int(total),
        "water": {
            "pixels": int(water_binary.sum()),
            "percent": round(100.0 * water_binary.sum() / total, 2),
            "method": water_method,
        },
        "vegetation": {
            "pixels": int(veg_binary.sum()),
            "percent": round(100.0 * veg_binary.sum() / total, 2),
            "method": veg_method,
        },
        "built_up": {
            "pixels": int(built_binary.sum()),
            "percent": round(100.0 * built_binary.sum() / total, 2),
            "method": built_method,
        },
        "desert": {
            "pixels": desert_count,
            "percent": round(100.0 * desert_count / total, 2),
            "method": "Radiometric Sand/Dune Threshold",
        },
        "unclassified": {
            "pixels": int(total - water_binary.sum() - veg_binary.sum() - built_binary.sum() - desert_count),
            "percent": round(
                100.0 * max(0, total - water_binary.sum() - veg_binary.sum() - built_binary.sum() - desert_count) / total,
                2,
            ),
        },
    }

    return overlay_rgb, stats


def _add_legend(img: np.ndarray) -> np.ndarray:
    """
    Draw a small legend bar in the bottom-right corner of the image.
    """
    h, w = img.shape[:2]
    legend_items = [
        ("Water",       COLOUR_WATER),
        ("Vegetation",  COLOUR_VEG),
        ("Buildings",   COLOUR_BUILT),
        ("Desert/Sand", COLOUR_DESERT),
    ]

    box_size  = max(14, min(22, h // 30))
    font_scale = max(0.35, min(0.55, h / 800))
    pad       = max(6, h // 80)
    line_h    = box_size + pad
    total_h   = line_h * len(legend_items) + pad * 2
    total_w   = max(160, w // 5)

    x0 = w - total_w - pad
    y0 = h - total_h - pad
    x1 = w - pad
    y1 = h - pad

    # Semi-transparent background panel
    panel = img[y0:y1, x0:x1].astype(np.float32)
    panel = panel * 0.45 + np.array([10, 10, 20], np.float32) * 0.55
    img[y0:y1, x0:x1] = np.clip(panel, 0, 255).astype(np.uint8)

    for i, (label, colour) in enumerate(legend_items):
        cy = y0 + pad + i * line_h
        cx = x0 + pad
        # Coloured box
        img[cy:cy + box_size, cx:cx + box_size] = colour
        # Label
        cv2.putText(
            img,
            label,
            (cx + box_size + 4, cy + box_size - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return img
