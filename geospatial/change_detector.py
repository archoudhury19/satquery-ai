from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def compute_bitemporal_change(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    feature: str = "water",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute pixel-level change metrics and change mask between two temporal observations (T1 -> T2).
    """
    t1_bool = mask_t1 > 0
    t2_bool = mask_t2 > 0

    gain = (~t1_bool) & t2_bool
    loss = t1_bool & (~t2_bool)
    stable = t1_bool & t2_bool

    # Change mask: 255 for gain, 128 for loss, 64 for stable
    change_map = np.zeros_like(mask_t1, dtype=np.uint8)
    change_map[gain] = 255
    change_map[loss] = 128
    change_map[stable] = 64

    total_pixels = float(max(mask_t1.size, 1))
    t1_active = int(t1_bool.sum())
    t2_active = int(t2_bool.sum())
    gain_active = int(gain.sum())
    loss_active = int(loss.sum())
    stable_active = int(stable.sum())

    t1_pct = 100.0 * t1_active / total_pixels
    t2_pct = 100.0 * t2_active / total_pixels
    delta_pp = t2_pct - t1_pct

    relative_change_pct = (100.0 * (t2_active - t1_active) / float(max(t1_active, 1))) if t1_active > 0 else 0.0

    if abs(delta_pp) < 0.5:
        direction = "remained unchanged"
    elif delta_pp > 0:
        direction = "increased"
    else:
        direction = "decreased"

    metrics = {
        "feature": feature,
        "t1_pixels": t1_active,
        "t2_pixels": t2_active,
        "gain_pixels": gain_active,
        "loss_pixels": loss_active,
        "stable_pixels": stable_active,
        "t1_coverage_percent": round(t1_pct, 2),
        "t2_coverage_percent": round(t2_pct, 2),
        "delta_percentage_points": round(delta_pp, 2),
        "relative_change_percent": round(relative_change_pct, 2),
        "direction": direction,
    }

    return change_map, metrics


def make_bitemporal_overlay(
    rgb_base: np.ndarray,
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
) -> np.ndarray:
    """
    Generate a dual-color visual change overlay:
    - Red for reduction / loss
    - Cyan / Green for expansion / gain
    - Yellow / Orange for stable presence
    """
    canvas = np.asarray(rgb_base, dtype=np.uint8).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    t1_bool = mask_t1 > 0
    t2_bool = mask_t2 > 0

    gain = (~t1_bool) & t2_bool
    loss = t1_bool & (~t2_bool)
    stable = t1_bool & t2_bool

    # Blend colors with alpha
    overlay = canvas.copy()
    # Gain: Cyan (0, 255, 255)
    overlay[gain] = [0, 255, 255]
    # Loss: Red (255, 50, 50)
    overlay[loss] = [255, 50, 50]
    # Stable: Blue/Teal (50, 150, 255)
    overlay[stable] = [50, 150, 255]

    blended = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)
    return blended
