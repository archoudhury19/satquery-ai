from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def align_mask_to_reference(
    src_mask: np.ndarray,
    src_data: Dict[str, Any],
    ref_data: Dict[str, Any],
) -> Tuple[np.ndarray, str]:
    """
    Geospatially align/reproject a source mask to the reference raster's
    CRS, transform, and pixel grid.
    """
    ref_h, ref_w = ref_data["height"], ref_data["width"]
    src_crs = src_data.get("crs")
    ref_crs = ref_data.get("crs")
    src_trans = src_data.get("transform")
    ref_trans = ref_data.get("transform")

    # If already identical grid and CRS, return directly
    if src_mask.shape == (ref_h, ref_w) and src_crs == ref_crs and src_trans == ref_trans:
        return src_mask.copy(), "direct_pixel_match"

    # Geospatial reprojection using rasterio.warp if CRS and transforms exist
    if src_crs and ref_crs and src_trans and ref_trans:
        try:
            from rasterio.warp import Resampling, reproject
            from rasterio.transform import Affine

            s_t = Affine(*src_trans) if not isinstance(src_trans, Affine) else src_trans
            r_t = Affine(*ref_trans) if not isinstance(ref_trans, Affine) else ref_trans

            dst = np.zeros((ref_h, ref_w), dtype=np.uint8)
            reproject(
                source=src_mask,
                destination=dst,
                src_transform=s_t,
                src_crs=src_crs,
                dst_transform=r_t,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
            return dst, f"geospatial_reproject ({src_crs} -> {ref_crs})"
        except Exception:
            pass

    # Fallback to nearest-neighbor resize
    resized = cv2.resize(src_mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
    return resized, "nearest_pixel_resize"


def fuse_cross_modal_masks(
    optical_mask: np.ndarray,
    sar_mask: np.ndarray,
    feature: str = "water",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Execute registration-tolerant cross-modal consensus fusion between
    optical and SAR masks.
    """
    opt_bool = optical_mask > 0
    sar_bool = sar_mask > 0

    # Apply small dilation kernel (5x5) to accommodate minor co-registration shifts
    kernel = np.ones((5, 5), np.uint8)
    opt_dilated = cv2.dilate((opt_bool.astype(np.uint8)), kernel) > 0
    sar_dilated = cv2.dilate((sar_bool.astype(np.uint8)), kernel) > 0

    # Cross-modal consensus (optical matched with nearby SAR evidence and vice versa)
    fused_bool = (opt_bool & sar_dilated) | (sar_bool & opt_dilated) | (opt_bool & sar_bool)
    fused_mask = (fused_bool.astype(np.uint8)) * 255

    opt_active = int(opt_bool.sum())
    sar_active = int(sar_bool.sum())
    fused_active = int(fused_bool.sum())
    union_active = int((opt_bool | sar_bool).sum())

    total_pixels = float(max(optical_mask.size, 1))
    opt_pct = 100.0 * opt_active / total_pixels
    sar_pct = 100.0 * sar_active / total_pixels
    fused_pct = 100.0 * fused_active / total_pixels

    agreement_pct = (100.0 * fused_active / float(max(union_active, 1))) if union_active > 0 else 0.0

    metrics = {
        "optical_pixels": opt_active,
        "sar_pixels": sar_active,
        "fused_pixels": fused_active,
        "union_pixels": union_active,
        "optical_coverage_pct": round(opt_pct, 2),
        "sar_coverage_pct": round(sar_pct, 2),
        "fused_coverage_pct": round(fused_pct, 2),
        "cross_modal_agreement_pct": round(agreement_pct, 2),
    }

    return fused_mask, metrics
