from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def apply_speckle_filter(
    sar_image: np.ndarray,
    method: str = "lee",
    window_size: int = 5,
    noise_variance: float = 0.25,
) -> np.ndarray:
    """
    Apply speckle noise reduction to SAR amplitude / intensity images.
    Supported methods: 'lee' (Adaptive Lee Filter) or 'median' (Median Filter).
    """
    img = np.nan_to_num(sar_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    k = max(3, window_size if window_size % 2 == 1 else window_size + 1)

    if method.lower() == "median":
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filtered_norm = cv2.medianBlur(norm_img, k)
        # Rescale back to original range
        min_v, max_v = float(np.min(img)), float(np.max(img))
        if max_v > min_v:
            return min_v + (filtered_norm.astype(np.float32) / 255.0) * (max_v - min_v)
        return img

    # Adaptive Lee Speckle Filter
    # W = var_local / (var_local + noise_variance)
    # y = mean_local + W * (x - mean_local)
    mean = cv2.boxFilter(img, -1, (k, k), normalize=True)
    mean_sq = cv2.boxFilter(img * img, -1, (k, k), normalize=True)
    var = np.maximum(0.0, mean_sq - mean * mean)

    # Adaptive weights clamped between 0 and 1
    denom = var + noise_variance
    weight = np.where(denom > 1e-7, var / denom, 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = mean + weight * (img - mean)
    return filtered.astype(sar_image.dtype if np.issubdtype(sar_image.dtype, np.floating) else np.float32)


def calibrate_sar_db(
    raw_band: np.ndarray,
    calibration_factor_db: float = 0.0,
    apply_lee: bool = True,
) -> np.ndarray:
    """
    Convert raw Sentinel-1 / RISAT SAR DN values to calibrated backscatter (dB) scale:
    sigma^0_dB = 10 * log10(DN^2 + eps) - calibration_factor_db
    Optionally applies an adaptive Lee speckle filter prior to log-transformation.
    """
    arr = np.nan_to_num(raw_band.astype(np.float32), nan=1e-5, posinf=1e4, neginf=1e-5)
    arr = np.maximum(arr, 1e-5)

    if apply_lee:
        arr = apply_speckle_filter(arr, method="lee", window_size=5)
        arr = np.maximum(arr, 1e-5)

    # sigma^0 = 10 * log10(DN^2) = 20 * log10(DN)
    db = 10.0 * np.log10(np.square(arr) + 1e-6) - calibration_factor_db
    return np.clip(db, -50.0, 20.0)


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
    dilation_kernel_size: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Execute registration-tolerant cross-modal consensus fusion between
    optical and SAR masks.
    """
    if optical_mask.shape != sar_mask.shape:
        sar_mask = cv2.resize(sar_mask, (optical_mask.shape[1], optical_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    opt_bool = optical_mask > 0
    sar_bool = sar_mask > 0

    # Apply small dilation kernel to accommodate sub-pixel co-registration errors
    k = max(3, dilation_kernel_size)
    kernel = np.ones((k, k), np.uint8)
    opt_dilated = cv2.dilate((opt_bool.astype(np.uint8)), kernel) > 0
    sar_dilated = cv2.dilate((sar_bool.astype(np.uint8)), kernel) > 0

    # Cross-modal consensus (optical matched with nearby SAR evidence and vice versa)
    fused_bool = (opt_bool & sar_dilated) | (sar_bool & opt_dilated) | (opt_bool & sar_bool)
    fused_mask = (fused_bool.astype(np.uint8)) * 255

    opt_active = int(opt_bool.sum())
    sar_active = int(sar_bool.sum())
    fused_active = int(fused_bool.sum())
    union_active = int((opt_bool | sar_bool).sum())
    intersection_active = int((opt_bool & sar_bool).sum())

    total_pixels = float(max(optical_mask.size, 1))
    opt_pct = 100.0 * opt_active / total_pixels
    sar_pct = 100.0 * sar_active / total_pixels
    fused_pct = 100.0 * fused_active / total_pixels

    agreement_pct = (100.0 * fused_active / float(max(union_active, 1))) if union_active > 0 else 0.0
    iou_pct = (100.0 * intersection_active / float(max(union_active, 1))) if union_active > 0 else 0.0

    metrics = {
        "feature": feature,
        "optical_pixels": opt_active,
        "sar_pixels": sar_active,
        "fused_pixels": fused_active,
        "union_pixels": union_active,
        "intersection_pixels": intersection_active,
        "optical_coverage_pct": round(opt_pct, 2),
        "sar_coverage_pct": round(sar_pct, 2),
        "fused_coverage_pct": round(fused_pct, 2),
        "cross_modal_agreement_pct": round(agreement_pct, 2),
        "cross_modal_iou_pct": round(iou_pct, 2),
    }

    return fused_mask, metrics


def make_fusion_overlay(
    optical_rgb: np.ndarray,
    sar_band: Optional[np.ndarray],
    fused_mask: np.ndarray,
    optical_mask: Optional[np.ndarray] = None,
    sar_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate multi-channel false-color consensus overlay:
    - Cyan: Consensus agreement between Optical & SAR
    - Amber/Yellow: Optical evidence only
    - Magenta: SAR penetration/backscatter evidence only
    """
    canvas = np.asarray(optical_rgb, dtype=np.uint8).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    overlay = canvas.copy()
    if optical_mask is not None and sar_mask is not None:
        opt_b = cv2.resize(optical_mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        sar_b = cv2.resize(sar_mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        consensus = opt_b & sar_b
        opt_only = opt_b & (~sar_b)
        sar_only = sar_b & (~opt_b)

        overlay[opt_only] = [255, 190, 0]    # Amber/Yellow
        overlay[sar_only] = [220, 30, 220]   # Magenta
        overlay[consensus] = [0, 240, 255]   # Vibrant Cyan
    else:
        fused_b = cv2.resize(fused_mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        overlay[fused_b] = [0, 240, 255]

    return cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)

