from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _determine_sector(cy: float, cx: float, h: int, w: int) -> str:
    """Classify pixel coordinates into a cardinal sector."""
    norm_y = cy / max(h, 1)
    norm_x = cx / max(w, 1)
    if 0.35 <= norm_y <= 0.65 and 0.35 <= norm_x <= 0.65:
        return "central sector"
    lat = "northern" if norm_y < 0.45 else ("southern" if norm_y > 0.55 else "")
    lon = "western" if norm_x < 0.45 else ("eastern" if norm_x > 0.55 else "")
    if lat and lon:
        return f"{lat}-{lon} sector"
    elif lat:
        return f"{lat} sector"
    elif lon:
        return f"{lon} sector"
    return "central sector"


def extract_change_clusters(
    change_mask: np.ndarray,
    cluster_type: str,
    target_value: int,
    pixel_resolution_m: float = 10.0,
    min_area_pixels: int = 15,
    max_clusters: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extract spatial connected component clusters for change areas (gain or loss).
    """
    binary = (change_mask == target_value).astype(np.uint8)
    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clusters: List[Dict[str, Any]] = []
    pixel_area_m2 = pixel_resolution_m * pixel_resolution_m
    pixel_to_ha = pixel_area_m2 / 10000.0

    for idx, cnt in enumerate(contours):
        area_px = float(cv2.contourArea(cnt))
        if area_px < min_area_pixels:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        area_ha = area_px * pixel_to_ha
        sector = _determine_sector(cy, cx, h, w)

        clusters.append({
            "id": idx + 1,
            "type": cluster_type,
            "area_pixels": int(area_px),
            "area_hectares": round(area_ha, 2),
            "bbox": [y, x, y + bh, x + bw],
            "centroid": [round(cy, 1), round(cx, 1)],
            "sector": sector,
        })

    # Sort descending by area
    clusters.sort(key=lambda c: c["area_pixels"], reverse=True)
    return clusters[:max_clusters]


def generate_cdvqa_answer(
    question: Optional[str],
    metrics: Dict[str, Any],
    dominant_sector: Optional[str] = None,
) -> str:
    """
    Synthesize a concise, publication-grade Change Detection Visual Question Answering (CDVQA) response.
    """
    q_lower = (question or "").lower()
    feature = metrics.get("feature", "surface")
    direction = metrics.get("direction", "remained unchanged")
    delta_pp = metrics.get("delta_percentage_points", 0.0)
    delta_ha = metrics.get("delta_hectares", 0.0)
    t1_pct = metrics.get("t1_coverage_percent", 0.0)
    t2_pct = metrics.get("t2_coverage_percent", 0.0)

    # Sector context
    sector_str = f", primarily concentrated in the {dominant_sector}" if dominant_sector else ""

    # Specific question intent checks
    if any(k in q_lower for k in ["did", "has", "is", "whether"]) and any(k in q_lower for k in ["increase", "expand", "grow"]):
        if direction == "increased":
            return f"Yes, {feature} coverage expanded significantly by +{abs(delta_pp):.1f}% (+{abs(delta_ha):.1f} ha){sector_str}."
        elif direction == "decreased":
            return f"No, {feature} coverage did not increase; it decreased by -{abs(delta_pp):.1f}% (-{abs(delta_ha):.1f} ha){sector_str}."
        else:
            return f"No, {feature} coverage remained approximately stable (delta: {delta_pp:+.1f}%, {delta_ha:+.1f} ha)."

    if any(k in q_lower for k in ["did", "has", "is", "whether"]) and any(k in q_lower for k in ["decrease", "shrink", "reduce", "loss"]):
        if direction == "decreased":
            return f"Yes, {feature} coverage decreased by -{abs(delta_pp):.1f}% (-{abs(delta_ha):.1f} ha){sector_str}."
        elif direction == "increased":
            return f"No, {feature} coverage did not decrease; it increased by +{abs(delta_pp):.1f}% (+{abs(delta_ha):.1f} ha){sector_str}."
        else:
            return f"No, {feature} coverage remained approximately stable (delta: {delta_pp:+.1f}%, {delta_ha:+.1f} ha)."

    if direction == "remained unchanged":
        return (
            f"{feature.capitalize()} coverage remained unchanged between observations "
            f"(T1: {t1_pct:.1f}%, T2: {t2_pct:.1f}%, delta: {delta_pp:+.1f}% / {delta_ha:+.1f} ha)."
        )
    elif direction == "increased":
        return (
            f"Bi-temporal change analysis reveals {feature} extent increased by +{abs(delta_pp):.1f}% "
            f"(+{abs(delta_ha):.1f} ha, shift from {t1_pct:.1f}% to {t2_pct:.1f}%){sector_str}."
        )
    else:
        return (
            f"Bi-temporal change analysis reveals {feature} extent decreased by -{abs(delta_pp):.1f}% "
            f"(-{abs(delta_ha):.1f} ha, shift from {t1_pct:.1f}% to {t2_pct:.1f}%){sector_str}."
        )


def compute_bitemporal_change(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    feature: str = "water",
    pixel_resolution_m: float = 10.0,
    question: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute pixel-level change metrics, directional reasoning, spatial clusters,
    and CDVQA response between two temporal observations (T1 -> T2).
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

    pixel_area_m2 = pixel_resolution_m * pixel_resolution_m
    pixel_to_ha = pixel_area_m2 / 10000.0

    t1_ha = t1_active * pixel_to_ha
    t2_ha = t2_active * pixel_to_ha
    gain_ha = gain_active * pixel_to_ha
    loss_ha = loss_active * pixel_to_ha
    delta_ha = t2_ha - t1_ha

    t1_pct = 100.0 * t1_active / total_pixels
    t2_pct = 100.0 * t2_active / total_pixels
    delta_pp = t2_pct - t1_pct

    relative_change_pct = (100.0 * (t2_active - t1_active) / float(max(t1_active, 1))) if t1_active > 0 else 0.0

    # Directional Head Classification (ISRO CDVQA standard)
    if abs(delta_pp) < 0.5:
        direction = "remained unchanged"
    elif delta_pp > 0:
        direction = "increased"
    else:
        direction = "decreased"

    # Extract spatial change clusters
    gain_clusters = extract_change_clusters(change_map, "gain", 255, pixel_resolution_m)
    loss_clusters = extract_change_clusters(change_map, "loss", 128, pixel_resolution_m)
    all_clusters = gain_clusters + loss_clusters
    all_clusters.sort(key=lambda c: c["area_pixels"], reverse=True)

    dominant_sector = all_clusters[0]["sector"] if all_clusters else None

    metrics: Dict[str, Any] = {
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
        "t1_hectares": round(t1_ha, 2),
        "t2_hectares": round(t2_ha, 2),
        "delta_hectares": round(delta_ha, 2),
        "gain_hectares": round(gain_ha, 2),
        "loss_hectares": round(loss_ha, 2),
        "clusters": all_clusters[:6],
        "dominant_sector": dominant_sector,
    }

    # Synthesize CDVQA answer
    metrics["cdvqa_answer"] = generate_cdvqa_answer(question, metrics, dominant_sector)

    return change_map, metrics


def compute_spectral_difference(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    index_type: str = "ndvi",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute differential spectral / feature difference vector between two co-registered scenes.
    Delta F = F_T2 - F_T1
    """
    t1_f = img_t1.astype(np.float32)
    t2_f = img_t2.astype(np.float32)
    if t1_f.shape != t2_f.shape:
        t2_f = cv2.resize(t2_f, (t1_f.shape[1], t1_f.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = t2_f - t1_f
    abs_diff = np.abs(diff)

    mean_diff = float(np.mean(diff))
    mean_abs_diff = float(np.mean(abs_diff))
    max_abs_diff = float(np.max(abs_diff))

    norm_diff = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    stats = {
        "index_type": index_type,
        "mean_diff": round(mean_diff, 4),
        "mean_absolute_diff": round(mean_abs_diff, 4),
        "max_absolute_diff": round(max_abs_diff, 4),
    }
    return norm_diff, stats


def make_bitemporal_overlay(
    rgb_base: np.ndarray,
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    draw_boxes: bool = False,
    clusters: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Generate a dual-color visual change overlay:
    - Red for reduction / loss
    - Cyan / Green for expansion / gain
    - Yellow / Teal for stable presence
    Optionally overlays change cluster bounding boxes.
    """
    canvas = np.asarray(rgb_base, dtype=np.uint8).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    if mask_t1.shape[:2] != canvas.shape[:2]:
        mask_t1 = cv2.resize(mask_t1, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_t2.shape[:2] != canvas.shape[:2]:
        mask_t2 = cv2.resize(mask_t2, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)

    t1_bool = mask_t1 > 0
    t2_bool = mask_t2 > 0

    gain = (~t1_bool) & t2_bool
    loss = t1_bool & (~t2_bool)
    stable = t1_bool & t2_bool

    overlay = canvas.copy()
    # Gain: Vibrant Cyan (0, 255, 255)
    overlay[gain] = [0, 255, 255]
    # Loss: Vibrant Red (255, 45, 45)
    overlay[loss] = [255, 45, 45]
    # Stable: Royal Blue/Teal (50, 140, 245)
    overlay[stable] = [50, 140, 245]

    blended = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)

    if draw_boxes and clusters:
        for c in clusters:
            y1, x1, y2, x2 = c["bbox"]
            color = (0, 255, 255) if c["type"] == "gain" else (255, 45, 45)
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
            label = f"{c['type'].upper()}: {c['area_hectares']} ha"
            cv2.putText(blended, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return blended

