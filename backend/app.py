from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pyproj import CRS, Transformer
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from agent.planner import build_plan
from geospatial import (
    detect_remote_sensing_water,
    detect_remote_sensing_vegetation,
    detect_remote_sensing_builtup,
    calibrate_sar_db,
    detect_sar_water_backscatter,
    detect_sar_builtup_backscatter,
    align_mask_to_reference,
    fuse_cross_modal_masks,
    compute_bitemporal_change,
    make_bitemporal_overlay,
)
from models.rs_vlm import RemoteSensingVLM


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOWED_EXT = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}

app = FastAPI(
    title="SatQuery AI MVP",
    version="0.7.0",
)

# Load the local adapted RS model once.
RS_VLM = RemoteSensingVLM()

# MVP in-memory state.
FILES: Dict[str, Dict[str, Any]] = {}
CONTEXT: Dict[str, Dict[str, Any]] = {}


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    primary_id: str
    secondary_id: Optional[str] = None
    query: str
    conversation_id: Optional[str] = None


# ============================================================
# IMAGE / RASTER UTILITIES
# ============================================================

def _normalize_band(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Robust 2%-98% percentile normalization for display.
    """

    a = arr.astype(
        np.float32
    )

    finite = np.isfinite(a)

    if not finite.any():
        return np.zeros_like(
            a,
            dtype=np.uint8,
        )

    lo, hi = np.percentile(
        a[finite],
        [2, 98],
    )

    if hi <= lo:
        lo = float(
            np.nanmin(a)
        )
        hi = float(
            np.nanmax(a)
        )

    if hi <= lo:
        return np.zeros_like(
            a,
            dtype=np.uint8,
        )

    a = np.clip(
        (a - lo) / (hi - lo),
        0.0,
        1.0,
    )

    return (
        a * 255
    ).astype(
        np.uint8
    )


def _read_raster(
    path: Path,
) -> Dict[str, Any]:
    """
    Read GeoTIFF/TIFF and collect:
      - RGB preview
      - dimensions
      - band count
      - CRS
      - transform
      - bounds
      - band descriptions
      - driver
      - georeferencing state
    """

    with rasterio.open(path) as src:

        count = src.count

        descriptions = [
            description or ""
            for description
            in src.descriptions
        ]

        if count >= 3:

            channels = [
                _normalize_band(
                    src.read(index)
                )
                for index in (
                    1,
                    2,
                    3,
                )
            ]

            rgb = np.stack(
                channels,
                axis=-1,
            )

        else:

            band = _normalize_band(
                src.read(1)
            )

            rgb = np.stack(
                [
                    band,
                    band,
                    band,
                ],
                axis=-1,
            )

        return {
            "rgb": rgb,
            "width": src.width,
            "height": src.height,
            "count": count,
            "crs": (
                src.crs.to_string()
                if src.crs
                else None
            ),
            "transform": tuple(
                src.transform
            ),
            "bounds": [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            "descriptions": descriptions,
            "dtypes": list(src.dtypes),
            "driver": src.driver,
            "is_georeferenced": bool(
                src.crs
            ),
        }


def _read_standard(
    path: Path,
) -> Dict[str, Any]:
    """
    Read PNG/JPEG as RGB.
    """

    bgr = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise ValueError(
            "Could not decode image."
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = rgb.shape[:2]

    return {
        "rgb": rgb,
        "width": width,
        "height": height,
        "count": 3,
        "crs": None,
        "transform": None,
        "bounds": None,
        "descriptions": [
            "R",
            "G",
            "B",
        ],
        "dtypes": [
            "uint8",
            "uint8",
            "uint8",
        ],
        "driver": (
            path.suffix
            .lower()
            .lstrip(".")
        ),
        "is_georeferenced": False,
    }


def read_image(
    path: Path,
) -> Dict[str, Any]:

    suffix = path.suffix.lower()

    if suffix in {
        ".tif",
        ".tiff",
    }:
        return _read_raster(path)

    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        return _read_standard(path)

    raise ValueError(
        f"Unsupported image format: {suffix}"
    )


def save_preview(
    rgb: np.ndarray,
    stem: str,
) -> str:

    output = (
        GENERATED_DIR
        / f"{stem}_preview.png"
    )

    cv2.imwrite(
        str(output),
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    return output.name


def _find_band_index(
    descriptions: List[str],
    keys: List[str],
) -> Optional[int]:

    for index, description in enumerate(
        descriptions,
        start=1,
    ):

        value = (
            description
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if any(
            key in value
            for key in keys
        ):
            return index

    return None


# ============================================================
# FEATURE DETECTION
# ============================================================

def detect_water(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, str, float]:
    """
    Robust remote-sensing water detection engine.

    Delegates to geospatial.water_detector.detect_remote_sensing_water,
    which provides:
    - Multispectral Sentinel-2 band resolution (Green B03, NIR B08, Red B04, Blue B02)
    - McFeeters NDWI + NDVI vegetation suppression + NIR absorption gating
    - Dynamic Otsu / zero-crossing thresholding
    - Robust RGB fallback when multispectral bands are absent
    """
    mask, method, conf, _ = detect_remote_sensing_water(path, data)
    return mask, method, conf


def detect_vegetation(
    data: Dict[str, Any],
) -> np.ndarray:

    rgb = data["rgb"].astype(
        np.int16
    )

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    mask = (
        (green > red + 12)
        & (green > blue + 8)
    ).astype(
        np.uint8
    ) * 255

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8,
        ),
    )


def detect_builtup(
    data: Dict[str, Any],
) -> np.ndarray:

    rgb = data["rgb"]

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    edges = cv2.Canny(
        gray,
        70,
        150,
    )

    edge_density = cv2.blur(
        (edges > 0).astype(
            np.float32
        ),
        (9, 9),
    )

    mask = (
        (hsv[:, :, 1] < 65)
        & (hsv[:, :, 2] > 75)
        & (edge_density > 0.05)
    ).astype(
        np.uint8
    ) * 255

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (5, 5),
            np.uint8,
        ),
    )


def feature_mask(
    feature: str,
    path: Path,
    data: Dict[str, Any],
) -> Tuple[
    np.ndarray,
    str,
    float,
]:

    if feature == "water":
        mask, method, conf, _ = detect_remote_sensing_water(path, data)
        return mask, method, conf

    if feature == "vegetation":
        mask, method, conf, _ = detect_remote_sensing_vegetation(path, data)
        return mask, method, conf

    if feature == "built-up":
        mask, method, conf, _ = detect_remote_sensing_builtup(path, data)
        return mask, method, conf

    mask, method, conf, _ = detect_remote_sensing_water(path, data)
    return mask, method, conf


def infer_feature(
    query: str,
    context: Dict[str, Any],
) -> str:

    q = query.lower()

    if any(
        key in q
        for key in [
            "water",
            "reservoir",
            "lake",
            "river",
            "pond",
            "flood",
        ]
    ):
        return "water"

    if any(
        key in q
        for key in [
            "built",
            "building",
            "urban",
            "construction",
            "settlement",
        ]
    ):
        return "built-up"

    if any(
        key in q
        for key in [
            "vegetation",
            "forest",
            "green",
            "crop",
            "agriculture",
        ]
    ):
        return "vegetation"

    if re.search(
        r"\b(it|this|that|same region|same area)\b",
        q,
    ):

        if context.get(
            "last_feature"
        ):
            return context[
                "last_feature"
            ]

    return context.get(
        "last_feature",
        "water",
    )


# ============================================================
# SPATIAL EVIDENCE
# ============================================================

def mask_stats(
    mask: np.ndarray,
) -> Dict[str, Any]:

    total = mask.size

    active = int(
        (mask > 0).sum()
    )

    fraction = (
        active / total
        if total
        else 0.0
    )

    return {
        "pixels": active,
        "fraction": fraction,
        "percent": 100.0 * fraction,
    }


def _largest_component(
    mask: np.ndarray,
) -> np.ndarray:

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (mask > 0).astype(
                np.uint8
            ),
            8,
        )
    )

    if count <= 1:
        return np.zeros_like(
            mask
        )

    index = 1 + int(
        np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
        )
    )

    return (
        labels == index
    ).astype(
        np.uint8
    ) * 255


def spatial_evidence(
    mask: np.ndarray,
    data: Dict[str, Any],
) -> Dict[str, Any]:

    component = _largest_component(
        mask
    )

    if (
        (component > 0).sum()
        < 20
    ):
        return {
            "available": False,
        }

    ys, xs = np.where(
        component > 0
    )

    cx_px = float(
        xs.mean()
    )

    cy_px = float(
        ys.mean()
    )

    location_x = (
        "west"
        if cx_px < data["width"] / 3
        else "east"
        if cx_px
        > 2 * data["width"] / 3
        else "central"
    )

    location_y = (
        "north"
        if cy_px < data["height"] / 3
        else "south"
        if cy_px
        > 2 * data["height"] / 3
        else "central"
    )

    if (
        location_x == "central"
        and location_y == "central"
    ):
        location = "central"

    elif location_x == "central":
        location = location_y

    elif location_y == "central":
        location = location_x

    else:
        location = (
            f"{location_y}-{location_x}"
        )

    x1 = int(
        xs.min()
    )

    y1 = int(
        ys.min()
    )

    x2 = int(
        xs.max()
    )

    y2 = int(
        ys.max()
    )

    evidence: Dict[str, Any] = {
        "available": True,

        "pixel_centroid": {
            "x": round(
                cx_px,
                1,
            ),
            "y": round(
                cy_px,
                1,
            ),
        },

        "pixel_bounding_box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },

        "location": location,

        "area_hectares": None,

        "centroid_wgs84": None,

        "geojson": None,
    }

    if not data.get(
        "is_georeferenced"
    ):
        return evidence

    transform_tuple = data.get(
        "transform"
    )

    crs_text = data.get(
        "crs"
    )

    if (
        not transform_tuple
        or not crs_text
    ):
        return evidence

    from affine import Affine

    affine_transform = Affine(
        *transform_tuple[:6]
    )

    geometries = []

    for geom, value in shapes(
        (
            component > 0
        ).astype(
            np.uint8
        ),
        mask=component > 0,
        transform=affine_transform,
    ):

        if value == 1:

            geometries.append(
                shape(geom)
            )

    if not geometries:
        return evidence

    geometry = max(
        geometries,
        key=lambda item: item.area,
    )

    source_crs = CRS.from_user_input(
        crs_text
    )

    try:

        to_wgs84 = (
            Transformer.from_crs(
                source_crs,
                "EPSG:4326",
                always_xy=True,
            ).transform
        )

        geometry_wgs84 = (
            shapely_transform(
                to_wgs84,
                geometry,
            )
        )

        centroid = (
            geometry_wgs84.centroid
        )

        evidence[
            "centroid_wgs84"
        ] = {
            "lat": round(
                centroid.y,
                6,
            ),
            "lon": round(
                centroid.x,
                6,
            ),
        }

        evidence[
            "geojson"
        ] = mapping(
            geometry_wgs84
        )

    except Exception:
        pass

    try:

        to_equal_area = (
            Transformer.from_crs(
                source_crs,
                "EPSG:6933",
                always_xy=True,
            ).transform
        )

        geometry_m = (
            shapely_transform(
                to_equal_area,
                geometry,
            )
        )

        evidence[
            "area_hectares"
        ] = round(
            abs(
                geometry_m.area
            )
            / 10000.0,
            3,
        )

    except Exception:
        pass

    return evidence


def make_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    name: str,
    change_mask: Optional[
        np.ndarray
    ] = None,
) -> str:

    output = rgb.copy()

    if change_mask is None:

        overlay = np.zeros_like(
            output
        )

        overlay[:, :, 2] = 255

        alpha = 0.42

        selected = mask > 0

        output[selected] = (
            (
                1 - alpha
            )
            * output[selected]
            + alpha
            * overlay[selected]
        ).astype(
            np.uint8
        )

        contours, _ = (
            cv2.findContours(
                (
                    mask > 0
                ).astype(
                    np.uint8
                ),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )

        bgr = cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR,
        )

        cv2.drawContours(
            bgr,
            contours,
            -1,
            (0, 255, 255),
            2,
        )

    else:

        overlay = np.zeros_like(
            output
        )

        overlay[:, :, 0] = 255

        selected = (
            change_mask > 0
        )

        output[selected] = (
            0.55
            * output[selected]
            + 0.45
            * overlay[selected]
        ).astype(
            np.uint8
        )

        bgr = cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR,
        )

    output_path = (
        GENERATED_DIR
        / (
            f"{name}_"
            f"{uuid.uuid4().hex[:8]}"
            ".png"
        )
    )

    cv2.imwrite(
        str(output_path),
        bgr,
    )

    return output_path.name


# ============================================================
# SINGLE IMAGE
# ============================================================

def scene_caption(
    data: Dict[str, Any],
) -> Tuple[str, float]:

    rgb = data["rgb"]

    vegetation_percent = (
        mask_stats(
            detect_vegetation(
                data
            )
        )["percent"]
    )

    built_percent = (
        mask_stats(
            detect_builtup(
                data
            )
        )["percent"]
    )

    mean = rgb.mean(
        axis=(0, 1)
    )

    pieces: List[str] = []

    if vegetation_percent > 18:

        pieces.append(
            "substantial vegetation"
        )

    if built_percent > 8:

        pieces.append(
            "textured built-up/constructed regions"
        )

    if mean[2] > mean[0] + 5:

        pieces.append(
            "blue-toned areas that may include water"
        )

    if not pieces:

        pieces.append(
            "mixed land-cover patterns"
        )

    return (
        (
            "Prototype scene summary: "
            + ", ".join(pieces)
            + "."
        ),
        0.58,
    )


def analyze_single(
    path: Path,
    data: Dict[str, Any],
    feature: str,
    task: str,
    query: str,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Captioning baseline
    # --------------------------------------------------------

    if task == "captioning":

        answer, confidence = (
            scene_caption(data)
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "tool": (
                "Remote-Sensing Scene "
                "Caption Prototype"
            ),
            "overlay": None,
            "evidence": None,
        }

    # --------------------------------------------------------
    # Real adapted RS-VQA
    # --------------------------------------------------------

    if RS_VLM.available:

        try:

            result = RS_VLM.analyze(
                image_path=path,
                question=query,
            )

            overlay = None
            evidence = result.get("evidence")

            # Attach visual grounding overlay and spatial evidence when querying about a specific feature
            if feature and feature not in {"auto", "scene", "multimodal", None}:
                f_mask, _, _ = feature_mask(feature, path, data)
                if (f_mask > 0).sum() >= 20:
                    evidence = spatial_evidence(f_mask, data)
                    overlay = make_overlay(
                        data["rgb"],
                        f_mask,
                        feature.replace("-", "_"),
                    )

            return {
                "answer": result.get(
                    "answer",
                    "",
                ),
                "confidence": float(
                    result.get(
                        "confidence",
                        0.0,
                    )
                ),
                "tool": result.get(
                    "model",
                    "GeoRSCLIP + RSVQA Adapter",
                ),
                "overlay": overlay,
                "evidence": evidence,
                "top_answers": result.get(
                    "top_answers",
                    [],
                ),
            }

        except Exception as exc:

            print(
                "[RS-VLM] Inference failed:",
                exc,
            )

    # --------------------------------------------------------
    # Scene fallback
    # --------------------------------------------------------

    if feature in {
        "auto",
        "scene",
        "multimodal",
        None,
    }:

        answer, confidence = (
            scene_caption(data)
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "tool": (
                "Remote-Sensing Scene "
                "Caption Fallback"
            ),
            "overlay": None,
            "evidence": None,
        }

    # --------------------------------------------------------
    # Feature-specific fallback
    # --------------------------------------------------------

    mask, method, base_conf = (
        feature_mask(
            feature,
            path,
            data,
        )
    )

    stats = mask_stats(
        mask
    )

    evidence = spatial_evidence(
        mask,
        data,
    )

    overlay = make_overlay(
        data["rgb"],
        mask,
        feature.replace(
            "-",
            "_",
        ),
    )

    exists = (
        stats["percent"]
        > (
            0.8
            if feature == "water"
            else 1.5
        )
    )

    if exists:

        area_phrase = ""

        if (
            evidence.get(
                "area_hectares"
            )
            is not None
        ):

            area_phrase = (
                " The largest detected "
                "region is approximately "
                f"{evidence['area_hectares']:.2f} "
                "hectares."
            )

        answer = (
            f"Yes — the prototype detected "
            f"{feature} evidence, covering "
            f"about {stats['percent']:.1f}% "
            "of the image."
            f"{area_phrase}"
        )

    else:

        answer = (
            "The prototype did not find "
            f"strong {feature} evidence "
            "in this image."
        )

    confidence = min(
        0.95,
        max(
            0.35,
            base_conf
            + min(
                stats["percent"]
                / 250.0,
                0.12,
            ),
        ),
    )

    return {
        "answer": answer,
        "confidence": round(
            confidence,
            2,
        ),
        "tool": method,
        "overlay": overlay,
        "evidence": evidence,
        "mask_stats": stats,
    }


# ============================================================
# HYBRID GROUNDING
# ============================================================

def analyze_grounding(
    path: Path,
    data: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:
    """
    Hybrid remote-sensing grounding.

    The query determines the target feature.
    A spectral/CV remote-sensing mask provides spatial
    localization, while the trained RS model remains the
    semantic VLM component used elsewhere in SatQuery.

    This is an MVP grounding implementation, not a
    pixel-perfect segmentation model.
    """

    q = query.lower().strip()

    # --------------------------------------------------------
    # Determine requested feature.
    # --------------------------------------------------------

    if any(
        term in q
        for term in [
            "water",
            "reservoir",
            "lake",
            "river",
            "pond",
            "flood",
        ]
    ):

        feature = "water"

    elif any(
        term in q
        for term in [
            "vegetation",
            "forest",
            "crop",
            "agriculture",
            "green area",
        ]
    ):

        feature = "vegetation"

    elif any(
        term in q
        for term in [
            "built-up",
            "built up",
            "building",
            "buildings",
            "urban",
            "settlement",
            "construction",
        ]
    ):

        feature = "built-up"

    else:

        feature = infer_feature(
            query,
            {},
        )

    # --------------------------------------------------------
    # Generate actual pixel-level candidate mask.
    # --------------------------------------------------------

    mask, method, base_conf = feature_mask(
        feature,
        path,
        data,
    )

    stats = mask_stats(
        mask
    )

    component = _largest_component(
        mask
    )

    component_pixels = int(
        (component > 0).sum()
    )

    active_mask = mask if (mask > 0).sum() >= 20 else component

    # --------------------------------------------------------
    # Spatial evidence.
    # --------------------------------------------------------

    evidence = spatial_evidence(
        active_mask,
        data,
    )

    # --------------------------------------------------------
    # Bounding box.
    # --------------------------------------------------------

    ys, xs = np.where(
        active_mask > 0
    )

    bounding_box = None

    if (
        len(xs) > 0
        and len(ys) > 0
    ):

        bounding_box = {
            "x1": int(
                xs.min()
            ),
            "y1": int(
                ys.min()
            ),
            "x2": int(
                xs.max()
            ),
            "y2": int(
                ys.max()
            ),
        }

    # --------------------------------------------------------
    # Visual evidence.
    # --------------------------------------------------------

    overlay = make_overlay(
        data["rgb"],
        active_mask,
        f"grounding_{feature.replace('-', '_')}",
    )

    detected = (
        component_pixels >= 20
        and stats["percent"] > 0.5
    )

    # --------------------------------------------------------
    # Answer.
    # --------------------------------------------------------

    if detected:

        answer = (
            f"Detected {feature} in the image."
        )

        location = evidence.get(
            "location"
        )

        if location:

            answer += (
                f" The main detected "
                f"region is in the "
                f"{location} part of "
                "the image."
            )

        area = evidence.get(
            "area_hectares"
        )

        if area is not None:

            answer += (
                f" Estimated area: "
                f"{area:.2f} hectares."
            )

    else:

        answer = (
            f"No strong {feature} "
            "region was detected."
        )

    # --------------------------------------------------------
    # Conservative confidence.
    # --------------------------------------------------------

    if detected:

        confidence = min(
            0.92,
            max(
                0.35,
                base_conf
                + min(
                    stats["percent"]
                    / 300.0,
                    0.12,
                ),
            ),
        )

    else:

        confidence = min(
            0.50,
            max(
                0.20,
                base_conf * 0.5,
            ),
        )

    return {
        "answer": answer,
        "confidence": round(
            confidence,
            3,
        ),
        "tool": (
            "Hybrid Remote-Sensing Grounding"
        ),
        "feature": feature,
        "method": method,
        "bounding_box": bounding_box,
        "location": evidence.get(
            "location"
        ),
        "overlay": overlay,
        "evidence": evidence,
        "mask_stats": stats,
    }


# ============================================================
# BI-TEMPORAL CHANGE
# ============================================================

def analyze_change(
    path1: Path,
    data1: Dict[str, Any],
    path2: Path,
    data2: Dict[str, Any],
    feature: str,
) -> Dict[str, Any]:

    if feature in {
        "auto",
        "scene",
        "multimodal",
        None,
    }:

        feature = "water"

    mask1, method1, conf1 = (
        feature_mask(
            feature,
            path1,
            data1,
        )
    )

    mask2, method2, conf2 = (
        feature_mask(
            feature,
            path2,
            data2,
        )
    )

    if mask2.shape != mask1.shape:

        mask2 = cv2.resize(
            mask2,
            (
                mask1.shape[1],
                mask1.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    stats1 = mask_stats(
        mask1
    )

    stats2 = mask_stats(
        mask2
    )

    delta = (
        stats2["percent"]
        - stats1["percent"]
    )

    relative_change = (
        None
        if stats1["percent"] < 1e-6
        else (
            delta
            / stats1["percent"]
        )
        * 100.0
    )

    changed = cv2.absdiff(
        (
            mask1 > 0
        ).astype(
            np.uint8
        ) * 255,
        (
            mask2 > 0
        ).astype(
            np.uint8
        ) * 255,
    )

    overlay = make_overlay(
        data2["rgb"],
        mask2,
        "change",
        changed,
    )

    evidence_before = (
        spatial_evidence(
            mask1,
            data1,
        )
    )

    evidence_after = (
        spatial_evidence(
            mask2,
            data2,
        )
    )

    if abs(delta) < 0.4:

        direction = (
            "remained approximately stable"
        )

    elif delta > 0:

        direction = "increased"

    else:

        direction = "decreased"

    area_sentence = ""

    before_area = (
        evidence_before.get(
            "area_hectares"
        )
    )

    after_area = (
        evidence_after.get(
            "area_hectares"
        )
    )

    if (
        before_area is not None
        and after_area is not None
    ):

        area_pct = (
            None
            if before_area == 0
            else (
                (
                    after_area
                    - before_area
                )
                / before_area
            )
            * 100.0
        )

        if area_pct is not None:

            area_sentence = (
                " Estimated largest-region "
                f"area: {before_area:.2f} ha "
                f"→ {after_area:.2f} ha "
                f"({area_pct:+.1f}%)."
            )

    relative_sentence = (
        ""
        if relative_change is None
        else (
            " Relative mask change: "
            f"{relative_change:+.1f}%."
        )
    )

    answer = (
        f"The detected {feature} area "
        f"{direction}. "
        f"Image coverage changed from "
        f"{stats1['percent']:.1f}% to "
        f"{stats2['percent']:.1f}%."
        f"{relative_sentence}"
        f"{area_sentence}"
    )

    return {
        "answer": answer,
        "confidence": round(
            min(
                0.92,
                (
                    conf1
                    + conf2
                ) / 2.0,
            ),
            2,
        ),
        "tool": (
            "Bi-temporal "
            f"{feature} comparison "
            f"({method1} + {method2})"
        ),
        "overlay": overlay,
        "evidence": {
            "before": evidence_before,
            "after": evidence_after,
            "delta_percentage_points": round(
                delta,
                2,
            ),
        },
        "mask_stats": {
            "before": stats1,
            "after": stats2,
        },
    }


# ============================================================
# MODALITY / SAR UTILITIES
# ============================================================

def infer_modality(
    path: Path,
    data: Dict[str, Any],
    original_filename: Optional[str] = None,
) -> str:
    """
    Infer a conservative modality label.

    IMPORTANT:
    Uploaded files are stored internally with random UUID
    filenames, so the original upload filename must be passed
    when available. This is what lets files such as
    ``sentinel1_vv.tif`` be recognized as SAR.

    This remains heuristic classification, not a learned
    modality classifier.
    """

    filename_candidates = [
        str(original_filename or ""),
        str(
            data.get(
                "original_filename",
                "",
            )
        ),
        str(path.name),
    ]

    descriptions = " ".join(
        str(x)
        for x in data.get(
            "descriptions",
            [],
        )
    )

    text = (
        " ".join(filename_candidates)
        + " "
        + descriptions
    ).lower()

    sar_tokens = [
        "sentinel-1",
        "sentinel1",
        "sentinel_1",
        "s1a_",
        "s1b_",
        "s1c_",
        "s1d_",
        " sar",
        "sar ",
        "_sar",
        "-sar",
        "vv",
        "vh",
        "hv",
        "hh",
        "risat",
        "alos",
        "palsar",
        "radar",
    ]

    if any(
        token in text
        for token in sar_tokens
    ):
        return "sar"

    # A single-band uint16/float32 GeoTIFF is a strong signal
    # for a SAR backscatter product, but not proof by itself.
    if (
        path.suffix.lower() in {
            ".tif",
            ".tiff",
        }
        and int(
            data.get(
                "count",
                0,
            )
        ) == 1
    ):
        dtype_text = " ".join(
            str(x).lower()
            for x in data.get(
                "dtypes",
                [],
            )
        )

        if any(
            token in dtype_text
            for token in [
                "uint16",
                "float32",
                "float64",
            ]
        ):
            return "sar"

    return "optical"


def read_sar_backscatter(
    path: Path,
) -> Dict[str, Any]:
    """
    Read a single-band SAR backscatter/intensity raster.

    The processor uses log10 scaling and percentile
    normalization. It is suitable for Sentinel-1-style
    single-band VV/VH GeoTIFFs.

    This does not convert an optical image into SAR.
    """

    with rasterio.open(path) as src:

        if src.count < 1:
            raise ValueError(
                "SAR raster has no bands."
            )

        band = src.read(
            1
        ).astype(
            np.float32
        )

        finite = np.isfinite(
            band
        )

        if not finite.any():
            raise ValueError(
                "SAR raster contains no finite pixels."
            )

        positive = band > 0

        if not positive.any():
            raise ValueError(
                "SAR raster contains no positive backscatter values."
            )

        safe = np.where(
            positive,
            band,
            np.nan,
        )

        db = (
            10.0
            * np.log10(
                np.maximum(
                    safe,
                    1e-6,
                )
            )
        )

        finite_db = np.isfinite(
            db
        )

        lo, hi = np.percentile(
            db[finite_db],
            [2, 98],
        )

        if hi <= lo:
            hi = lo + 1e-6

        normalized = np.clip(
            (
                db - lo
            )
            / (
                hi - lo
            ),
            0.0,
            1.0,
        )

        normalized_uint8 = (
            np.nan_to_num(
                normalized,
                nan=0.0,
            )
            * 255.0
        ).astype(
            np.uint8
        )

        return {
            "raw": band,
            "db": db,
            "normalized": normalized_uint8,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": (
                src.crs.to_string()
                if src.crs
                else None
            ),
            "transform": tuple(
                src.transform
            ),
            "bounds": [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            "descriptions": [
                description or ""
                for description
                in src.descriptions
            ],
            "is_georeferenced": bool(
                src.crs
            ),
        }


def detect_sar_water(
    path: Path,
    data: Dict[str, Any],
) -> Tuple[
    np.ndarray,
    str,
    float,
]:
    """
    SAR-specific water baseline.

    Uses the lower backscatter tail with an adaptive threshold,
    then removes speckle-sized components.

    The threshold is intentionally tighter than the previous 20th
    percentile baseline so ordinary dark urban/shadow pixels are
    less likely to become water.
    """

    sar = read_sar_backscatter(
        path
    )

    db = sar["db"]

    finite = np.isfinite(
        db
    )

    values = db[finite]

    if values.size == 0:
        raise ValueError(
            "No finite SAR backscatter pixels."
        )

    # Sentinel-1 water is generally in the lower backscatter tail.
    # A tighter adaptive threshold reduces false positives.
    threshold = float(
        np.percentile(
            values,
            12,
        )
    )

    mask = (
        np.isfinite(db)
        & (db <= threshold)
    ).astype(
        np.uint8
    ) * 255

    # Remove isolated speckle-like pixels and bridge small gaps.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (7, 7),
            np.uint8,
        ),
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (mask > 0).astype(
                np.uint8
            ),
            8,
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    min_component = max(
        40,
        int(
            mask.size * 0.00008
        ),
    )

    for label_id in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= min_component:
            cleaned[
                labels == label_id
            ] = 255

    return (
        cleaned,
        "SAR adaptive low-backscatter water baseline",
        0.78,
    )

def detect_sar_builtup(
    path: Path,
) -> Tuple[
    np.ndarray,
    str,
    float,
]:
    """
    Simple SAR high-backscatter/texture baseline for
    built-up areas.

    Urban/constructed surfaces can produce relatively high
    radar backscatter, but this is only a heuristic.
    """

    sar = read_sar_backscatter(
        path
    )

    db = sar["db"]

    finite = np.isfinite(
        db
    )

    values = db[finite]

    if values.size == 0:
        raise ValueError(
            "No finite SAR backscatter pixels."
        )

    threshold = float(
        np.percentile(
            values,
            80,
        )
    )

    mask = (
        np.isfinite(db)
        & (db >= threshold)
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return (
        mask,
        "SAR high-backscatter built-up baseline",
        0.60,
    )


def detect_modality_feature(
    path: Path,
    data: Dict[str, Any],
    modality: str,
    feature: str,
) -> Tuple[
    np.ndarray,
    str,
    float,
]:
    """
    Dispatch a feature detector according to modality.
    """

    if modality == "sar":

        if feature == "water":
            return detect_sar_water(
                path,
                data,
            )

        if feature == "built-up":
            return detect_sar_builtup(
                path,
            )

        # There is no defensible generic SAR vegetation
        # detector in this MVP.
        raise ValueError(
            f"No SAR-specific baseline implemented for "
            f"feature '{feature}'."
        )

    return feature_mask(
        feature,
        path,
        data,
    )


def _align_mask_to_primary(
    mask: np.ndarray,
    source_data: Dict[str, Any],
    target_data: Dict[str, Any],
) -> Tuple[
    np.ndarray,
    str,
]:
    """
    Align a mask to the primary image grid.

    When both rasters are properly georeferenced, use rasterio
    reprojection. Otherwise, use a transparent pixel-grid
    resize and report that limitation.
    """

    target_shape = (
        target_data["height"],
        target_data["width"],
    )

    same_grid = (
        mask.shape == target_shape
    )

    same_crs = (
        source_data.get("crs")
        and target_data.get("crs")
        and source_data.get("crs")
        == target_data.get("crs")
    )

    source_transform = (
        source_data.get(
            "transform"
        )
    )

    target_transform = (
        target_data.get(
            "transform"
        )
    )

    if (
        same_grid
        and same_crs
        and source_transform
        and target_transform
    ):
        return (
            mask,
            "native aligned grid",
        )

    if (
        source_data.get(
            "is_georeferenced"
        )
        and target_data.get(
            "is_georeferenced"
        )
        and source_data.get(
            "crs"
        )
        and target_data.get(
            "crs"
        )
        and source_transform
        and target_transform
    ):

        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        destination = np.zeros(
            target_shape,
            dtype=np.uint8,
        )

        reproject(
            source=mask.astype(
                np.uint8
            ),
            destination=destination,
            src_transform=Affine(
                *source_transform[:6]
            ),
            src_crs=source_data[
                "crs"
            ],
            dst_transform=Affine(
                *target_transform[:6]
            ),
            dst_crs=target_data[
                "crs"
            ],
            resampling=Resampling.nearest,
        )

        return (
            destination,
            "geospatial reprojection",
        )

    resized = cv2.resize(
        mask,
        (
            target_data["width"],
            target_data["height"],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return (
        resized,
        "pixel-grid resize (no shared georeferencing)",
    )


# ============================================================
# OPTICAL + SAR
# ============================================================

def analyze_cross_modal(
    path1: Path,
    data1: Dict[str, Any],
    path2: Path,
    data2: Dict[str, Any],
    feature: str = "water",
) -> Dict[str, Any]:
    """
    SAR-aware optical + SAR fusion baseline.

    The optical and SAR masks are generated independently, SAR is
    geospatially reprojected to the optical grid when both inputs
    have valid georeferencing, and a small registration tolerance
    is used when forming the consensus.

    This is a deterministic multimodal fusion baseline, not a
    learned multimodal neural network.
    """

    modality1 = infer_modality(
        path1,
        data1,
        data1.get("original_filename"),
    )

    modality2 = infer_modality(
        path2,
        data2,
        data2.get("original_filename"),
    )

    if {
        modality1,
        modality2,
    } != {
        "optical",
        "sar",
    }:

        raise ValueError(
            "Optical-SAR fusion requires exactly "
            "one optical input and one SAR input. "
            f"Detected: primary={modality1}, "
            f"secondary={modality2}."
        )

    if modality1 == "sar":

        sar_path = path1
        sar_data = data1

        optical_path = path2
        optical_data = data2

    else:

        optical_path = path1
        optical_data = data1

        sar_path = path2
        sar_data = data2

    # --------------------------------------------------------
    # Independent modality detectors
    # --------------------------------------------------------

    optical_mask, optical_method, optical_conf = (
        detect_modality_feature(
            optical_path,
            optical_data,
            "optical",
            feature,
        )
    )

    sar_mask, sar_method, sar_conf = (
        detect_modality_feature(
            sar_path,
            sar_data,
            "sar",
            feature,
        )
    )

    # --------------------------------------------------------
    # Geospatial alignment
    # --------------------------------------------------------

    sar_mask_aligned, alignment_method = (
        _align_mask_to_primary(
            sar_mask,
            sar_data,
            optical_data,
        )
    )

    optical_binary = (
        optical_mask > 0
    ).astype(
        np.uint8
    ) * 255

    sar_binary = (
        sar_mask_aligned > 0
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Registration-tolerant consensus
    # --------------------------------------------------------
    #
    # Exact pixel AND is too strict for independently produced
    # masks, especially when optical and SAR resolutions differ.
    # A 5x5 dilation allows a few pixels of registration tolerance
    # without simply taking the whole union.
    # --------------------------------------------------------

    tolerance_kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    optical_tolerant = cv2.dilate(
        optical_binary,
        tolerance_kernel,
        iterations=1,
    )

    sar_tolerant = cv2.dilate(
        sar_binary,
        tolerance_kernel,
        iterations=1,
    )

    tolerant_consensus = cv2.bitwise_or(
        cv2.bitwise_and(
            optical_tolerant,
            sar_binary,
        ),
        cv2.bitwise_and(
            optical_binary,
            sar_tolerant,
        ),
    )

    # Keep consensus compact and remove one-pixel fragments.
    tolerant_consensus = cv2.morphologyEx(
        tolerant_consensus,
        cv2.MORPH_CLOSE,
        np.ones(
            (5, 5),
            np.uint8,
        ),
    )

    tolerant_consensus = cv2.morphologyEx(
        tolerant_consensus,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8,
        ),
    )

    # If tolerant consensus is empty, fall back to exact consensus.
    exact_consensus = cv2.bitwise_and(
        optical_binary,
        sar_binary,
    )

    consensus = (
        tolerant_consensus
        if (tolerant_consensus > 0).any()
        else exact_consensus
    )

    union = cv2.bitwise_or(
        optical_binary,
        sar_binary,
    )

    optical_pixels = int(
        (optical_binary > 0).sum()
    )

    sar_pixels = int(
        (sar_binary > 0).sum()
    )

    union_pixels = int(
        (union > 0).sum()
    )

    exact_consensus_pixels = int(
        (exact_consensus > 0).sum()
    )

    consensus_pixels = int(
        (consensus > 0).sum()
    )

    # Exact IoU-style agreement is retained for auditability.
    exact_agreement = (
        0.0
        if union_pixels == 0
        else (
            100.0
            * exact_consensus_pixels
            / union_pixels
        )
    )

    # The displayed agreement uses the registration-tolerant
    # consensus because the two sensors rarely land on identical
    # pixels after reprojection.
    agreement = (
        0.0
        if union_pixels == 0
        else (
            100.0
            * consensus_pixels
            / union_pixels
        )
    )

    optical_coverage = (
        100.0
        * optical_pixels
        / optical_binary.size
    )

    sar_coverage = (
        100.0
        * sar_pixels
        / sar_binary.size
    )

    fused_coverage = (
        100.0
        * consensus_pixels
        / consensus.size
    )

    # --------------------------------------------------------
    # Remove tiny fused regions from the final overlay.
    # --------------------------------------------------------

    fused_component = _largest_component(
        consensus
    )

    fused_component_pixels = int(
        (fused_component > 0).sum()
    )

    if fused_component_pixels >= 40:
        final_mask = fused_component
    else:
        final_mask = consensus

    final_pixels = int(
        (final_mask > 0).sum()
    )

    final_coverage = (
        100.0
        * final_pixels
        / final_mask.size
    )

    # --------------------------------------------------------
    # Spatial evidence on the final fused region.
    # --------------------------------------------------------

    final_evidence = spatial_evidence(
        final_mask,
        optical_data,
    )

    bounding_box = None

    ys, xs = np.where(
        final_mask > 0
    )

    if len(xs) > 0 and len(ys) > 0:

        bounding_box = {
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
        }

    # --------------------------------------------------------
    # Overlay on optical imagery.
    # --------------------------------------------------------

    overlay = make_overlay(
        optical_data["rgb"],
        final_mask,
        "fusion",
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = round(
        min(
            0.93,
            max(
                0.35,
                (
                    optical_conf
                    + sar_conf
                )
                / 2.0
                + min(
                    agreement / 500.0,
                    0.10,
                )
                + (
                    0.04
                    if alignment_method
                    == "geospatial reprojection"
                    else 0.0
                ),
            ),
        ),
        2,
    )

    if agreement >= 60:
        agreement_phrase = "strong agreement"
    elif agreement >= 30:
        agreement_phrase = "moderate agreement"
    elif agreement >= 10:
        agreement_phrase = "limited agreement"
    else:
        agreement_phrase = "low agreement"

    if final_pixels > 0:

        answer = (
            f"The optical and SAR inputs show "
            f"{agreement_phrase} for the requested "
            f"{feature} feature ({agreement:.1f}% "
            "over their candidate union). "
            f"The fused water region covers "
            f"about {final_coverage:.1f}% of the optical image. "
            f"Optical candidates cover "
            f"{optical_coverage:.1f}% and SAR candidates cover "
            f"{sar_coverage:.1f}%."
        )

    else:

        answer = (
            f"No spatially consistent {feature} region "
            "was found between the optical and SAR candidates. "
            f"Optical coverage is {optical_coverage:.1f}% "
            f"and SAR coverage is {sar_coverage:.1f}%."
        )

    return {
        "answer": answer,
        "confidence": confidence,
        "tool": (
            "Optical-SAR Fusion "
            "Baseline (registration-tolerant)"
        ),
        "overlay": overlay,
        "bounding_box": bounding_box,
        "location": final_evidence.get(
            "location"
        ),
        "evidence": {
            "feature": feature,
            "primary_modality": modality1,
            "secondary_modality": modality2,
            "optical_method": optical_method,
            "sar_method": sar_method,
            "alignment_method": alignment_method,
            "candidate_water_agreement_percent": round(
                agreement,
                2,
            ),
            "exact_pixel_agreement_percent": round(
                exact_agreement,
                2,
            ),
            "optical_candidate_coverage_percent": round(
                optical_coverage,
                2,
            ),
            "sar_candidate_coverage_percent": round(
                sar_coverage,
                2,
            ),
            "fused_water_coverage_percent": round(
                final_coverage,
                2,
            ),
            "registration_tolerance_pixels": 2,
            "fused_spatial_evidence": final_evidence,
        },
        "mask_stats": {
            "optical": mask_stats(
                optical_binary
            ),
            "sar": mask_stats(
                sar_binary
            ),
            "fused": mask_stats(
                final_mask
            ),
        },
    }


# ============================================================
# PAIR VALIDATION
# ============================================================

def validate_pair_compatibility(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate dimensions, CRS, georeferencing, band counts,
    and conservative modality hints.
    """

    issues: List[str] = []
    warnings: List[str] = []

    p = primary["data"]
    s = secondary["data"]

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    if (
        p["width"] != s["width"]
        or p["height"] != s["height"]
    ):

        warnings.append(
            "Image dimensions differ; "
            "the workflow may resize or reproject "
            "the secondary image."
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if (
        p.get("crs")
        and s.get("crs")
    ):

        if p["crs"] != s["crs"]:

            warnings.append(
                "CRS differs between inputs; "
                "geospatial reprojection will be required "
                "for spatial fusion."
            )

    elif (
        p.get("crs")
        or s.get("crs")
    ):

        warnings.append(
            "Only one image contains CRS metadata; "
            "true geospatial co-registration cannot "
            "be guaranteed."
        )

    else:

        warnings.append(
            "Neither image contains CRS metadata; "
            "pairwise spatial alignment is limited "
            "to pixel-grid resizing."
        )

    # --------------------------------------------------------
    # Georeferencing
    # --------------------------------------------------------

    if (
        bool(
            p.get(
                "is_georeferenced"
            )
        )
        != bool(
            s.get(
                "is_georeferenced"
            )
        )
    ):

        warnings.append(
            "Georeferencing status differs."
        )

    # --------------------------------------------------------
    # Band count
    # --------------------------------------------------------

    if (
        p.get("count")
        != s.get("count")
    ):

        warnings.append(
            "Band counts differ."
        )

    # --------------------------------------------------------
    # Modality
    # --------------------------------------------------------

    p_modality = (
        infer_modality(
            primary["path"],
            p,
            primary.get("filename"),
        )
        if isinstance(
            primary.get("path"),
            Path,
        )
        else "unknown"
    )

    s_modality = (
        infer_modality(
            secondary["path"],
            s,
            secondary.get("filename"),
        )
        if isinstance(
            secondary.get("path"),
            Path,
        )
        else "unknown"
    )

    # The in-memory FILES entries do contain Path objects,
    # but keep this safe for direct function use.
    if p_modality == "unknown":
        p_modality = "optical"

    if s_modality == "unknown":
        s_modality = "optical"

    if p_modality == "sar" or s_modality == "sar":

        if (
            p_modality == "sar"
            and s_modality == "sar"
        ):

            warnings.append(
                "Both inputs are detected as SAR; "
                "this is not an optical-SAR pair."
            )

        elif (
            p_modality == "optical"
            and s_modality == "optical"
        ):

            warnings.append(
                "No SAR input detected; "
                "this is an optical-optical pair."
            )

        else:

            warnings.append(
                "Optical-SAR pair detected. "
                "SAR-specific preprocessing will be used."
            )

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "modality": {
            "primary": p_modality,
            "secondary": s_modality,
        },
        "optical_sar_pair": {
            "is_optical_sar": (
                {
                    p_modality,
                    s_modality,
                }
                == {
                    "optical",
                    "sar",
                }
            ),
        },
    }


# ============================================================
# API
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():

    index_path = (
        FRONTEND_DIR
        / "index.html"
    )

    if not index_path.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Frontend index.html not found."
            ),
        )

    return HTMLResponse(
        index_path.read_text(
            encoding="utf-8"
        )
    )


@app.get(
    "/generated/{name}"
)
def generated(
    name: str,
):

    path = (
        GENERATED_DIR
        / Path(name).name
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Generated file not found."
            ),
        )

    return FileResponse(
        path
    )


@app.get(
    "/api/health"
)
def health():

    return {
        "ok": True,
        "name": "SatQuery AI MVP",
        "version": "0.7.0",
        "agent_planner": "enabled",
        "rs_vlm_available": RS_VLM.available,
        "rs_vlm_model": (
            "GeoRSCLIP + RSVQA Adapter"
            if RS_VLM.available
            else None
        ),
    }


@app.get(
    "/api/uploads"
)
def list_uploads():

    return [
        {
            "id": file_id,
            "filename": info[
                "filename"
            ],
        }
        for file_id, info
        in FILES.items()
    ]


@app.post(
    "/api/upload"
)
async def upload(
    file: UploadFile = File(...),
):

    suffix = (
        Path(
            file.filename or ""
        )
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXT:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported format. "
                "Use: "
                + ", ".join(
                    sorted(
                        ALLOWED_EXT
                    )
                )
            ),
        )

    file_id = uuid.uuid4().hex

    target = (
        UPLOAD_DIR
        / f"{file_id}{suffix}"
    )

    content = await file.read()

    target.write_bytes(
        content
    )

    try:

        data = read_image(
            target
        )

    except Exception as exc:

        target.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read image: "
                f"{exc}"
            ),
        )

    preview = save_preview(
        data["rgb"],
        file_id,
    )

    data["original_filename"] = (
        file.filename or target.name
    )

    FILES[file_id] = {
        "path": target,
        "data": data,
        "filename": file.filename,
    }

    modality = infer_modality(
        target,
        data,
        file.filename,
    )

    metadata = {
        key: value
        for key, value
        in data.items()
        if key != "rgb"
    }

    metadata["modality"] = modality

    return {
        "id": file_id,
        "filename": file.filename,
        "metadata": metadata,
        "preview_url": (
            f"/generated/{preview}"
        ),
    }


@app.post(
    "/api/analyze"
)
def analyze(
    req: AnalyzeRequest,
):

    # --------------------------------------------------------
    # Primary input
    # --------------------------------------------------------

    if req.primary_id not in FILES:

        raise HTTPException(
            status_code=404,
            detail=(
                "Primary image not found; "
                "upload again."
            ),
        )

    primary = FILES[
        req.primary_id
    ]

    # --------------------------------------------------------
    # Secondary input
    # --------------------------------------------------------

    secondary = (
        FILES.get(
            req.secondary_id
        )
        if req.secondary_id
        else None
    )

    if (
        req.secondary_id
        and secondary is None
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Secondary image not found; "
                "upload again."
            ),
        )

    # --------------------------------------------------------
    # Pair validation
    # --------------------------------------------------------

    pair_validation = None

    if secondary is not None:

        pair_validation = (
            validate_pair_compatibility(
                primary,
                secondary,
            )
        )

    # --------------------------------------------------------
    # Conversation state
    # --------------------------------------------------------

    conversation_id = (
        req.conversation_id
        or "default"
    )

    context = CONTEXT.setdefault(
        conversation_id,
        {},
    )

    # --------------------------------------------------------
    # Agent planner
    # --------------------------------------------------------

    image_count = (
        2
        if secondary is not None
        else 1
    )

    try:

        plan = build_plan(
            query=req.query,
            image_count=image_count,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    task = plan["task"]

    planned_feature = plan.get(
        "feature"
    )

    if planned_feature not in {
        None,
        "auto",
    }:

        feature = planned_feature

    else:

        feature = infer_feature(
            req.query,
            context,
        )

    context[
        "last_feature"
    ] = feature

    context[
        "last_task"
    ] = task

    # --------------------------------------------------------
    # Execution trace
    # --------------------------------------------------------

    trace: List[
        Dict[str, Any]
    ] = [
        {
            "step": "Input Validator",
            "status": "ok",
            "detail": (
                f"Primary: "
                f"{primary['filename']}"
            ),
        },
        {
            "step": "Query Interpreter",
            "status": "ok",
            "detail": (
                f"Task={task}; "
                f"feature={feature}"
            ),
        },
        {
            "step": "Agent Planner",
            "status": "ok",
            "detail": (
                f"Planned "
                f"{len(plan.get('steps', []))} "
                "execution steps"
            ),
        },
    ]

    if pair_validation is not None:

        trace.append(
            {
                "step": "Pair Compatibility",
                "status": (
                    "ok"
                    if pair_validation[
                        "compatible"
                    ]
                    else "failed"
                ),
                "detail": {
                    "modality": pair_validation[
                        "modality"
                    ],
                    "warnings": pair_validation[
                        "warnings"
                    ],
                    "issues": pair_validation[
                        "issues"
                    ],
                },
            }
        )

    # --------------------------------------------------------
    # Execute specialist
    # --------------------------------------------------------

    if (
        task == "change_analysis"
        and secondary is not None
    ):

        trace.append(
            {
                "step": "Change Engine",
                "status": "ok",
                "detail": (
                    "Bi-temporal change "
                    "specialist selected."
                ),
            }
        )

        result = analyze_change(
            primary["path"],
            primary["data"],
            secondary["path"],
            secondary["data"],
            feature,
        )

    elif (
        task == "cross_modal"
        and secondary is not None
    ):

        trace.append(
            {
                "step": "Optical-SAR Fusion",
                "status": "ok",
                "detail": (
                    "SAR-aware optical-SAR "
                    "fusion specialist selected."
                ),
            }
        )

        try:
            result = analyze_cross_modal(
                primary["path"],
                primary["data"],
                secondary["path"],
                secondary["data"],
                feature=(
                    feature
                    if feature
                    not in {
                        "auto",
                        "scene",
                        "multimodal",
                        None,
                    }
                    else "water"
                ),
            )
        except ValueError as exc:
            trace.append(
                {
                    "step": "Optical-SAR Fusion",
                    "status": "error",
                    "detail": str(exc),
                }
            )

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

    elif task == "grounding":

        trace.append(
            {
                "step": "RS Grounding",
                "status": "ok",
                "detail": (
                    "Hybrid remote-sensing "
                    "grounding specialist selected."
                ),
            }
        )

        try:

            result = analyze_grounding(
                primary["path"],
                primary["data"],
                req.query,
            )

        except Exception as exc:

            trace.append(
                {
                    "step": "RS Grounding",
                    "status": "error",
                    "detail": str(exc),
                }
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Grounding failed: "
                    f"{exc}"
                ),
            )

    elif task == "captioning":

        trace.append(
            {
                "step": "RS Captioner",
                "status": "ok",
                "detail": (
                    "Scene description "
                    "specialist selected."
                ),
            }
        )

        result = analyze_single(
            primary["path"],
            primary["data"],
            feature,
            "captioning",
            req.query,
        )

    elif task in {
        "vqa",
        "multi_image_vqa",
    }:

        trace.append(
            {
                "step": "RS-VQA",
                "status": "ok",
                "detail": (
                    "GeoRSCLIP + RSVQA Adapter "
                    "selected."
                ),
            }
        )

        result = analyze_single(
            primary["path"],
            primary["data"],
            feature,
            "vqa",
            req.query,
        )

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported planned task: "
                f"{task}"
            ),
        )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    trace.append(
        {
            "step": "Evidence Builder",
            "status": "ok",
            "detail": (
                "Spatial/visual evidence "
                "generated when available."
            ),
        }
    )

    # --------------------------------------------------------
    # Result integrator
    # --------------------------------------------------------

    trace.append(
        {
            "step": "Response Integrator",
            "status": "ok",
            "detail": (
                "Answer + confidence + "
                "audit trace returned."
            ),
        }
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "query": req.query,

        "task": task,

        "feature": feature,

        "answer": result.get(
            "answer",
            "",
        ),

        "confidence": result.get(
            "confidence",
            0.0,
        ),

        "tool": result.get(
            "tool",
            result.get(
                "model",
                "unknown",
            ),
        ),

        "execution_plan": plan,

        "execution_trace": trace,

        "input_validation": {
            "pair": pair_validation,
        },

        "modalities": (
            pair_validation.get(
                "modality"
            )
            if pair_validation
            else {
                "primary": infer_modality(
                    primary["path"],
                    primary["data"],
                    primary.get("filename"),
                )
            }
        ),

        "evidence": result.get(
            "evidence"
        ),

        "overlay_url": (
            f"/generated/"
            f"{result['overlay']}"
            if result.get(
                "overlay"
            )
            else None
        ),

        "bounding_box": result.get(
            "bounding_box"
        ),

        "grounding_location": result.get(
            "location"
        ),

        "grounding_method": result.get(
            "method"
        ),

        "mask_stats": result.get(
            "mask_stats"
        ),

        "top_answers": result.get(
            "top_answers",
            [],
        ),

        "prototype_notice": (
            "SatQuery MVP combines "
            "remote-sensing model inference, "
            "GIS processing, specialist "
            "workflow orchestration, and "
            "observable execution traces. "
            "The current bi-temporal "
            "and optical-SAR specialists "
            "are deterministic MVP baselines; "
            "the optical-SAR path uses "
            "SAR-specific backscatter processing "
            "but is not a learned multimodal model."
        ),
    }


# ============================================================
# DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )