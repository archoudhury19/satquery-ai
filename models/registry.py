from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "input_validator": {
        "name": "Input Compatibility & Modality Validator",
        "type": "pre_execution_guard",
        "description": "Validates raster integrity, CRS coordinates, sensor band formats, and temporal/multimodal compatibility.",
        "modalities": ["optical", "multispectral", "sar", "panchromatic"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["format_check", "crs_verification", "spatial_overlap", "modality_inference"],
        "parameters_schema": {
            "strict": {"type": "boolean", "default": True},
        },
        "returns_schema": {
            "valid": "boolean",
            "primary": "string",
        },
        "status": "active",
    },
    "rs_vqa": {
        "name": "Remote-Sensing VQA",
        "type": "vision_language",
        "description": "GeoRSCLIP ViT-B/32 paired with fine-tuned RSVQA adapter for answering remote-sensing questions.",
        "modalities": ["optical", "multispectral", "sar"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["single_image_vqa", "top5_confidence", "entropy_calibration"],
        "parameters_schema": {
            "query": {"type": "string", "required": True, "description": "Natural language query"},
            "top_k": {"type": "integer", "default": 5, "description": "Top-k predictions"},
        },
        "returns_schema": {
            "answer": "string",
            "confidence": "float",
            "top_answers": "list[dict]",
        },
        "status": "active",
    },
    "rs_captioner": {
        "name": "Remote-Sensing Captioner",
        "type": "vision_language",
        "description": "Remote-sensing scene describer and spatial cardinal layout synthesizer.",
        "modalities": ["optical", "multispectral"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["scene_description", "land_cover_summary", "cardinal_sector_breakdown"],
        "parameters_schema": {
            "include_sectors": {"type": "boolean", "default": True},
        },
        "returns_schema": {
            "answer": "string",
            "confidence": "float",
            "sectors": "dict",
        },
        "status": "active",
    },
    "rs_grounding": {
        "name": "Hybrid Remote-Sensing Grounding",
        "type": "vision_language_spatial",
        "description": "Dense feature activation map grounding combining text similarity and spectral masks.",
        "modalities": ["optical", "multispectral", "sar"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["text_guided_localization", "bounding_box", "wgs84_centroid", "area_hectares", "dense_heatmaps"],
        "parameters_schema": {
            "query": {"type": "string", "required": True},
            "method": {"type": "string", "default": "dense_feature_activation"},
            "threshold": {"type": "float", "default": 0.4},
        },
        "returns_schema": {
            "bounding_box": "list[int]",
            "location": "string",
            "confidence": "float",
            "overlay": "string",
        },
        "status": "active",
    },
    "change_engine": {
        "name": "Bi-Temporal Change Engine",
        "type": "multitemporal_specialist",
        "description": "Bi-temporal change reasoning, directional classification, spatial clusters, and CDVQA answer synthesis.",
        "modalities": ["optical", "multispectral", "sar"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["bitemporal_change", "cdvqa_reasoning", "delta_hectares", "delta_percentage_points", "change_clusters"],
        "parameters_schema": {
            "feature": {"type": "string", "default": "auto"},
            "compare_before_after": {"type": "boolean", "default": True},
        },
        "returns_schema": {
            "answer": "string",
            "direction": "string",
            "delta_percentage_points": "float",
            "delta_hectares": "float",
            "overlay": "string",
        },
        "status": "active",
    },
    "optical_sar_fusion": {
        "name": "Optical-SAR Cross-Modal Fusion",
        "type": "multimodal_specialist",
        "description": "Co-registered optical/multispectral + SAR consensus reasoning with adaptive Lee speckle filtering and agreement scoring.",
        "modalities": ["optical", "sar"],
        "input_formats": ["GeoTIFF", "TIFF"],
        "capabilities": ["geospatial_reprojection", "registration_tolerant_fusion", "agreement_score", "speckle_filtering", "fused_overlay"],
        "parameters_schema": {
            "feature": {"type": "string", "default": "water"},
            "speckle_filter": {"type": "string", "default": "lee"},
        },
        "returns_schema": {
            "answer": "string",
            "confidence": "float",
            "agreement_pct": "float",
            "overlay": "string",
        },
        "status": "active",
    },
    "geospatial_tools": {
        "name": "Deterministic GIS & Spectral Tools",
        "type": "deterministic_geospatial",
        "description": "Physics-based remote-sensing spectral indices (NDWI, NDVI, NDBI), SAR backscatter calibration, and equal-area projection geometry.",
        "modalities": ["multispectral", "optical", "sar"],
        "input_formats": ["GeoTIFF", "TIFF"],
        "capabilities": ["ndwi", "ndvi", "ndbi", "sar_db", "crs_transform", "wgs84_reprojection", "equal_area_hectares"],
        "parameters_schema": {
            "generate_spatial_evidence": {"type": "boolean", "default": True},
        },
        "returns_schema": {
            "wgs84_centroid": "dict",
            "area_hectares": "float",
            "geojson": "dict",
        },
        "status": "active",
    },
    "land_cover_segmenter": {
        "name": "Multi-Class Land-Cover Segmenter",
        "type": "multiclass_segmentation",
        "description": "Multi-class false-colour land-cover segmentation across water, vegetation, built-up, and bare land.",
        "modalities": ["optical", "multispectral"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["multiclass_map", "coverage_statistics", "spectral_thresholding"],
        "parameters_schema": {
            "classes": {"type": "list[string]", "default": ["water", "vegetation", "built_up"]},
        },
        "returns_schema": {
            "answer": "string",
            "mask_stats": "dict",
            "overlay": "string",
        },
        "status": "active",
    },
}

# Dynamic Dispatch Handler Registry
_HANDLERS: Dict[str, Callable[..., Any]] = {}


def register_handler(tool_name: str, handler: Callable[..., Any]) -> None:
    """Register an execution callable handler for a tool."""
    _HANDLERS[tool_name] = handler


def has_handler(tool_name: str) -> bool:
    """Check if a callable handler is registered for the tool."""
    return tool_name in _HANDLERS


def get_tool(name: str) -> Dict[str, Any]:
    """Retrieve metadata and schema for a registered tool."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model/tool: '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_tools() -> Dict[str, Dict[str, Any]]:
    """Return a dictionary of all registered tools with their schemas."""
    return MODEL_REGISTRY.copy()


def execute_tool(
    tool_name: str,
    context: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Dynamically dispatch execution to the registered tool handler.
    """
    if tool_name not in _HANDLERS:
        raise KeyError(f"No execution handler registered for tool '{tool_name}'.")
    handler = _HANDLERS[tool_name]
    params = parameters or {}
    return handler(context, **params)