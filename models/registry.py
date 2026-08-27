from __future__ import annotations

from typing import Any, Dict


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rs_vqa": {
        "name": "Remote-Sensing VQA",
        "type": "vision_language",
        "description": "GeoRSCLIP ViT-B/32 paired with fine-tuned RSVQA adapter for answering remote-sensing questions.",
        "modalities": ["optical", "multispectral", "sar"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["single_image_vqa", "top5_confidence"],
        "status": "active",
    },
    "rs_captioner": {
        "name": "Remote-Sensing Captioner",
        "type": "vision_language",
        "description": "Remote-sensing scene describer and land-cover summarizer.",
        "modalities": ["optical", "multispectral"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["scene_description", "land_cover_summary"],
        "status": "active",
    },
    "rs_grounding": {
        "name": "Hybrid Remote-Sensing Grounding",
        "type": "vision_language_spatial",
        "description": "Text-guided spatial region grounding combining patch similarity and physics-based spectral masks.",
        "modalities": ["optical", "multispectral", "sar"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["text_guided_localization", "bounding_box", "wgs84_centroid", "area_hectares", "geojson"],
        "status": "active",
    },
    "change_engine": {
        "name": "Bi-Temporal Change Engine",
        "type": "multitemporal_specialist",
        "description": "Bi-temporal change detection, quantitative metric shift, direction estimation, and dual-color change mapping.",
        "modalities": ["optical", "multispectral", "sar"],
        "input_formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "capabilities": ["bitemporal_change", "change_vqa", "delta_hectares", "delta_percentage_points", "change_map"],
        "status": "active",
    },
    "optical_sar_fusion": {
        "name": "Optical-SAR Cross-Modal Fusion",
        "type": "multimodal_specialist",
        "description": "Co-registered optical/multispectral + SAR consensus reasoning and cross-modal agreement scoring.",
        "modalities": ["optical", "sar"],
        "input_formats": ["GeoTIFF", "TIFF"],
        "capabilities": ["geospatial_reprojection", "registration_tolerant_fusion", "agreement_score", "fused_overlay"],
        "status": "active",
    },
    "geospatial_tools": {
        "name": "Deterministic GIS & Spectral Tools",
        "type": "deterministic_geospatial",
        "description": "Physics-based remote-sensing spectral indices (NDWI, NDVI, NDBI), SAR backscatter calibration, and equal-area projection geometry.",
        "modalities": ["multispectral", "optical", "sar"],
        "input_formats": ["GeoTIFF", "TIFF"],
        "capabilities": ["ndwi", "ndvi", "ndbi", "sar_db", "crs_transform", "wgs84_reprojection", "equal_area_hectares"],
        "status": "active",
    },
}


def get_tool(name: str) -> Dict[str, Any]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model/tool: {name}")
    return MODEL_REGISTRY[name]


def list_tools() -> Dict[str, Dict[str, Any]]:
    return MODEL_REGISTRY.copy()