from __future__ import annotations

from typing import Any, Dict


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vqa": {
        "name": "rs_vqa",
        "type": "vision_language",
        "description": "Remote-sensing visual question answering specialist",
    },
    "captioning": {
        "name": "rs_captioner",
        "type": "vision_language",
        "description": "Remote-sensing scene description specialist",
    },
    "change": {
        "name": "change_engine",
        "type": "specialist",
        "description": "Bi-temporal change analysis specialist",
    },
    "optical_sar": {
        "name": "optical_sar_fusion",
        "type": "multimodal",
        "description": "Optical-SAR joint information extraction specialist",
    },
    "gis": {
        "name": "geospatial_tools",
        "type": "deterministic",
        "description": "Raster, spatial evidence and measurement tools",
    },
}


def get_tool(task: str) -> Dict[str, Any]:
    """Return the registered tool for a task."""
    if task not in MODEL_REGISTRY:
        raise KeyError(f"Unknown task: {task}")

    return MODEL_REGISTRY[task]


def list_tools() -> Dict[str, Dict[str, Any]]:
    """Return all registered tools."""
    return MODEL_REGISTRY.copy()