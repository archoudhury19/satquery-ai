from __future__ import annotations

from typing import Any, Dict, List


import re


def understand_query(
    query: str,
    image_count: int,
) -> Dict[str, Any]:

    q = query.lower().strip()

    # 1. Captioning / Scene Description intent
    captioning = bool(re.search(
        r"\b(describe|description|caption|scene\s+summary|summarize|overview|what\s+is\s+visible|what\s+does\s+this\s+(?:image|scene)\s+(?:show|depict)|tell\s+me\s+about)\b",
        q
    ))

    # 2. Bi-Temporal Change intent
    temporal = bool(re.search(
        r"\b(change|changed|changes|difference|delta|between\s+these\s+two|before\s+and\s+after|increase|increased|decrease|decreased|loss|gain|expansion|expanded|shrink|shrunk|deforestation|wildfire|burn|scar|flood|inundation|remained\s+unchanged|growth|grown)\b",
        q
    ))

    # 3. Cross-Modal Optical-SAR intent (when 2 images are present)
    cross_modal = (image_count >= 2) and bool(re.search(
        r"\b(sar|radar|optical|optical-sar|cross-modal|multimodal|multi-sensor|both\s+images|together|joint|complementary|consensus)\b",
        q
    ))

    # 4. Multi-Class Land-Cover Segmentation
    segmentation = (not captioning) and bool(re.search(
        r"\b(segment|segmentation|classify\s+all|colour-code|color-code|land-cover\s+map|all\s+classes|different\s+colou?rs|map\s+land\s+cover)\b",
        q
    ))

    # 5. Dense Visual Grounding intent (spatial localization of specific objects/features)
    grounding = (not captioning) and (not segmentation) and bool(re.search(
        r"\b(highlight|locate|pinpoint|where\s+is|show\s+me\s+where|point\s+out|box|delineate|find|isolate|water\s+body|river|lake|reservoir|channel)\b",
        q
    ))

    # 6. Spatial context flag
    spatial = grounding or bool(re.search(r"\b(where|location|coordinates|region|area|centroid|sector|north|south|east|west)\b", q))

    return {
        "spatial": spatial,
        "temporal": temporal,
        "cross_modal": cross_modal,
        "captioning": captioning,
        "grounding": grounding,
        "segmentation": segmentation,
        "vqa": not captioning and not grounding and not segmentation,
    }


def add_step(
    steps: List[Dict[str, Any]],
    tool: str,
    parameters: Dict[str, Any] | None = None,
) -> None:

    steps.append(
        {
            "tool": tool,
            "parameters": parameters or {},
        }
    )


def build_plan(
    query: str,
    image_count: int,
) -> Dict[str, Any]:

    if image_count < 1:
        raise ValueError(
            "At least one image is required."
        )

    if image_count > 2:
        raise ValueError(
            "MVP currently supports a maximum of two images."
        )

    intent = understand_query(
        query,
        image_count,
    )

    steps: List[Dict[str, Any]] = []

    add_step(
        steps,
        "input_validator",
    )

    # --------------------------------------------------------
    # SINGLE IMAGE
    # --------------------------------------------------------

    if image_count == 1:

        if intent["temporal"]:
            raise ValueError(
                "Bi-temporal change analysis requires two images (Time-1 and Time-2). "
                "Please upload a second image in the 'Second image' slot to compare changes."
            )

        if intent["captioning"] and not intent["segmentation"]:

            task = "captioning"
            feature = "scene"

            add_step(
                steps,
                "rs_captioner",
            )

        elif intent["segmentation"]:

            task = "segmentation"
            feature = "multiclass"

            add_step(
                steps,
                "land_cover_segmenter",
                {
                    "classes": ["water", "vegetation", "built_up"],
                },
            )

        elif intent["grounding"]:

            task = "grounding"
            feature = "auto"

            add_step(
                steps,
                "rs_grounding",
                {
                    "method": "grid_similarity",
                    "grid_size": 4,
                },
            )

            add_step(
                steps,
                "geospatial_tools",
                {
                    "generate_spatial_evidence": True,
                },
            )

        else:

            task = "vqa"
            feature = "auto"

            add_step(
                steps,
                "rs_vqa",
            )

            if intent["spatial"]:

                add_step(
                    steps,
                    "geospatial_tools",
                    {
                        "generate_spatial_evidence": True,
                    },
                )

    # --------------------------------------------------------
    # TWO IMAGES
    # --------------------------------------------------------

    else:

        if intent["cross_modal"]:

            task = "cross_modal"
            feature = "multimodal"

            add_step(
                steps,
                "optical_sar_fusion",
                {
                    "use_primary_image": True,
                    "use_secondary_image": True,
                },
            )

            add_step(
                steps,
                "geospatial_tools",
                {
                    "generate_spatial_evidence": True,
                },
            )

        elif intent["temporal"]:

            task = "change_analysis"
            feature = "auto"

            add_step(
                steps,
                "change_engine",
                {
                    "compare_before_after": True,
                },
            )

            add_step(
                steps,
                "geospatial_tools",
                {
                    "generate_change_evidence": True,
                },
            )

        else:

            task = "multi_image_vqa"
            feature = "auto"

            add_step(
                steps,
                "rs_vqa",
                {
                    "use_primary_image": True,
                    "use_secondary_image": True,
                },
            )

    add_step(
        steps,
        "result_integrator",
    )

    return {
        "task": task,
        "feature": feature,
        "query": query,
        "image_count": image_count,
        "intent": intent,
        "steps": steps,
    }