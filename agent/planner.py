from __future__ import annotations

from typing import Any, Dict, List


def understand_query(
    query: str,
    image_count: int,
) -> Dict[str, Any]:

    q = query.lower().strip()

    spatial = any(
        term in q
        for term in [
            "where",
            "highlight",
            "locate",
            "region",
            "area",
            "coordinates",
            "show me where",
            "point out",
            "find",
            "detect",
            "identify",
            "isolate",
            "show",
            "water body",
            "river",
            "lake",
            "reservoir",
        ]
    )

    temporal = (
        image_count >= 2
        and any(
            term in q
            for term in [
                "change",
                "changed",
                "increase",
                "increased",
                "decrease",
                "decreased",
                "before",
                "after",
                "between",
                "grown",
                "grew",
                "shrunk",
                "declined",
                "expanded",
            ]
        )
    )

    cross_modal = (
        image_count >= 2
        and any(
            term in q
            for term in [
                "sar",
                "optical",
                "optical-sar",
                "optical sar",
                "cross-modal",
                "cross modal",
                "both images",
                "together",
                "multimodal",
                "multi-modal",
                "use both",
            ]
        )
    )

    captioning = any(
        term in q
        for term in [
            "describe",
            "description",
            "caption",
            "scene",
            "scene description",
            "what is visible",
            "land-cover",
            "land cover",
        ]
    )

    grounding = (
        spatial
        and not captioning
        and any(
            term in q
            for term in [
                "highlight",
                "locate",
                "where",
                "show",
                "region",
                "point out",
                "find",
                "detect",
                "identify",
                "isolate",
                "water body",
                "river",
                "lake",
                "reservoir",
            ]
        )
    )

    return {
        "spatial": spatial,
        "temporal": temporal,
        "cross_modal": cross_modal,
        "captioning": captioning,
        "grounding": grounding,
        "vqa": not captioning and not grounding,
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

        if intent["captioning"]:

            task = "captioning"
            feature = "scene"

            add_step(
                steps,
                "rs_captioner",
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