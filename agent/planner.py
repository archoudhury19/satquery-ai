from __future__ import annotations

from typing import Any, Dict, List


def understand_query(query: str, image_count: int) -> Dict[str, Any]:
    """
    Lightweight deterministic query understanding for the MVP.

    This is intentionally predictable. It can later be replaced
    by an LLM-based planner without changing the rest of the system.
    """

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
        ]
    )

    temporal = image_count >= 2 and any(
        term in q
        for term in [
            "change",
            "changed",
            "increase",
            "decrease",
            "before",
            "after",
            "between",
            "grown",
            "shrunk",
        ]
    )

    cross_modal = image_count >= 2 and any(
        term in q
        for term in [
            "optical",
            "sar",
            "both",
            "together",
            "multimodal",
            "cross-modal",
        ]
    )

    captioning = any(
        term in q
        for term in [
            "describe",
            "caption",
            "scene",
            "land-cover",
            "land cover",
        ]
    )

    return {
        "spatial": spatial,
        "temporal": temporal,
        "cross_modal": cross_modal,
        "captioning": captioning,
    }


def build_plan(
    query: str,
    image_count: int,
) -> Dict[str, Any]:
    """
    Convert query + input configuration into an observable
    execution plan.

    The output is intentionally auditable.
    """

    intent = understand_query(query, image_count)

    steps: List[Dict[str, Any]] = [
        {
            "tool": "input_validator",
            "parameters": {},
        }
    ]

    if image_count == 1:
        if intent["captioning"]:
            task = "captioning"
            steps.append(
                {
                    "tool": "rs_captioner",
                    "parameters": {},
                }
            )
        else:
            task = "vqa"
            steps.append(
                {
                    "tool": "rs_vqa",
                    "parameters": {},
                }
            )

    elif intent["cross_modal"]:
        task = "cross_modal"

        steps.extend(
            [
                {
                    "tool": "optical_sar_fusion",
                    "parameters": {},
                },
                {
                    "tool": "geospatial_tools",
                    "parameters": {},
                },
            ]
        )

    elif intent["temporal"]:
        task = "change_analysis"

        steps.extend(
            [
                {
                    "tool": "change_engine",
                    "parameters": {},
                },
                {
                    "tool": "geospatial_tools",
                    "parameters": {},
                },
            ]
        )

    else:
        task = "multi_image_vqa"

        steps.append(
            {
                "tool": "rs_vqa",
                "parameters": {},
            }
        )

    # Add GIS evidence where spatial information was requested.
    if intent["spatial"] and not any(
        s["tool"] == "geospatial_tools"
        for s in steps
    ):
        steps.append(
            {
                "tool": "geospatial_tools",
                "parameters": {},
            }
        )

    steps.append(
        {
            "tool": "result_integrator",
            "parameters": {},
        }
    )

    return {
        "task": task,
        "query": query,
        "image_count": image_count,
        "steps": steps,
    }