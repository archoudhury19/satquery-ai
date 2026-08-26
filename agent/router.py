from __future__ import annotations

from typing import Dict, List


def build_plan(
    query: str,
    image_count: int,
) -> Dict[str, object]:
    """
    Lightweight deterministic planner for the MVP.

    Later, this can be replaced or augmented by an LLM planner.
    """

    q = query.lower()
    steps: List[str] = []

    # Single image workflows
    if image_count == 1:

        if any(
            word in q
            for word in [
                "describe",
                "caption",
                "scene",
                "land-cover",
                "land cover",
            ]
        ):
            task = "captioning"
            steps.append("rs_vlm")

        elif any(
            word in q
            for word in [
                "highlight",
                "where",
                "locate",
                "water body",
                "reservoir",
            ]
        ):
            task = "grounding"
            steps.extend(
                ["rs_vlm", "spatial_evidence"]
            )

        else:
            task = "vqa"
            steps.append("rs_vlm")

    # Two-image workflows
    else:

        if any(
            word in q
            for word in [
                "sar",
                "optical",
                "together",
                "cross-modal",
                "multimodal",
            ]
        ):
            task = "cross_modal"
            steps.extend(
                ["optical_sar", "spatial_evidence"]
            )

        else:
            task = "change_analysis"
            steps.extend(
                ["change_vqa", "spatial_evidence"]
            )

    return {
        "task": task,
        "steps": steps,
    }