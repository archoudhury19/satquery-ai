from __future__ import annotations

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app import app, FILES, read_image


class TestRiverDetectionAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        p_kol = Path("uploads/d32885b32880498cb70fc53f9446e43b_georef.tif")
        d_kol = read_image(p_kol)
        FILES["test_kol"] = {"path": p_kol, "data": d_kol, "filename": p_kol.name}

    def _assert_grounding_response(self, res, feature="water"):
        """Shared assertions for grounding responses."""
        self.assertEqual(res["task"], "grounding")
        self.assertEqual(res["feature"], feature)
        self.assertIsNotNone(res["overlay_url"])
        self.assertIsNotNone(res["bounding_box"])
        # Answer may say "Detected water" (legacy) OR "Localized ..." (CLIP grounding)
        answer_lower = res["answer"].lower()
        self.assertTrue(
            "water" in answer_lower
            or "river" in answer_lower
            or "localized" in answer_lower,
            f"Expected water/river/localized in answer, got: {res['answer']}"
        )

    def test_find_river(self):
        res = self.client.post(
            "/api/analyze",
            json={"primary_id": "test_kol", "query": "Find the river"},
        ).json()
        self._assert_grounding_response(res)

    def test_where_is_the_river(self):
        res = self.client.post(
            "/api/analyze",
            json={"primary_id": "test_kol", "query": "Where is the river?"},
        ).json()
        self._assert_grounding_response(res)

    def test_highlight_water_body(self):
        res = self.client.post(
            "/api/analyze",
            json={"primary_id": "test_kol", "query": "Highlight the water body referred to in the query."},
        ).json()
        self._assert_grounding_response(res)

    def test_vqa_water_body_tile(self):
        res = self.client.post(
            "/api/analyze",
            json={"primary_id": "test_kol", "query": "Is there a water body in the north-west part of this tile?"},
        ).json()
        self.assertEqual(res["feature"], "water")
        self.assertIsNotNone(res["overlay_url"])
        answer_lower = res["answer"].lower()
        self.assertTrue(
            "water" in answer_lower or "localized" in answer_lower,
            f"Expected water/localized in answer, got: {res['answer']}"
        )


if __name__ == "__main__":
    unittest.main()
