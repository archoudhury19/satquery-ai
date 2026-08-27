from __future__ import annotations

import unittest
from agent.planner import build_plan


class TestRepresentativeQueries(unittest.TestCase):

    def test_query_1_captioning(self):
        plan = build_plan("Describe the land-cover and major objects visible in this image.", 1)
        self.assertEqual(plan["task"], "captioning")
        tools = [s["tool"] for s in plan["steps"]]
        self.assertIn("rs_captioner", tools)

    def test_query_2_grounding(self):
        plan = build_plan("Highlight the water body referred to in the query.", 1)
        self.assertEqual(plan["task"], "grounding")
        tools = [s["tool"] for s in plan["steps"]]
        self.assertIn("rs_grounding", tools)
        self.assertIn("geospatial_tools", tools)

    def test_query_3_change_analysis(self):
        plan = build_plan("What changed between these two dates, and where did the change occur?", 2)
        self.assertEqual(plan["task"], "change_analysis")
        tools = [s["tool"] for s in plan["steps"]]
        self.assertIn("change_engine", tools)
        self.assertIn("geospatial_tools", tools)

    def test_query_4_cross_modal(self):
        plan = build_plan("Use the optical and SAR images together to identify built-up and water-covered regions.", 2)
        self.assertEqual(plan["task"], "cross_modal")
        tools = [s["tool"] for s in plan["steps"]]
        self.assertIn("optical_sar_fusion", tools)
        self.assertIn("geospatial_tools", tools)

    def test_query_5_change_vqa(self):
        plan = build_plan("Has the built-up area increased, decreased, or remained unchanged?", 2)
        self.assertEqual(plan["task"], "change_analysis")
        tools = [s["tool"] for s in plan["steps"]]
        self.assertIn("change_engine", tools)


if __name__ == "__main__":
    unittest.main()
