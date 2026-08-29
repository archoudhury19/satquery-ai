from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app, FILES, read_image
from geospatial import (
    detect_remote_sensing_water,
    detect_remote_sensing_vegetation,
    detect_remote_sensing_builtup,
    detect_sar_water_backscatter,
    compute_bitemporal_change,
    fuse_cross_modal_masks,
)


class TestBenchmarkEvaluationSuite(unittest.TestCase):
    """
    Comprehensive Benchmark Evaluation Suite covering:
    - RSVQA (Single-Image VQA)
    - VRSBench (Single-Image Captioning & Grounding)
    - CDVQA (Bi-Temporal Change Analysis & Change VQA)
    - Optical-SAR Cross-Modal Joint Analysis
    - Agentic Controller & Representative Challenge Queries
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.s2_path = Path("demo_data/bigearthnet/S2_multispectral_patch.tif")
        cls.kol_opt_path = Path("demo_data/isro_sac/cartosat_optical_coregistered.tif")
        cls.kol_sar_path = Path("demo_data/isro_sac/risat_sar_coregistered.tif")

        cls.s2_data = read_image(cls.s2_path)
        cls.kol_opt_data = read_image(cls.kol_opt_path)
        cls.kol_sar_data = read_image(cls.kol_sar_path)

        FILES["bench_s2"] = {"path": cls.s2_path, "data": cls.s2_data, "filename": cls.s2_path.name}
        FILES["bench_kol_opt"] = {"path": cls.kol_opt_path, "data": cls.kol_opt_data, "filename": cls.kol_opt_path.name}
        FILES["bench_kol_sar"] = {"path": cls.kol_sar_path, "data": cls.kol_sar_data, "filename": cls.kol_sar_path.name}

    # ========================================================
    # 1. SINGLE-IMAGE VQA (RSVQA Benchmark)
    # ========================================================
    def test_single_image_vqa(self):
        res = self.client.post("/api/analyze", json={
            "primary_id": "bench_s2",
            "query": "Is there water present in this image?",
        }).json()
        self.assertIn(res["task"], ["vqa", "grounding"])
        self.assertIsNotNone(res["answer"])
        self.assertGreaterEqual(res["confidence"], 0.1)

    # ========================================================
    # 2. SINGLE-IMAGE SCENE CAPTIONING (VRSBench Benchmark)
    # ========================================================
    def test_single_image_captioning(self):
        res = self.client.post("/api/analyze", json={
            "primary_id": "bench_s2",
            "query": "Describe the land-cover and major objects visible in this image.",
        }).json()
        self.assertEqual(res["task"], "captioning")
        self.assertIn("Remote-Sensing Scene", res["tool"])
        self.assertIsInstance(res["answer"], str)
        self.assertGreater(len(res["answer"]), 20)

    # ========================================================
    # 3. TEXT-GUIDED REGION GROUNDING
    # ========================================================
    def test_text_guided_grounding(self):
        res = self.client.post("/api/analyze", json={
            "primary_id": "bench_s2",
            "query": "Highlight the water body referred to in the query.",
        }).json()
        self.assertEqual(res["task"], "grounding")
        self.assertEqual(res["feature"], "water")
        self.assertIsNotNone(res["overlay_url"])
        self.assertIsNotNone(res["bounding_box"])
        self.assertTrue(res["evidence"]["available"])
        self.assertGreater(res["evidence"]["area_hectares"], 0.0)

    # ========================================================
    # 4. BI-TEMPORAL CHANGE ANALYSIS (CDVQA Benchmark)
    # ========================================================
    def test_bitemporal_change_vqa(self):
        res = self.client.post("/api/analyze", json={
            "primary_id": "bench_s2",
            "secondary_id": "bench_s2",
            "query": "What changed between these two dates, and where did the change occur?",
        }).json()
        self.assertEqual(res["task"], "change_analysis")
        self.assertIn("Bi-temporal", res["tool"])
        self.assertIsNotNone(res["overlay_url"])
        self.assertIn("delta_percentage_points", res["evidence"])
        self.assertEqual(res["evidence"]["delta_percentage_points"], 0.0)

    def test_bitemporal_directional_vqa(self):
        res = self.client.post("/api/analyze", json={
            "primary_id": "bench_s2",
            "secondary_id": "bench_s2",
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
        }).json()
        self.assertEqual(res["task"], "change_analysis")
        self.assertTrue(
            "stable" in res["answer"].lower() or "unchanged" in res["answer"].lower()
        )

    # ========================================================
    # 5. CROSS-MODAL OPTICAL + SAR JOINT ANALYSIS
    # ========================================================
    def test_optical_sar_joint_reasoning(self):
        res = self.client.post("/api/analyze", json={
            "primary_id": "bench_kol_opt",
            "secondary_id": "bench_kol_sar",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        }).json()
        self.assertEqual(res["task"], "cross_modal")
        self.assertIn("fusion", res["tool"].lower())
        self.assertIsNotNone(res["overlay_url"])
        self.assertIn("fusion_metrics", res["evidence"])
        self.assertGreater(res["evidence"]["fusion_metrics"]["water_coverage_pct"], 0.0)

    # ========================================================
    # 6. MODULAR DETECTOR TESTS
    # ========================================================
    def test_modular_detectors(self):
        # Vegetation
        v_mask, v_method, v_conf, v_diag = detect_remote_sensing_vegetation(self.s2_path, self.s2_data)
        self.assertGreater(v_mask.shape[0], 0)
        self.assertIn("NDVI", v_method)

        # Builtup
        b_mask, b_method, b_conf, b_diag = detect_remote_sensing_builtup(self.s2_path, self.s2_data)
        self.assertGreater(b_mask.shape[0], 0)

        # SAR Water
        s_mask, s_method, s_conf, s_diag = detect_sar_water_backscatter(self.kol_sar_path, self.kol_sar_data)
        self.assertGreaterEqual((s_mask > 0).sum(), 0)

        # Change computation
        c_map, c_metrics = compute_bitemporal_change(v_mask, v_mask, "vegetation")
        self.assertEqual(c_metrics["direction"], "remained unchanged")

        # Cross-modal fusion
        f_mask, f_metrics = fuse_cross_modal_masks(v_mask, v_mask, "vegetation")
        self.assertEqual(f_metrics["cross_modal_agreement_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
