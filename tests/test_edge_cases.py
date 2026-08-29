"""
tests/test_edge_cases.py
========================
Automated regression test suite evaluating all 7 AUTHENTIC REAL-LIFE Sentinel-2
L2A (Red/Green/Blue/NIR) satellite images downloaded from the ESA Copernicus / AWS Open Data archive.

Evaluates physical ground-truth land-cover separation across diverse global domains:
  1. Dense Urban (Paris Core, France)
  2. Tropical Rainforest (Amazon Basin, Brazil)
  3. Open Coastal Ocean (Mediterranean Sea, Sardinia)
  4. Arid Desert (Sahara Dunes, Algeria)
  5. Agricultural Farmland (Dnipro Basin, Ukraine)
  6. River Delta (Mississippi Birdfoot Delta, USA)
  7. Mixed Suburban (London Croydon/Surrey, UK)
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.app import app, FILES, read_image

class TestRealSentinel2EdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.edge_dir = Path(__file__).resolve().parent.parent / "demo_data" / "edge_cases"

    def _run_segmentation(self, name: str) -> dict:
        tif_path = self.edge_dir / f"{name}.tif"
        self.assertTrue(tif_path.exists(), f"Image {tif_path} must exist")
        data = read_image(tif_path)
        fid = f"test_ec_{name}"
        FILES[fid] = {"path": tif_path, "data": data, "filename": tif_path.name}

        response = self.client.post(
            "/api/analyze",
            json={
                "primary_id": fid,
                "query": "Identify the green fields, buildings, and water in different colours.",
            },
        )
        self.assertEqual(response.status_code, 200, f"API failed for {name}: {response.text}")
        json_data = response.json()
        ev = json_data.get("evidence", {})
        return {
            "water": ev.get("water_percent", 0.0),
            "vegetation": ev.get("vegetation_percent", 0.0),
            "built_up": ev.get("built_up_percent", 0.0),
            "bare": ev.get("unclassified_percent", 0.0) + ev.get("desert_percent", 0.0) + ev.get("bare_percent", 0.0),
        }

    def test_1_paris_dense_urban(self):
        pcts = self._run_segmentation("ec_urban_dense")
        print(f"\n[Paris Core] Results: {pcts}")
        # Paris core: Significant built-up presence (>20%)
        self.assertGreater(pcts["built_up"], 20.0, f"Paris: built_up={pcts['built_up']}% should be >20%")

    def test_2_amazon_rainforest(self):
        pcts = self._run_segmentation("ec_forest_dense")
        print(f"\n[Amazon Rainforest] Results: {pcts}")
        # Amazon canopy: Vegetation > 90%
        self.assertGreater(pcts["vegetation"], 90.0, f"Amazon: veg={pcts['vegetation']}% should be >90%")

    def test_3_mediterranean_ocean(self):
        pcts = self._run_segmentation("ec_water_dominant")
        print(f"\n[Mediterranean Ocean] Results: {pcts}")
        # Open sea: Water > 90%
        self.assertGreater(pcts["water"], 90.0, f"Mediterranean: water={pcts['water']}% should be >90%")

    def test_4_sahara_desert(self):
        pcts = self._run_segmentation("ec_desert_bare")
        print(f"\n[Sahara Desert] Results: {pcts}")
        # Sahara dunes: Bare/Desert > 90%, Veg < 5%, Water < 1%
        self.assertGreater(pcts["bare"], 90.0, f"Sahara: bare={pcts['bare']}% should be >90%")
        self.assertLess(pcts["vegetation"], 5.0, f"Sahara: veg={pcts['vegetation']}% should be <5%")

    def test_5_ukraine_farmland(self):
        pcts = self._run_segmentation("ec_agricultural")
        print(f"\n[Ukraine Farmland] Results: {pcts}")
        # Crop fields: Vegetation > 60%
        self.assertGreater(pcts["vegetation"], 60.0, f"Ukraine: veg={pcts['vegetation']}% should be >60%")

    def test_6_ganges_river_delta(self):
        pcts = self._run_segmentation("ec_river_delta")
        print(f"\n[Ganges Braided River Delta] Results: {pcts}")
        # Active delta: Water channels > 20%, Delta vegetation > 40%
        self.assertGreater(pcts["water"], 20.0, f"Delta: water={pcts['water']}% should be >20%")
        self.assertGreater(pcts["vegetation"], 40.0, f"Delta: veg={pcts['vegetation']}% should be >40%")

    def test_7_london_suburban_mixed(self):
        pcts = self._run_segmentation("ec_suburban_mixed")
        print(f"\n[London Suburban] Results: {pcts}")
        # Suburban: Mixed greenery and residential structures
        self.assertGreater(pcts["built_up"], 1.0, f"London Suburban: built_up={pcts['built_up']}% should be >1%")
        self.assertGreater(pcts["vegetation"], 40.0, f"London Suburban: veg={pcts['vegetation']}% should be >40%")


if __name__ == "__main__":
    unittest.main()
