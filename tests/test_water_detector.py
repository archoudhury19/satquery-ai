from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import rasterio

from backend.app import read_image
from geospatial.water_detector import (
    calculate_ndvi,
    calculate_ndwi,
    compute_otsu_ndwi_threshold,
    detect_remote_sensing_water,
    detect_rgb_water,
    detect_spectral_water,
    resolve_spectral_bands,
)


class TestWaterDetector(unittest.TestCase):

    def test_spectral_band_resolution_named(self):
        desc = ["Red B04", "Green B03", "Blue B02", "NIR B08"]
        bands = resolve_spectral_bands(desc, 4)
        self.assertEqual(bands.get("green"), 2)
        self.assertEqual(bands.get("nir"), 4)
        self.assertEqual(bands.get("red"), 1)
        self.assertEqual(bands.get("blue"), 3)

    def test_spectral_band_resolution_fallback(self):
        desc = [None, None, None, None]
        bands = resolve_spectral_bands(desc, 4)
        self.assertEqual(bands.get("green"), 2)
        self.assertEqual(bands.get("nir"), 4)
        self.assertEqual(bands.get("red"), 1)
        self.assertEqual(bands.get("blue"), 3)

    def test_ndwi_calculation(self):
        green = np.array([[100.0, 20.0]], dtype=np.float32)
        nir = np.array([[20.0, 100.0]], dtype=np.float32)
        ndwi = calculate_ndwi(green, nir)
        # (100-20)/(100+20) = 80/120 = 0.6667
        self.assertAlmostEqual(ndwi[0, 0], 0.6667, places=3)
        # (20-100)/(20+100) = -80/120 = -0.6667
        self.assertAlmostEqual(ndwi[0, 1], -0.6667, places=3)

    def test_ndvi_calculation(self):
        nir = np.array([[100.0, 20.0]], dtype=np.float32)
        red = np.array([[20.0, 100.0]], dtype=np.float32)
        ndvi = calculate_ndvi(nir, red)
        self.assertAlmostEqual(ndvi[0, 0], 0.6667, places=3)
        self.assertAlmostEqual(ndvi[0, 1], -0.6667, places=3)

    def test_multispectral_file_detection(self):
        p = Path("uploads/1c6a451c304044858f88177f3e19fec4.tif")
        if p.exists():
            data = read_image(p)
            mask, method, conf, diag = detect_remote_sensing_water(p, data)
            self.assertEqual(mask.shape, (512, 512))
            self.assertEqual(diag["mode"], "multispectral")
            self.assertGreater(diag["water_pixels"], 0)
            self.assertGreater(conf, 0.90)

    def test_rgb_fallback_detection(self):
        p = Path("uploads/d32885b32880498cb70fc53f9446e43b.jpeg")
        if p.exists():
            data = read_image(p)
            mask, method, conf, diag = detect_remote_sensing_water(p, data)
            self.assertEqual(diag["mode"], "rgb_fallback")
            self.assertGreater(diag["water_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
