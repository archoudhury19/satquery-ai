# SatQuery AI: Agentic Vision-Language Remote-Sensing Intelligence Platform

**SatQuery AI** is an agentic, query-driven vision-language platform designed for advanced remote-sensing image analysis. It seamlessly orchestrates domain-adapted vision-language models (GeoRSCLIP + RSVQA Adapter) and modular geospatial tools to answer natural-language queries across single optical/SAR images, bi-temporal change pairs, and co-registered cross-modal optical–SAR datasets.

---

## Key Features

1. **Remote-Sensing Domain Adaptation**:
   - **Base Model**: `GeoRSCLIP ViT-B/32` pre-trained on multisensor remote-sensing image-text representations.
   - **VQA Adaptation**: Multi-layer perceptron (MLP) adapter trained for remote-sensing question answering across 50 domain classes (adapted via BigEarthNet / RSVQA representations).
   - **Text-Guided Grounding**: 4x4 spatial tile patch projection for natural language localization.

2. **Single-Image Baselines (RSVQA & VRSBench)**:
   - **Visual Question Answering (VQA)**: Class prediction with Top-5 confidence rankings.
   - **Scene Captioning**: Multi-feature scene description and land-cover summarization.
   - **Text-Guided Grounding**: Bounding boxes, WGS84 centroids, equal-area hectare metrics, and visual overlays.

3. **Bi-Temporal Change Analysis (CDVQA Benchmark)**:
   - Automated before/after comparison ($T_1 \to T_2$).
   - Direction classification (`increased`, `decreased`, `stable`).
   - Quantitative area shifts in hectares, delta percentage points, and relative change %.
   - Dual-color spatial difference map overlays.

4. **Cross-Modal Optical + SAR Joint Analysis (ISRO/SAC Benchmark)**:
   - Co-registered Optical/Multispectral (Sentinel-2 / Cartosat) + SAR (Sentinel-1 / RISAT).
   - SAR radiometric backscatter calibration ($\text{dB} = 10 \log_{10} \text{DN}$).
   - Geospatial reprojection (`rasterio.warp`) across coordinate reference systems.
   - Sub-pixel registration-tolerant consensus fusion and cross-modal agreement scoring.

5. **Agentic Orchestration & Auditable Controller**:
   - Natural language query understanding and task classification.
   - Modality, CRS, and dimension validation.
   - Multi-step execution planning with observable execution traces.
   - Downloadable JSON audit reports.

---

## Repository Structure

```
satquery-ai/
├── agent/                      # Agentic planner and task router
│   ├── planner.py              # Query understanding and multi-step plan builder
│   └── router.py               # Task routing rules
├── backend/                    # FastAPI web server and API handlers
│   └── app.py                  # Core API application and pipeline coordinator
├── benchmarks/                 # Benchmark evaluation runners
│   └── evaluate_benchmarks.py  # RSVQA, VRSBench, CDVQA, and ISRO/SAC benchmark evaluator
├── demo_data/                  # Curated benchmark test images
│   ├── 01_kolkata_optical_georef.tif
│   ├── 02_kolkata_sar_sentinel1.tif
│   ├── 03_sentinel2_multispectral_t1.tif
│   ├── 04_sentinel2_multispectral_t2.tif
│   └── 05_kolkata_optical.jpg
├── frontend/                   # Single-page interactive web dashboard
│   └── index.html              # Multi-tab viewer, live chat, spatial evidence card, trace
├── geospatial/                 # Modular remote-sensing spectral & SAR engines
│   ├── builtup_detector.py     # NDBI & optical urban texture extraction
│   ├── change_detector.py      # Bi-temporal subtraction & dual-color change mapping
│   ├── fusion.py               # Reprojection & optical-SAR consensus fusion
│   ├── sar_processor.py        # Calibrated SAR dB backscatter extraction
│   ├── vegetation_detector.py  # Multispectral NDVI & ExG vegetation extraction
│   └── water_detector.py       # Multispectral NDWI, NDVI gating, & turbid river detection
├── models/                     # Vision-Language models and adaptation checkpoints
│   ├── checkpoints/            # Pre-trained GeoRSCLIP weights and adapter
│   ├── registry.py             # Predefined specialist tool registry
│   ├── rs_vlm.py               # GeoRSCLIP + RSVQA adapter model implementation
│   └── train_adapter.py        # Adapter fine-tuning script for BigEarthNet/RSVQA
├── tests/                      # Automated test suite
│   ├── test_benchmarks.py      # Benchmark evaluation tests
│   ├── test_queries.py         # Representative query routing tests
│   ├── test_river_api.py       # Grounding and river detection tests
│   └── test_water_detector.py  # Physical spectral index unit tests
└── requirements.txt            # Python dependencies
```

---

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/archoudhury19/satquery-ai.git
cd satquery-ai

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the SatQuery AI Server

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to interact with the dashboard.

### 3. Run Automated Tests & Benchmark Evaluation

```bash
# Run all unit and integration tests (22 tests)
python -m unittest discover -s tests -p "test_*.py"

# Run full benchmark evaluation
python benchmarks/evaluate_benchmarks.py
```

---

## Representative Challenge Queries

| Target Task | Example Query | Primary Model / Specialist Tool |
| :--- | :--- | :--- |
| **Scene Captioning** | *"Describe the land-cover and major objects visible in this image."* | `rs_captioner` |
| **Spatial Grounding** | *"Highlight the water body referred to in the query."* | `rs_grounding` + `geospatial_tools` |
| **Bi-Temporal Change VQA** | *"What changed between these two dates, and where did the change occur?"* | `change_engine` + `geospatial_tools` |
| **Directional Change VQA** | *"Has the built-up area increased, decreased, or remained unchanged?"* | `change_engine` |
| **Cross-Modal Optical + SAR** | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | `optical_sar_fusion` + `geospatial_tools` |

---

## Benchmark Evaluation Results

| Benchmark Dataset | Evaluated Task | Evaluation Metric | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **RSVQA** | Single-Image VQA | Top-5 Confidence Prediction | Top-1 Confidence: `16.0%`, Class: `rural` | **PASSED** |
| **VRSBench** | Scene Captioning | Multi-class Land-Cover Summary | Structured Land-Cover Text | **PASSED** |
| **VRSBench** | Region Grounding | Spatial Localization & Bounding Box | BBox `(9, 12, 416, 348)`, Area `94.87 ha` | **PASSED** |
| **CDVQA** | Change Description & VQA | Area Shift & Delta Coverage | $\Delta = +0.83 \text{ pp}$, Relative: $+26.0\%$ | **PASSED** |
| **CDVQA** | Directional Change VQA | Directional Trend | `increased` | **PASSED** |
| **ISRO/SAC** | Optical + SAR Joint Analysis | Consensus Agreement & Fused Area | Agreement: `13.13%`, Fused: `1.08%` | **PASSED** |

---

## License & Citations
- **BigEarthNet.txt Paper**: [arXiv:2603.29630](https://arxiv.org/abs/2603.29630)
- **GeoRSCLIP**: Remote Sensing Vision-Language Pre-training
- **RSVQA / VRSBench / CDVQA**: Remote Sensing Vision-Language Benchmarks
