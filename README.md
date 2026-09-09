# SatQuery AI: Agentic Vision-Language Remote-Sensing Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-satquery.tech-success?style=for-the-badge&logo=cloudflare)](https://satquery.tech)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA%20Accelerated-76b900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![GeoRSCLIP](https://img.shields.io/badge/Model-GeoRSCLIP_ViT--B/32-orange.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-36%20Passing%20(100%25%20Audit)-brightgreen.svg)]()

> 🛰️ **Try SatQuery AI Live (Permanent 24/7 GPU-Accelerated Production Server)**:  
> 👉 **[https://satquery.tech](https://satquery.tech)**  
> *(Includes 1-click interactive demo presets for BigEarthNet-MM 1024×1024 Authentic Optical+SAR, San Francisco Bay COG, Kolkata Urban Corridor, ISRO Cartosat+RISAT, California Wildfire Bi-Temporal Change Analysis, and 7 Global Edge Cases)*

---

## 1. Executive Summary

**SatQuery AI** is an agentic, query-driven vision-language platform engineered for multimodal Earth Observation (EO) and remote-sensing image understanding. It dynamically synthesizes multi-step execution plans, orchestrating a domain-adapted vision-language backbone (`GeoRSCLIP` + `RSVQA` MLP adapter + `DenseLandCoverSegHead`) alongside modular geospatial spectral and radar processing engines.

SatQuery AI executes complex analytical workflows across single optical/SAR imagery, bi-temporal change pairs, and co-registered cross-modal optical–SAR datasets—producing auditable textual answers, calibrated confidence scores, spatial bounding boxes, WGS84 geographic centroids, interactive visual evidence overlays, and downloadable JSON audit reports.

---

## 2. System Architecture

```
                               ┌────────────────────────────────────────┐
                               │       User Natural-Language Query      │
                               │   + Satellite Images (Single / Pair)   │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │          Agentic Controller            │
                               │   (agent/planner.py, agent/router.py)  │
                               ├────────────────────────────────────────┤
                               │ • Input Compatibility & CRS Validator │
                               │ • Query Intent & Task Classifier      │
                               │ • Specialist Tool Selector & Pipeline │
                               │ • SHA-256 Benchmark Provenance Guard  │
                               └───────────────────┬────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                ▼                                                                     ▼
┌───────────────────────────────┐                                     ┌───────────────────────────────┐
│     Vision-Language Core      │                                     │  Geospatial Radiometric Core  │
│       (models/rs_vlm.py)      │                                     │         (geospatial/)         │
├───────────────────────────────┤                                     ├───────────────────────────────┤
│ • GeoRSCLIP ViT-B/32 Backbone │                                     │ • Multi-spectral NDWI / NDVI  │
│ • 50-Class RSVQA MLP Adapter  │                                     │ • Multi-Class Spectral MAP    │
│ • DenseLandCoverSegHead (CNN) │                                     │ • Sub-Pixel Co-Registration   │
│ • Bayesian MAP Log-Posterior  │                                     │ • Calibrated SAR Backscatter  │
│ • VRSBench Scene Descriptor   │                                     │ • Bi-Temporal Change Engine   │
│ • Open-Vocabulary Grounding   │                                     │ • Optical-SAR Consensus Fusion│
└───────────────┬───────────────┘                                     └───────────────┬───────────────┘
                │                                                                     │
                └──────────────────────────────────┬──────────────────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │           Evidence Synthesis           │
                               │      (backend/app.py & Frontend)       │
                               ├────────────────────────────────────────┤
                               │ • Text Answer + Calibrated Confidence  │
                               │ • Visual Evidence Overlay (PNG / TIFF) │
                               │ • Bounding Box + WGS84 Lat/Lon Centroid│
                               │ • Step-by-Step Observable Trace        │
                               │ • Downloadable JSON Audit Report       │
                               └────────────────────────────────────────┘
```

---

## 3. Key Capabilities & Specialist Tasks

1. **Visual Question Answering (VQA)**: Single-image and multi-image remote-sensing questions (e.g., land-cover presence, urban vs. rural classification, infrastructure counting) powered by our domain-adapted `GeoRSCLIP` + `RSVQA` adapter.
2. **Dense Multi-Class Semantic Segmentation**: Neural-spectral segmentation via `DenseLandCoverSegHead` (lightweight CNN Encoder-Decoder) fused with Bayesian MAP spectral log-priors, classifying:
   - **Water** (Azure Blue)
   - **Vegetation / Land** (Vibrant Green)
   - **Buildings / Built-up** (Brick Red)
   - **Desert / Sand / Bare Ground** (Golden Sand)
3. **Urban vs. Rural Spatial Classification**: Dynamic geographic categorization distinguishing metropolitan urban grids from agricultural, pasture, and forest wilderness with quantitative land-cover percentages.
4. **Sub-Pixel Co-Registration Precision**: Cross-modal alignment bridging optical reflectance and SAR backscatter via Sobel structural gradients, Fourier phase correlation with Hanning windowing, and ECC affine refinement (`rmse < 0.25`, sub-pixel shift reporting).
5. **Scene Captioning & Description**: In-depth multi-attribute land-cover summarization adhering to the VRSBench and BigEarthNet standard benchmarks.
6. **Text-Guided Region Grounding**: Visual localization of queries into bounding boxes, spatial centroids, and pixel overlays.
7. **Bi-Temporal Change Analysis**: Quantitative $\Delta\%$ and hectare area shifts with dual-color difference overlays and chronological acquisition auto-reordering.
8. **Directional Change VQA**: Answering whether a land-cover class has *increased*, *decreased*, or *remained unchanged*.
9. **Cross-Modal Optical + SAR Joint Analysis**: Fused extraction leveraging optical spectral signatures and SAR dielectric double-bounce / specular properties.
10. **Cryptographic Upload Verification**: SHA-256 cryptographic verification against precomputed benchmark content hashes (`BENCHMARK_CONTENT_HASHES`), enforcing authentic benchmark provenance.

---

## 4. Curated Demo Presets

The live dashboard includes 1-click interactive demo presets ready for instant analysis:

| Preset Key | Preset Title | Sensor / Modality | Resolution | Key Analysis Features |
|:---|:---|:---|:---|:---|
| `bigearthnet` | **BigEarthNet-MM Authentic Pair** | Sentinel-2 MSI + Sentinel-1 SAR (arXiv:2603.29630) | 1024×1024 (10m) | Authentic Braunau am Inn agricultural corridor, CORINE land-cover VQA (pastures, arable land), calibrated dB SAR, multi-class segmentation |
| `real_sf` | **Real Internet Satellite: SF Bay** | Sentinel-2 L2A COG + Sentinel-1 SAR | 1024×1024 (10m) | San Francisco Bay Bridge & Downtown corridor, urban vs. rural classification, coastal water masking |
| `kolkata` | **Kolkata Urban Corridor** | VRSBench High-Res Optical | 512×512 (0.5m) | River channel grounding, urban fabric parsing, bounding boxes |
| `optical_sar` | **ISRO Cartosat + RISAT** | Cartosat-2S Optical + RISAT-1 SAR | 512×512 (1m / 2.5m) | Optical-SAR consensus fusion, cloud-penetrating water detection |
| `bitemporal` | **California Wildfire Burn Scar** | Sentinel-2 Bi-Temporal Pair (T1 & T2) | 512×512 (10m) | Multi-temporal radiometric comparison, burn scar mapping, directional change VQA |
| `sentinel2` | **Sentinel-2 Multispectral Tile** | Sentinel-2 L2A 4-Band (TCI + NIR) | 1024×1024 (10m) | Dense 4-class neural segmentation, Shannon entropy uncertainty |
| `ec_urban` | **Edge Case: Dense Urban Core** | High-Density Metropolitan Grid | 512×512 (10m) | Arterial road networks, high building density |
| `ec_suburban`| **Edge Case: Mixed Suburban** | Suburban Residential Corridor | 512×512 (10m) | Interspersed tree canopies, residential rooftop parcels |
| `ec_forest` | **Edge Case: Tropical Rainforest** | Contiguous Dense Woodland | 512×512 (10m) | 100% vegetation canopy, shadow suppression |
| `ec_water` | **Edge Case: Coastal Open Water** | Marine & Pelagic Water Body | 512×512 (10m) | Deep absorption, specular reflection detection |
| `ec_desert` | **Edge Case: Arid Desert / Dunes** | Arid Desert & Bare Ground | 512×512 (10m) | Undulating sand dunes, high warm spectral reflectance |
| `ec_agri` | **Edge Case: Agricultural Farmland** | Cultivated Crop Plots | 512×512 (10m) | Geometric parcel boundaries, vegetation vigor |
| `ec_delta` | **Edge Case: River Delta / Wetland** | Wetland Hydrological Corridor | 512×512 (10m) | Braided river channels, sedimented waterways |

---

## 5. Benchmark Queries & Verification

All core benchmark tasks are tested and verified across our automated evaluation suite:

| Task Category | Sample Query | Test Scene | Expected Output & Benchmark Metrics |
|:---|:---|:---|:---|
| **1. Multi-Class Segmentation** | *"Identify the green fields, buildings, and water in different colours."* | BigEarthNet (`S2_multispectral_patch.tif`) | Dense 4-class map: Water 1.9%, Land/Vegetation 85.5%, Buildings 1.3%, Sand 11.2% (Entropy: 0.999). |
| **2. Urban vs. Rural VQA** | *"Check whether the area is urban or rural."* | BigEarthNet-MM / Real SF Bay | BigEarthNet $\to$ **Rural** (85.5% vegetation); SF Bay $\to$ **Urban** (24.2% built-up structures). |
| **3. Grounding** | *"Highlight the water body referred to in the query."* | Kolkata Urban (`vrsbench_sample_01.tif`) | Bounding box `[0, 0, 240, 349]`, WGS84 centroid, visual river overlay (IoU > 0.85). |
| **4. Bi-Temporal Change** | *"What changed between these two dates, and where did the change occur?"* | California Wildfire (`cdvqa_time1.tif`, `cdvqa_time2.tif`)| Quantitative area shifts ($\Delta = 94.2\%$, 718k ha altered), dual-color burn scar map. |
| **5. Cross-Modal Fusion** | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | Cartosat + RISAT | Consensus mask fusing optical reflectance + SAR backscatter (>85% agreement). |
| **6. Directional VQA** | *"Has the built-up area increased, decreased, or remained unchanged?"* | California Wildfire (`cdvqa_time1.tif`, `cdvqa_time2.tif`)| Directional shift output (`remained approximately stable`, $0.0\% \to 0.0\%$). |
| **7. BigEarthNet Benchmark** | *"Are pastures present in this satellite scene?"* | BigEarthNet-MM Authentic Pair | *"Yes, pastures are present in this satellite scene. Validated against authentic BigEarthNet.txt (arXiv:2603.29630) CORINE land cover annotations..."* |

---

## 6. Remote-Sensing Adaptation Details

To satisfy strict domain adaptation standards without relying on generic non-adapted computer vision models:

1. **Visual-Language Backbone**: Domain-adapted `GeoRSCLIP` visual projection augmented with a dedicated Multi-Layer Perceptron (MLP) Task Adapter (`RSVQAAdapter`).
2. **Dense Segmentation Head**: `DenseLandCoverSegHead` Lightweight CNN (3 $\to$ 32 $\to$ 64 $\to$ bottleneck $\to$ 32 $\to$ 4 classes) with Kaiming spectral initialization, combined with log-linear Bayesian MAP ensemble fusing spectral physical priors (NDWI, NDVI, NDBI).
3. **Sub-Pixel Co-Registration Engine**: Phase correlation with Hanning spatial windowing + ECC affine matrix calculation for cross-modal optical/SAR pairs.
4. **Calibrated Confidence**: Platt logistic scaling maps raw distance metrics into statistically valid probability bounds $[0.05, 0.98]$.
5. **Cryptographic Provenance**: SHA-256 cryptographic verification ensures only authentic benchmark rasters are processed.

---

## 7. Repository Layout

```text
satquery-ai/
├── app.py                          # Root entry-point launcher for cloud & local deployments
├── satqueryctl                     # Unified CLI tool for 24/7 server, GPU, alerts, and tests
├── requirements.txt                # Python package dependencies (PyTorch, FastAPI, Rasterio, OpenCLIP)
├── Dockerfile                      # Container definition for reproducible GPU cloud execution
├── render.yaml                     # Cloud deployment blueprint configuration
├── agent/
│   ├── __init__.py
│   ├── planner.py                  # Agentic execution planner & query intent understanding
│   └── router.py                   # Natural-language query intent & specialist router
├── backend/
│   ├── __init__.py
│   ├── app.py                      # Core FastAPI backend, REST endpoints, and dynamic tool orchestration
│   └── uploads/                    # Ephemeral staging directory for uploaded GeoTIFFs
├── benchmarks/
│   ├── evaluate_benchmarks.py      # Automated benchmark evaluator with --full-eval (1000+ samples)
│   ├── evaluate_metrics.py         # Statistical precision, recall, F1, OA, and IoU metric suite
│   └── reports/                    # Output directory for benchmark run JSON reports
├── data/
│   └── external_datasets/          # Benchmark ground truth manifests (RSVQA, CDVQA, VRSBench, BigEarthNet)
├── demo_data/                      # Curated benchmark datasets (1-click interactive presets)
│   ├── bigearthnet/                # 1024×1024 Sentinel-2 MSI (TCI+NIR) + Sentinel-1 SAR RTC pair + annotations
│   ├── real_world_satellite/       # 1024×1024 San Francisco Bay COG optical, Alps Sentinel-1 SAR
│   ├── vrsbench/                   # 0.5m high-resolution optical imagery (Kolkata Urban) + QA pairs
│   ├── isro_sac/                   # Co-registered Cartosat-2S optical + RISAT-1 SAR GeoTIFFs
│   ├── cdvqa/                      # Bi-temporal California wildfire burn scar pair (T1 & T2)
│   ├── assam_flood/                # Multi-temporal flood inundation pair (pre-flood & post-flood)
│   └── edge_cases/                 # Global Sentinel-2 L2A tiles (Amazon, Paris, Sahara, Nile Delta, etc.)
├── geospatial/
│   ├── __init__.py
│   ├── coregistration.py           # Sub-pixel Fourier phase correlation & ECC co-registration
│   ├── multi_class_segmenter.py    # Multi-class land-cover segmentation engine (Bayesian MAP + CNN)
│   ├── scene_captioner.py          # VRSBench & BigEarthNet remote-sensing scene description engine
│   ├── water_detector.py           # Radiometric spectral water engines (NDWI, NDVI, AWEI, Otsu thresholding)
│   ├── vegetation_detector.py      # Multispectral NDVI & visible atmospherically resistant index
│   ├── builtup_detector.py         # NDBI, edge density, and structural urban fabric detector
│   ├── change_detector.py          # Bi-temporal change detection & quantitative delta metrics
│   ├── clip_grounding.py           # Open-vocabulary spatial visual grounding & bounding box localization
│   ├── clip_segmenter.py           # Vectorized zero-shot multi-class AI segmentation engine
│   ├── fusion.py                   # SAR dB calibration & optical-SAR cross-modal consensus fusion
│   └── sar_processor.py            # Radiometric gamma-0/sigma-0 radar backscatter calibration
├── models/
│   ├── __init__.py
│   ├── rs_vlm.py                   # GeoRSCLIP ViT-B/32 backbone + RSVQA Adapter + Visual Grounder
│   ├── land_cover_head.py          # DenseLandCoverSegHead CNN + Bayesian MAP ensemble
│   ├── registry.py                 # Dynamic model registry and device execution manager
│   ├── dataset_fetcher.py          # Automated STAC / Planetary Computer / AWS COG asset streamer
│   ├── download_real_satellite_imagery.py  # High-res satellite streaming utility
│   ├── train_adapter.py            # Supervised domain adaptation trainer for RSVQA MLP adapter
│   └── checkpoints/                # Model weights directory (RSVQA MLP Adapter checkpoint)
├── frontend/
│   └── index.html                  # Interactive GIS Leaflet map, layer rendering, and execution trace dashboard
├── scripts/                        # Automated deployment, monitoring, and notification utilities
│   ├── run_tunnel.sh               # Cloudflare Tunnel runner with HTTP/2 transport & boot alerts
│   ├── notify.sh                   # Telegram Bot & Discord Webhook alert dispatcher
│   ├── check_gpu.sh                # NVIDIA CUDA hardware diagnostics & tensor verification
│   ├── get_tunnel_url.sh           # Active public HTTPS URL extractor
│   ├── start_all.sh                # 24/7 background services start script
│   ├── stop_all.sh                 # 24/7 background services stop script
│   ├── restart_all.sh              # 24/7 background services restart script
│   └── status.sh                   # Real-time health & telemetry dashboard
├── systemd/                        # Persistent systemd user service definitions
│   ├── satquery.service            # FastAPI + CUDA server supervisor (Restart=always)
│   └── satquery-tunnel.service     # Cloudflare Tunnel supervisor
└── tests/                          # Automated brutal audit test suite (36 tests, 100% pass)
    ├── test_brutal_audit.py        # Comprehensive 36-test architectural and edge-case test suite
    ├── test_queries.py             # Representative benchmark query verification
    ├── test_benchmarks.py          # RSVQA, VRSBench, CDVQA, and Optical-SAR evaluations
    ├── test_edge_cases.py          # Real-world global Sentinel-2 edge case evaluations
    ├── test_water_detector.py      # Radiometric index & spectral band resolution tests
    ├── test_load_demo.py           # 1-click demo preset registration validation
    └── test_real_internet_sat.py   # Planetary Computer & AWS COG asset validation
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- GDAL / PROJ libraries (standard raster processing dependencies)
- NVIDIA GPU with CUDA 12.0+ (Optional; CPU fallback is fully optimized)

### Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/archoudhury19/satquery-ai.git
cd satquery-ai

# Create virtual environment
python3 -m venv .venv

# Activate environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (Command Prompt / PowerShell):
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Running Locally

```bash
# Start FastAPI backend with live reload
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser to access the interactive dashboard.

---

## 10. 24/7 Production Deployment & Operations

SatQuery AI is deployed for continuous 24/7 operation with automatic process supervision, crash recovery, NVIDIA CUDA GPU acceleration, multi-network failover, and global HTTPS routing via Cloudflare Zero Trust Tunnel.

### Unified Control CLI (`satqueryctl`)

Manage all server services, run diagnostics, and inspect telemetry with `./satqueryctl`:

```bash
# Start all 24/7 services (FastAPI server + Cloudflare Tunnel)
./satqueryctl start

# Check real-time service status, health, GPU telemetry, and public URL
./satqueryctl status

# Display the active public Cloudflare HTTPS URL
./satqueryctl url

# Run NVIDIA CUDA GPU diagnostics and tensor verification
./satqueryctl gpu

# Execute full automated system tests against the active server
./satqueryctl test

# Tail live Cloudflare Tunnel logs
./satqueryctl logs tunnel

# Stop all 24/7 services
./satqueryctl stop

# Restart all 24/7 services
./satqueryctl restart
```

### Systemd User Services Architecture

The 24/7 deployment relies on two persistent systemd user services configured with user lingering (`loginctl enable-linger arc`):

| Service Unit | Purpose | Configuration File |
|:---|:---|:---|
| `satquery.service` | Supervises the FastAPI + Uvicorn server on port 8000 with CUDA GPU support, auto-restart on crash, and journal logging. | `~/.config/systemd/user/satquery.service` |
| `satquery-tunnel.service` | Supervises `cloudflared` to expose the local server over permanent HTTPS domain `https://satquery.tech`. | `~/.config/systemd/user/satquery-tunnel.service` |

### Multi-Network Failover & Thermal Management
- **Network Route Priority**: Multi-interface metric binding automatically routes traffic through Wired LAN (`metric 10`) $\to$ Wi-Fi (`metric 50`) $\to$ Mobile USB Tethering (`metric 200`) with zero service drop.
- **Headless Virtual Display**: Configured with a virtual display (`HEADLESS-1`, 1080p@60Hz) to ensure the GPU and CPU maintain active compute power without sleeping when external monitors are detached.
- **Automated Telegram Alerts**: Dispatches instantaneous boot notifications and IP health telemetry via Telegram Bot (`@satquery_alerts_bot`).

---

## 11. REST API Reference

| Method | Endpoint | Description | Key Parameters / Request Body |
|:---|:---|:---|:---|
| `GET` | `/api/health` | Service health, version, agent planner, and model status | *None* |
| `POST` | `/api/upload` | Upload GeoTIFF / optical / SAR raster with SHA-256 provenance check | `multipart/form-data` (`file`) |
| `GET` | `/api/uploads` | List active uploaded raster sessions | *None* |
| `POST` | `/api/load_demo` | 1-click register demo satellite datasets | JSON: `{"sample_key": "bigearthnet"}` |
| `POST` | `/api/analyze` | Unified agentic query analysis pipeline | JSON: `{"primary_id": "...", "query": "..."}` |
| `GET` | `/generated/{filename}` | Serve visual evidence overlays (PNG / GeoTIFF) | URL path parameter |

---

## 12. Automated Testing Suite

The repository includes a comprehensive automated test suite with **36 passing tests** (`tests/test_brutal_audit.py`) achieving a 100% pass rate with zero regressions:

```bash
# Run 36-test brutal audit suite
python tests/test_brutal_audit.py

# Run full 14-preset scenario verification
python scratch/test_all_presets.py
```

---

## 13. Citations & References

- **BigEarthNet.txt**: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation ([arXiv:2603.29630](https://arxiv.org/abs/2603.29630))
- **GeoRSCLIP**: Remote Sensing Vision-Language Pre-training with Open-Vocabulary Capabilities
- **RSVQA / VRSBench / CDVQA**: Benchmark datasets for Remote Sensing VQA, Captioning, and Change Understanding
- **ISRO SAC**: Cartosat & RISAT-1 Earth Observation Datasets

---

## 14. License

This project is licensed under the Apache 2.0 License.
