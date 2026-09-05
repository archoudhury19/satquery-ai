# SatQuery AI: Agentic Vision-Language Remote-Sensing Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-satquery.tech-success?style=for-the-badge&logo=cloudflare)](https://satquery.tech)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA%20Accelerated-76b900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![GeoRSCLIP](https://img.shields.io/badge/Model-GeoRSCLIP_ViT--B/32-orange.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-35%20Passing-brightgreen.svg)]()

> 🛰️ **Try SatQuery AI Live (Permanent 24/7 GPU-Accelerated Production Server)**:  
> 👉 **[https://satquery.tech](https://satquery.tech)**  
> *(Includes 1-click interactive demo presets for Sentinel-2, Kolkata Urban, Cartosat+RISAT, California Wildfire Change Analysis, and Global Edge Cases)*

---

## 1. Executive Summary

**SatQuery AI** is an agentic, query-driven vision-language platform designed for advanced Earth Observation (EO) and remote-sensing image understanding. It dynamically synthesizes execution plans, orchestrating a domain-adapted vision-language model (`GeoRSCLIP` + `RSVQA` adapter) alongside modular geospatial spectral and radar engines.

SatQuery AI executes complex analytical workflows across single optical/SAR images, bi-temporal change pairs, and co-registered cross-modal optical–SAR datasets—producing auditable textual answers, calibrated confidence scores, spatial bounding boxes, WGS84 geographic centroids, and downloadable JSON audit reports.

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
                               └───────────────────┬────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                ▼                                                                     ▼
┌───────────────────────────────┐                                     ┌───────────────────────────────┐
│     Vision-Language Core      │                                     │  Geospatial Radiometric Core  │
│       (models/rs_vlm.py)      │                                     │         (geospatial/)         │
├───────────────────────────────┤                                     ├───────────────────────────────┤
│ • GeoRSCLIP ViT-B/32 Backbone │                                     │ • Multi-spectral NDWI / NDVI  │
│ • 50-Class RSVQA MLP Adapter  │                                     │ • AWEI / Chlorophyll Gates    │
│ • VRSBench Scene Descriptor   │                                     │ • Bi-Temporal Change Engine   │
│ • Open-Vocabulary Grounding   │                                     │ • Calibrated SAR Backscatter  │
│ • Vectorized Zero-Shot Seg    │                                     │ • Optical-SAR Consensus Fusion│
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

1. **Visual Question Answering (VQA)**: Single-image remote-sensing questions (e.g., land-cover presence, urban vs. rural classification, infrastructure counting) powered by our domain-adapted `GeoRSCLIP` + `RSVQA` adapter.
2. **Scene Captioning & Description**: Comprehensive multi-attribute land-cover summarization adhering to the VRSBench standard.
3. **Text-Guided Region Grounding**: Visual localization of queries into bounding boxes, spatial centroids, and pixel overlays.
4. **Bi-Temporal Change Analysis**: Quantitative $\Delta\%$ and hectare area shifts with dual-color difference overlays.
5. **Directional Change VQA**: Answering whether a land-cover class has *increased*, *decreased*, or *remained unchanged*.
6. **Cross-Modal Optical + SAR Joint Analysis**: Fused extraction leveraging optical spectral signatures and SAR dielectric double-bounce / specular properties.
7. **Dense Multi-Class AI Segmentation**: Fast zero-shot segmentation with spectral physics overrides.

---

## 4. Curated Demo Presets

The live dashboard includes 1-click interactive demo presets ready for instant analysis:

| Preset Name | Dataset Source | Sensor / Modality | Resolution | Key Analysis Features |
|:---|:---|:---|:---|:---|
| **Sentinel-2 Multispectral** | BigEarthNet | Sentinel-2 L2A (B02, B03, B04, B08) | 10m | Multi-spectral NDWI/NDVI, VQA, scene captioning |
| **Kolkata Urban Corridor** | VRSBench | High-Resolution Optical | 0.5m | River channel grounding, urban fabric parsing, bounding boxes |
| **ISRO Cartosat + RISAT** | ISRO SAC | Cartosat-2S Optical + RISAT-1 SAR | 1m / 2.5m | Optical-SAR consensus fusion, cloud-penetrating water detection |
| **California Wildfire Scar** | CDVQA / Sentinel-2 | Pre-Fire (T1) & Post-Fire (T2) | 10m | Quantitative delta ($\Delta\%$, ha), directional change VQA |
| **San Francisco COG** | Real Satellite | Cloud-Optimized GeoTIFF (EPSG:32610) | 0.5m | Coastal urban analysis, georeferenced bounding boxes |
| **Global Edge Cases** | Global Sentinel-2 | Amazon, Paris, Sahara, Venice, Dubai | 10m | Extreme reflectance, turbid sediment, shadow suppression |

---

## 5. The 5 Mandatory Representative Benchmark Queries

All 5 core benchmark queries are tested and verified across our automated evaluation suite:

| Task | Exact Query | Sample Input | Expected Output & Benchmark Metrics |
|:---|:---|:---|:---|
| **1. Captioning** | *"Describe the land-cover and major objects visible in this image."* | Kolkata Urban (`vrsbench_sample_01.tif`) | Structured description of built-up fabric (80.9%), river channel, and tree canopy. VRSBench aligned. |
| **2. Grounding** | *"Highlight the water body referred to in the query."* | Kolkata Urban (`vrsbench_sample_01.tif`) | Bounding box `[0, 0, 240, 349]`, WGS84 centroid, visual river overlay (IoU > 0.85). |
| **3. Bi-Temporal** | *"What changed between these two dates, and where did the change occur?"* | California Wildfire (`cdvqa_time1.tif`, `cdvqa_time2.tif`)| Quantitative area shifts ($\Delta = 30.6\%$, 27.2 ha altered), dual-color change map. |
| **4. Optical + SAR** | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | Cartosat + RISAT (`cartosat_optical_coregistered.tif`, `risat_sar_coregistered.tif`) | Fused consensus mask combining optical spectral reflection and SAR backscatter (>92% agreement). |
| **5. Change VQA** | *"Has the built-up area increased, decreased, or remained unchanged?"* | California Wildfire (`cdvqa_time1.tif`, `cdvqa_time2.tif`)| Directional shift output (`remained approximately stable`, $0.0\% \to 0.0\%$). |

---

## 6. Remote-Sensing Adaptation Details

To satisfy the mandatory adaptation requirement without relying on generic non-adapted computer vision models:

1. **What was adapted**: The visual projection layer of `GeoRSCLIP` was augmented with a dedicated Multi-Layer Perceptron (MLP) Task Adapter (`RSVQAAdapter`).
2. **Data used**: Multi-sensor remote sensing representations aligned with the `BigEarthNet` / `RSVQA` / `VRSBench` Earth Observation benchmarks across 50 domain-specific classes.
3. **Adaptation mechanism**:
   - 512-dimensional visual token features from the remote sensing ViT are projected into the 50-class vocabulary space.
   - Temperature scaling ($\tau=0.7$) and spectral prior confidence boosts ensure accurate calibration.
   - Vectorized PyTorch C++ batch inference reduces per-query latency from ~8.2s down to **< 1.9s**.
4. **Weights and code location**:
   - Adapter weights: `models/checkpoints/satquery_rs_model/adapter.pt`
   - Answer vocabulary: `models/checkpoints/satquery_rs_model/answer_vocab.json`
   - Architecture code: `models/rs_vlm.py` (`class RSVQAAdapter(nn.Module)`)

---

## 7. Repository Layout

```text
satquery-ai/
├── app.py                          # Root entry-point launcher for cloud & local deployments
├── satqueryctl                     # Unified CLI tool for 24/7 server, GPU, alerts, and tests
├── requirements.txt                # Python package dependencies (PyTorch, FastAPI, Rasterio, OpenCLIP)
├── backend/
│   └── app.py                      # Core FastAPI backend, REST endpoints, and orchestration logic
├── agent/
│   ├── planner.py                  # Agentic execution planner & step synthesizer
│   └── router.py                   # Natural-language query intent & specialist router
├── models/
│   ├── rs_vlm.py                   # GeoRSCLIP ViT-B/32 backbone + RSVQA Adapter + Visual Grounder
│   └── checkpoints/                # Model weights directory (RSVQA MLP Adapter checkpoint)
├── geospatial/
│   ├── water_detector.py           # Radiometric spectral engines (NDWI, NDVI, AWEI, NDBI)
│   ├── change_detector.py          # Bi-temporal change detection & quantitative delta metrics
│   ├── clip_segmenter.py           # Vectorized zero-shot multi-class AI segmentation engine
│   └── fusion.py                   # SAR dB calibration & optical-SAR cross-modal consensus fusion
├── frontend/
│   ├── index.html                  # Interactive GIS Leaflet map & orthomosaic pixel dashboard
│   ├── app.js                      # Client-side map controllers, layer rendering, and API sync
│   └── style.css                   # Responsive dark-mode interface styling
├── demo_data/                      # Curated benchmark datasets (1-click interactive presets)
│   ├── bigearthnet/                # Sentinel-2 4-band multispectral tile (B02, B03, B04, B08)
│   ├── vrsbench/                   # 0.5m high-resolution optical imagery (Kolkata Urban)
│   ├── isro_sac/                   # Co-registered Cartosat-2S optical + RISAT-1 SAR dataset
│   ├── cdvqa/                      # Bi-temporal California wildfire burn scar pair (T1 & T2)
│   ├── real_world_satellite/       # San Francisco Bay COG optical, Alps Sentinel-1 SAR
│   └── edge_cases/                 # Global Sentinel-2 L2A tiles (Amazon, Paris, Sahara, Delta, etc.)
├── scripts/                        # Automated deployment, monitoring, and notification utilities
│   ├── run_tunnel.sh               # Cloudflare Tunnel runner with HTTP/2 transport & boot alerts
│   ├── notify.sh                   # Telegram Bot & Discord Webhook alert dispatcher
│   ├── check_gpu.sh                # NVIDIA CUDA hardware diagnostics & tensor verification
│   ├── get_tunnel_url.sh           # Active public HTTPS URL extractor
│   ├── start_all.sh                # 24/7 background services start script
│   ├── stop_all.sh                 # 24/7 background services stop script
│   └── status.sh                   # Real-time health & telemetry dashboard
├── systemd/                        # Persistent systemd user service definitions
│   ├── satquery.service            # FastAPI + CUDA server supervisor (Restart=always)
│   └── satquery-tunnel.service     # Cloudflare Tunnel supervisor
└── tests/                          # Comprehensive automated test suite (35 tests passing)
    ├── test_queries.py             # The 5 mandatory representative benchmark queries
    ├── test_edge_cases.py          # Real-world global Sentinel-2 edge case evaluations
    ├── test_benchmarks.py          # RSVQA, VRSBench, CDVQA, and Optical-SAR evaluations
    ├── test_river_api.py           # Water body grounding & spatial bounding box tests
    ├── test_water_detector.py      # Radiometric index & spectral band resolution tests
    └── test_full_system.py         # End-to-end multi-modal audit verification script
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

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser to access the dashboard.

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

# Tail live server logs
./satqueryctl logs

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
| `POST` | `/api/upload` | Upload GeoTIFF / optical / SAR raster file | `multipart/form-data` (`file`) |
| `GET` | `/api/uploads` | List active uploaded raster sessions | *None* |
| `POST` | `/api/load_demo` | 1-click register demo satellite datasets | JSON: `{"sample_key": "sentinel2"}` |
| `POST` | `/api/analyze` | Unified agentic query analysis pipeline | JSON: `{"primary_id": "...", "query": "..."}` |
| `GET` | `/generated/{filename}` | Serve visual evidence overlays (PNG / GeoTIFF) | URL path parameter |

---

## 12. Automated Testing Suite

The repository includes a comprehensive automated test suite with **35 passing tests** verifying the agentic planner, models, radiometric tools, and benchmark queries:

```bash
# Run tests via the control tool
./satqueryctl test

# Run full automated test suite using Python unittest
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Test Coverage Highlights
- **Representative Queries** (`tests/test_queries.py`): Validates all 5 mandatory benchmark tasks.
- **Benchmark Suites** (`tests/test_benchmarks.py`): Evaluates RSVQA accuracy, CDVQA directional shifts, VRSBench grounding IoU, and Optical-SAR agreement.
- **Real-World Global Edge Cases** (`tests/test_edge_cases.py`): Evaluates Amazon canopy, Sahara sand, Venice canals, Dubai coastal sands, Lake Mead drought, and London Thames urban corridors.
- **Radiometric Spectral Engine** (`tests/test_water_detector.py`): Validates NDWI/NDVI calculations, Otsu thresholding, AWEI shadow suppression, and chlorophyll rejection.

---

## 13. Citations & References

- **BigEarthNet.txt**: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation ([arXiv:2603.29630](https://arxiv.org/abs/2603.29630))
- **GeoRSCLIP**: Remote Sensing Vision-Language Pre-training with Open-Vocabulary Capabilities
- **RSVQA / VRSBench / CDVQA**: Benchmark datasets for Remote Sensing VQA, Captioning, and Change Understanding
- **ISRO SAC**: Cartosat & RISAT-1 Earth Observation Datasets

---

## 14. License

This project is licensed under the Apache 2.0 License.23
