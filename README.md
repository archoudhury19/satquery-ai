# SatQuery AI: Agentic Vision-Language Remote-Sensing Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online%2024%2F7-success?style=for-the-badge&logo=cloudflare)](https://rejected-booking-nuts-silly.trycloudflare.com)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA%20Accelerated-76b900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![GeoRSCLIP](https://img.shields.io/badge/Model-GeoRSCLIP_ViT--B/32-orange.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-All%20Passing-brightgreen.svg)]()

> 🛰️ **Try SatQuery AI Live (24/7 GPU Accelerated Server)**:  
> 👉 **[https://rejected-booking-nuts-silly.trycloudflare.com](https://rejected-booking-nuts-silly.trycloudflare.com)**  
> *(Includes 1-click demo presets for Sentinel-2, Kolkata Urban, Cartosat+RISAT, and Wildfire Change Analysis)*

**SatQuery AI** is an agentic, query-driven vision-language platform designed for advanced Earth Observation (EO) and remote-sensing image understanding. It seamlessly orchestrates a domain-adapted vision-language model (`GeoRSCLIP` + `RSVQA` adapter) alongside modular geospatial spectral and radar engines to execute complex analytical workflows across single optical/SAR images, bi-temporal change pairs, and co-registered cross-modal optical–SAR datasets.

---

## 1. What SatQuery AI Is

SatQuery AI provides an evidence-grounded remote-sensing assistant that accepts natural-language queries and satellite imagery, dynamically synthesizes an execution plan, selects specialist vision-language models or radiometric tools, and produces auditable textual answers, spatial overlays, bounding boxes, geographic centroids (WGS84), and downloadable JSON audit reports.

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

## 3. Repository & Directory Structure

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
└── tests/                          # Comprehensive automated test suite (29 tests passing)
    ├── test_queries.py             # The 5 mandatory representative benchmark queries
    ├── test_edge_cases.py          # Real-world global Sentinel-2 edge case evaluations
    ├── test_benchmarks.py          # RSVQA, VRSBench, CDVQA, and Optical-SAR evaluations
    ├── test_river_api.py           # Water body grounding & spatial bounding box tests
    ├── test_water_detector.py      # Radiometric index & spectral band resolution tests
    └── test_full_system.py         # End-to-end multi-modal audit verification script
```

---

## 4. Supported Inputs

| Input Configuration | Modality / Sensor | Supported File Formats | Use Cases |
|:---|:---|:---|:---|
| **Single Image** | Optical, Multispectral (Sentinel-2, Cartosat), or SAR (Sentinel-1, RISAT) | GeoTIFF (`.tif`), TIFF, PNG, JPEG | VQA, Scene Captioning, Object/Region Grounding, Multi-Class Land-Cover Segmentation |
| **Cross-Modal Pair** | Co-registered Optical/Multispectral + SAR (e.g. Cartosat-2S + RISAT) | GeoTIFF (`.tif`), TIFF | Joint Information Extraction, Water & Urban Extraction under cloud cover / shadow |
| **Bi-Temporal Pair** | Two spatially corresponding images acquired at $T_1$ and $T_2$ | GeoTIFF (`.tif`), TIFF | Land-Cover Change Detection, Area Shift ($\Delta\%$, ha), Directional Change VQA |

---

## 5. Supported Queries & Specialist Tasks

1. **Visual Question Answering (VQA)**: Single-image remote-sensing questions (e.g. land-cover presence, urban vs. rural classification).
2. **Scene Captioning & Description**: Comprehensive land-cover summarization adhering to the VRSBench standard.
3. **Text-Guided Region Grounding**: Visual localization of queries into bounding boxes, spatial centroids, and pixel overlays.
4. **Bi-Temporal Change Analysis**: Quantitative $\Delta\%$ and hectare area shifts with dual-color difference overlays.
5. **Directional Change VQA**: Answering whether a land-cover class has *increased*, *decreased*, or *remained unchanged*.
6. **Cross-Modal Optical + SAR Joint Analysis**: Fused extraction leveraging optical spectral signatures and SAR dielectric double-bounce / specular properties.
7. **Dense Multi-Class AI Segmentation**: Fast zero-shot segmentation with spectral physics overrides.

---

## 6. Models & Specialists Registry

| Specialist Name | Component Type | Implementation Location | Purpose |
|:---|:---|:---|:---|
| `rs_vlm` | Domain-Adapted VLM | `models/rs_vlm.py` | GeoRSCLIP ViT-B/32 backbone with 50-class RSVQA MLP adapter |
| `rs_captioner` | Scene Descriptor | `models/rs_vlm.py` | Multi-feature scene summarization (VRSBench standard) |
| `rs_grounding` | Region Localizer | `models/rs_vlm.py` | Open-vocabulary spatial patch projection with bounding boxes |
| `clip_segmenter`| AI Segmenter | `geospatial/clip_segmenter.py` | Vectorized PyTorch batch tensor zero-shot land-cover segmenter |
| `change_engine` | Temporal Analyzer | `geospatial/change_detector.py`| Bi-temporal spectral subtraction and quantitative delta evaluator |
| `optical_sar_fusion`| Multimodal Fusion | `geospatial/fusion.py` | SAR dB calibration, spatial reprojection, and consensus fusion |
| `geospatial_tools`| Radiometric Indices | `geospatial/water_detector.py` | Deterministic 16-bit multi-spectral indices (NDWI, NDVI, NDBI, AWEI) |

---

## 7. Remote-Sensing Adaptation Details

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

## 8. Installation

```bash
# Clone the repository
git clone https://github.com/archoudhury19/satquery-ai.git
cd satquery-ai

# Activate Python virtual environment (Python 3.11 recommended)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Starting the Web Dashboard

```bash
# Start FastAPI backend with live reload
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern browser to access the interface.

---

## 10. 24/7 Server & Cloudflare Tunnel Operations

SatQuery AI is configured for continuous 24/7 operation with automatic process supervision, crash recovery, NVIDIA CUDA GPU acceleration, and public HTTPS exposure via Cloudflare Tunnel.

### Unified Control CLI (`satqueryctl`)

You can manage all 24/7 services and run diagnostics using the unified `./satqueryctl` CLI:

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

The 24/7 deployment relies on two persistent systemd user services:

| Service Unit | Purpose | Configuration File |
|:---|:---|:---|
| `satquery.service` | Supervises the FastAPI + Uvicorn server on port 8000 with CUDA GPU support, auto-restart on crash, and journal logging. | `~/.config/systemd/user/satquery.service` |
| `satquery-tunnel.service` | Supervises `cloudflared` to expose the local server over a secure, globally accessible Cloudflare HTTPS tunnel. | `~/.config/systemd/user/satquery-tunnel.service` |

### NVIDIA CUDA GPU Acceleration

SatQuery AI automatically offloads vision-language embeddings and neural inference to NVIDIA GPUs:
- **Detected Hardware**: NVIDIA GeForce MX450 (Compute Capability 7.5, CUDA 12.4 / 13.3)
- **Framework**: PyTorch 2.6.0+cu124 with CUDA tensor acceleration
- **Memory Footprint**: ~694 MB VRAM allocated for GeoRSCLIP ViT-B/32 backbone and RSVQA MLP adapter

---

## 11. Running Automated Tests

```bash
# Run tests via the control tool
./satqueryctl test

# Run full automated test suite using Python
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 12. The 5 Mandatory Representative Queries

| Task | Exact Query | Sample Input | Expected Output |
|:---|:---|:---|:---|
| **Captioning** | *"Describe the land-cover and major objects visible in this image."* | Kolkata Urban (`vrsbench_sample_01.tif`) | Structured description of built-up fabric (80.9%), river channel, and tree canopy. |
| **Grounding** | *"Highlight the water body referred to in the query."* | Kolkata Urban (`vrsbench_sample_01.tif`) | Bounding box `[0, 0, 240, 349]`, WGS84 centroid, visual river overlay. |
| **Bi-Temporal** | *"What changed between these two dates, and where did the change occur?"* | Sentinel-2 T1/T2 (`cdvqa_time1.tif`, `cdvqa_time2.tif`)| Quantitative area shifts ($\Delta = 30.6\%$, 27.2 ha altered), dual-color change map. |
| **Optical + SAR** | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | Cartosat + RISAT (`cartosat_optical_coregistered.tif`, `risat_sar_coregistered.tif`) | Fused consensus mask combining optical spectral reflection and SAR backscatter. |
| **Change VQA** | *"Has the built-up area increased, decreased, or remained unchanged?"* | Sentinel-2 T1/T2 (`cdvqa_time1.tif`, `cdvqa_time2.tif`)| Directional shift output (`remained approximately stable`, $0.0\% \to 0.0\%$). |

---

## 13. Outputs & Visual Evidence

Every analysis query returns:
- **Textual Answer**: Domain-specific answer with calibrated confidence percentage.
- **Visual Evidence Overlay**: Color-coded overlay image (`/generated/*.png`) displaying segmented masks, bounding boxes, or change maps.
- **Spatial Metadata Card**: Bounding box coordinates, pixel area, and WGS84 latitude/longitude centroids when georeferenced.
- **Observable Execution Trace**: Chronological log of query interpretation, input validation, tool selection, and execution parameters (no internal chain-of-thought clutter).
- **Downloadable JSON Report**: Standardized machine-readable audit report.

---

## 14. Known Limitations

- **Co-Registration Quality**: Cross-modal fusion assumes images are reasonably aligned; severely misaligned pairs require prior ground control point (GCP) orthorectification.
- **Cloud Cover in Optical Bands**: Dense cloud cover can obscure optical spectral indices; SAR backscatter thresholding is used as a fallback for water and urban structure.
- **CPU vs. GPU Inference**: The platform is fully optimized for CPU execution (~1.9s per image); deploying on CUDA GPU further accelerates batch patch projection.

---

## 15. Citations & References
- **BigEarthNet.txt**: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation ([arXiv:2603.29630](https://arxiv.org/abs/2603.29630))
- **GeoRSCLIP**: Remote Sensing Vision-Language Pre-training with Open-Vocabulary Capabilities
- **RSVQA / VRSBench / CDVQA**: Benchmark datasets for Remote Sensing VQA, Captioning, and Change Understanding.
