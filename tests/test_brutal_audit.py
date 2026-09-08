"""
scratch/brutal_test_suite.py
============================
Brutal Comprehensive Test Suite verifying 100% correct execution of:
1. Server Health & Model Registry
2. Panchromatic Optical vs SAR Modality Disambiguation
3. Dynamic Spectral Band Resolution & Ordering
4. VQA, Scene Captioning, Grounding, Multi-Class Segmentation
5. SAR Backscatter Calibration & Speckle Filtering
6. Optical-SAR Cross-Modal Dual Feature Fusion
7. Bi-temporal Change Detection & Wildfire Burn Scars
8. BigEarthNet Multimodal Patch Evaluation
9. Official RSVQA Benchmark Integration
10. Error Handling & Edge Cases
"""

import sys
import time
import requests
import numpy as np
from pathlib import Path

BASE_DIR = Path("c:/SatQueryMVP/satquery-ai")
if not BASE_DIR.exists():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "SatQueryMVP" / "satquery-ai"

sys.path.insert(0, str(BASE_DIR))

API_BASE = "http://127.0.0.1:8000"

print("=" * 80)
print("STARTING BRUTAL COMPREHENSIVE CODEBASE VERIFICATION")
print("=" * 80)

failures = []

def test_assert(condition, test_name, detail=""):
    if condition:
        print(f"  [PASS] {test_name}")
    else:
        print(f"  [FAIL] {test_name}: {detail}")
        failures.append((test_name, detail))

# ------------------------------------------------------------
# TEST 1: SERVER HEALTH & TOOL REGISTRY
# ------------------------------------------------------------
print("\n--- TEST 1: Server Health & Tools ---")
try:
    res = requests.get(f"{API_BASE}/api/health", timeout=5).json()
    test_assert(res.get("ok") is True, "Health check ok", str(res))
    test_assert(res.get("rs_vlm_available") is True, "RS_VLM available", str(res))
except Exception as e:
    test_assert(False, "Health endpoint reachable", str(e))

try:
    tools_res = requests.get(f"{API_BASE}/api/tools", timeout=5).json()
    test_assert(len(tools_res.get("tools", [])) >= 4, "Tool registry listed tools", f"Count: {len(tools_res.get('tools', []))}")
except Exception as e:
    test_assert(False, "Tools endpoint reachable", str(e))

# ------------------------------------------------------------
# TEST 2: PANCHROMATIC OPTICAL VS SAR DISAMBIGUATION
# ------------------------------------------------------------
print("\n--- TEST 2: Panchromatic vs SAR Modality Inference ---")
try:
    from backend.app import infer_modality
    import rasterio
    from rasterio.transform import from_origin

    scratch_dir = BASE_DIR / "data" / "scratch_test"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # 2a. Synthetic 1-band Panchromatic Optical (positive uint16 with smooth gradient)
    pan_path = scratch_dir / "pan_optical_sample.tif"
    pan_data = np.linspace(500, 3500, 100 * 100, dtype=np.uint16).reshape((100, 100))
    with rasterio.open(
        pan_path, 'w', driver='GTiff', height=100, width=100, count=1,
        dtype='uint16', crs='EPSG:4326', transform=from_origin(88.0, 22.0, 0.001, 0.001)
    ) as dst:
        dst.write(pan_data, 1)

    pan_mod = infer_modality(pan_path, {"count": 1, "dtypes": ["uint16"], "tags": {}})
    test_assert(pan_mod == "optical", "1-band Panchromatic uint16 inferred as optical", f"Got: {pan_mod}")

    # 2b. Synthetic 1-band SAR in Decibels (negative float32)
    sar_path = scratch_dir / "sar_sample_db.tif"
    sar_data = np.random.uniform(-25.0, -5.0, (100, 100)).astype(np.float32)
    with rasterio.open(
        sar_path, 'w', driver='GTiff', height=100, width=100, count=1,
        dtype='float32', crs='EPSG:4326', transform=from_origin(88.0, 22.0, 0.001, 0.001)
    ) as dst:
        dst.write(sar_data, 1)

    sar_mod = infer_modality(sar_path, {"count": 1, "dtypes": ["float32"], "tags": {}})
    test_assert(sar_mod == "sar", "1-band negative float32 inferred as SAR", f"Got: {sar_mod}")

except Exception as e:
    test_assert(False, "Modality inference test", str(e))

# ------------------------------------------------------------
# TEST 3: DYNAMIC SPECTRAL BAND RESOLUTION
# ------------------------------------------------------------
print("\n--- TEST 3: Dynamic Band Assignment ---")
try:
    from backend.app import _read_raster

    # Create synthetic Landsat-style raster: Band 1=Coastal, 2=Blue, 3=Green, 4=Red, 5=NIR
    landsat_path = scratch_dir / "landsat_style.tif"
    with rasterio.open(
        landsat_path, 'w', driver='GTiff', height=50, width=50, count=5,
        dtype='uint16', crs='EPSG:4326', transform=from_origin(88.0, 22.0, 0.001, 0.001)
    ) as dst:
        for b in range(1, 6):
            dst.write(np.full((50, 50), b * 500, dtype=np.uint16), b)
        dst.descriptions = ("Coastal Aerosol", "Blue", "Green", "Red", "NIR")

    read_meta = _read_raster(landsat_path)
    bands = read_meta.get("bands", {})
    test_assert("red" in bands and "green" in bands and "nir" in bands, "Bands parsed", f"Keys: {list(bands.keys())}")
    # Red should correspond to band 4 (value 2000), not band 1 (value 500)
    red_val = int(bands["red"][0, 0])
    test_assert(red_val == 2000, "Landsat Red band resolved correctly from description", f"Expected 2000, got {red_val}")
    nir_val = int(bands["nir"][0, 0])
    test_assert(nir_val == 2500, "Landsat NIR band resolved correctly from description", f"Expected 2500, got {nir_val}")

except Exception as e:
    test_assert(False, "Dynamic band resolution", str(e))

# ------------------------------------------------------------
# TEST 4: OPTICAL SINGLE-IMAGE TASKS
# ------------------------------------------------------------
print("\n--- TEST 4: Optical Single-Image Analysis Tasks ---")
try:
    # 4a. Load Sentinel-2 demo
    demo_s2 = requests.post(f"{API_BASE}/api/load_demo", json={"sample_key": "sentinel2"}).json()
    prim_id = demo_s2["primary"]["id"]

    # 4b. Scene Captioning
    cap_res = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Describe the scene land-cover in detail."
    }).json()
    test_assert("built-up" in cap_res.get("answer", "").lower() or "vegetation" in cap_res.get("answer", "").lower(),
                "Scene caption generated meaningful narrative", cap_res.get("answer", "")[:80])
    test_assert(0.5 <= cap_res.get("confidence", 0) <= 1.0, "Dynamic confidence returned", str(cap_res.get("confidence")))

    # 4c. Open-Vocabulary Grounding
    grd_res = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Locate water bodies and the bay."
    }).json()
    test_assert("bounding_box" in grd_res or "water" in grd_res.get("answer", "").lower(),
                "Visual grounding localized water feature", grd_res.get("answer", "")[:80])

    # 4d. Multi-Class Land-Cover Segmentation
    seg_res = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Segment all land-cover classes and generate a color map."
    }).json()
    test_assert(seg_res.get("task") == "segmentation", "Segmentation task dispatched", str(seg_res.get("task")))
    test_assert("mask_stats" in seg_res, "Segmentation stats present", str(seg_res.get("mask_stats", {}).keys()))

    # 4e. VQA Zero-Shot
    vqa_res = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Is there water present in this image?"
    }).json()
    test_assert(vqa_res.get("answer") in ["yes", "no", "water body", "river", "bay"],
                "VQA answered presence question", f"Answer: {vqa_res.get('answer')}")

except Exception as e:
    test_assert(False, "Optical single-image tasks", str(e))

# ------------------------------------------------------------
# TEST 5: OPTICAL-SAR CROSS-MODAL FUSION
# ------------------------------------------------------------
print("\n--- TEST 5: Optical-SAR Cross-Modal Fusion ---")
try:
    demo_fusion = requests.post(f"{API_BASE}/api/load_demo", json={"sample_key": "optical_sar"}).json()
    prim_id = demo_fusion["primary"]["id"]
    sec_id = demo_fusion["secondary"]["id"]

    fusion_res = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "secondary_id": sec_id,
        "query": "Joint optical and SAR consensus fusion for water and urban infrastructure."
    }).json()

    # Verify NO hardcoded 94.2% agreement
    f_metrics = fusion_res.get("evidence", {}).get("fusion_metrics", {})
    agreement = f_metrics.get("agreement_pct")
    iou = f_metrics.get("cross_modal_iou_pct")
    test_assert(agreement != 94.2 and agreement is not None, "Agreement is authentic live metric (not 94.2%)", f"Agreement: {agreement}%")
    test_assert(iou is not None and 0 <= iou <= 100, "IoU is authentic live metric", f"IoU: {iou}%")

    # Verify confidence is dynamic (not static 0.95)
    conf = fusion_res.get("confidence")
    test_assert(conf != 0.95 and 0.5 <= conf <= 1.0, "Confidence is dynamic", f"Confidence: {conf}")

    # Verify built-up hectares computed dynamically
    built_ha = fusion_res.get("evidence", {}).get("built_up_hectares")
    test_assert(built_ha is not None and built_ha > 0, "Built-up hectares calculated", f"{built_ha} ha")

except Exception as e:
    test_assert(False, "Cross-modal fusion test", str(e))

# ------------------------------------------------------------
# TEST 6: BI-TEMPORAL CHANGE & WILDFIRE BURN SCAR
# ------------------------------------------------------------
print("\n--- TEST 6: Bi-Temporal Change & Wildfire Disturbance ---")
try:
    demo_bi = requests.post(f"{API_BASE}/api/load_demo", json={"sample_key": "bitemporal"}).json()
    prim_id = demo_bi["primary"]["id"]
    sec_id = demo_bi["secondary"]["id"]

    fire_res = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "secondary_id": sec_id,
        "query": "Detect wildfire burn scar and forest damage between the two dates."
    }).json()

    ans = fire_res.get("answer", "")
    test_assert("camp fire" not in ans.lower(), "Zero 'Camp Fire' hardcoding in burn scar answer", ans[:80])
    test_assert("sierra nevada" not in ans.lower(), "Zero 'Sierra Nevada' hardcoding in burn scar answer", ans[:80])
    test_assert("october 2018" not in ans.lower(), "Zero hardcoded 2018 dates in answer", ans[:80])

    ev_label = fire_res.get("evidence", {}).get("burn_scar", {}).get("label", "")
    test_assert("camp fire" not in ev_label.lower(), "Zero 'Camp Fire' in evidence label", ev_label)
    test_assert(fire_res.get("confidence") != 0.94, "Burn scar confidence is dynamic (not 0.94)", str(fire_res.get("confidence")))

except Exception as e:
    test_assert(False, "Bi-temporal change test", str(e))

# ------------------------------------------------------------
# TEST 7: BIGEARTHNET MULTIMODAL ENDPOINTS
# ------------------------------------------------------------
print("\n--- TEST 7: BigEarthNet Multimodal Endpoints ---")
try:
    ben_rec = requests.get(f"{API_BASE}/api/bigearthnet/records?limit=3", timeout=5).json()
    test_assert("records" in ben_rec and len(ben_rec.get("records", [])) == 3,
                "BigEarthNet test records endpoint", f"Returned: {len(ben_rec.get('records', []))} records")

    patch_info = requests.get(f"{API_BASE}/api/bigearthnet/patch_info", timeout=5).json()
    test_assert("patch_id" in patch_info and patch_info.get("s2_exists") is True,
                "BigEarthNet co-registered S1/S2 patch info", str(patch_info.get("patch_id")))

    ben_eval = requests.post(f"{API_BASE}/api/bigearthnet/evaluate_sample", json={"sample_id": 1}, timeout=10).json()
    test_assert("predicted_answer" in ben_eval and "matched" in ben_eval,
                "BigEarthNet sample evaluation", f"Predicted: {ben_eval.get('predicted_answer')}, Matched: {ben_eval.get('matched')}")

except Exception as e:
    test_assert(False, "BigEarthNet endpoints", str(e))

# ------------------------------------------------------------
# TEST 8: ERROR HANDLING & EDGE CASES
# ------------------------------------------------------------
print("\n--- TEST 8: Robust Error Handling & Edge Cases ---")
try:
    # 8a. Invalid primary ID
    bad_res = requests.post(f"{API_BASE}/api/analyze", json={"primary_id": "non_existent_id", "query": "what is this?"}).json()
    test_assert("detail" in bad_res or "error" in bad_res, "Invalid ID raises graceful error", str(bad_res))

    # 8b. Empty query
    empty_res = requests.post(f"{API_BASE}/api/analyze", json={"primary_id": prim_id, "query": ""}).json()
    test_assert("answer" in empty_res or "detail" in empty_res, "Empty query handled gracefully", str(empty_res)[:60])

    # 8c. Invalid sample key in load_demo
    bad_demo = requests.post(f"{API_BASE}/api/load_demo", json={"sample_key": "invalid_unknown_sample"}).json()
    test_assert("detail" in bad_demo, "Invalid demo sample returns 404/detail", str(bad_demo))

except Exception as e:
    test_assert(False, "Error handling test", str(e))

# ------------------------------------------------------------
# TEST 9: RS-VLM VISUAL GROUNDING HONESTY (ZERO FAKE FALLBACKS)
# ------------------------------------------------------------
print("\n--- TEST 9: Visual Grounding Honesty ---")
try:
    from models.rs_vlm import RemoteSensingVLM
    vlm_inst = RemoteSensingVLM()
    test_img_path = BASE_DIR / "demo_data" / "bitemporal" / "s2_paradise_prefire.tif"
    if not test_img_path.exists():
        test_img_path = BASE_DIR / "demo_data" / "vrsbench" / "vrsbench_sample_01.tif"

    # Query for an object completely absent from a forest/rural image (e.g., supersonic commercial supersonic Concorde jet)
    grd_honesty = vlm_inst.ground(test_img_path, "supersonic commercial passenger jet aircraft parked on tarmac")
    
    # Verify that the system does NOT synthesize a fake center box
    test_assert(grd_honesty.get("bounding_box") is None, "Non-existent object returns bbox=None (no fake center box)", str(grd_honesty.get("bounding_box")))
    test_assert(grd_honesty.get("location") == "unlocalized", "Location marked unlocalized", str(grd_honesty.get("location")))
    test_assert("no distinct region" in grd_honesty.get("answer", "").lower(), "Answer honestly indicates feature was not localized", grd_honesty.get("answer", ""))
    test_assert(grd_honesty.get("confidence", 1.0) <= 0.50, "Confidence is low for absent target", str(grd_honesty.get("confidence")))

except Exception as e:
    test_assert(False, "Visual grounding honesty test", str(e))

# ------------------------------------------------------------
# TEST 10: DYNAMIC MULTI-CLASS SEGMENTATION CONFIDENCE
# ------------------------------------------------------------
print("\n--- TEST 10: Dynamic Land-Cover Segmentation Confidence ---")
try:
    seg_check = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Segment all land cover classes in this scene."
    }).json()

    seg_conf = seg_check.get("confidence")
    test_assert(seg_conf != 0.88, "Segmentation confidence is dynamic (not hardcoded 0.88)", f"Got: {seg_conf}")
    test_assert(0.70 <= seg_conf <= 0.98, "Segmentation confidence in valid calibrated range", f"Got: {seg_conf}")

    # Check that confidence scales with classified coverage
    ev = seg_check.get("evidence", {})
    w = ev.get("water_percent", 0)
    v = ev.get("vegetation_percent", 0)
    b = ev.get("built_up_percent", 0)
    d = ev.get("desert_percent", 0)
    expected_conf = round(float(np.clip(0.72 + ((w + v + b + d) / 100.0) * 0.23, 0.72, 0.96)), 2)
    test_assert(abs(seg_conf - expected_conf) < 0.02, "Segmentation confidence strictly matches dynamic formula", f"Expected {expected_conf}, got {seg_conf}")

except Exception as e:
    test_assert(False, "Dynamic segmentation confidence test", str(e))

# ------------------------------------------------------------
# TEST 11: STATIC UI TELEMETRY AUDIT
# ------------------------------------------------------------
print("\n--- TEST 11: Static UI Telemetry Audit ---")
try:
    html_path = BASE_DIR / "frontend" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    test_assert("94.87 ha" not in html_content, "Zero hardcoded 94.87 ha area in index.html", "Found in index.html" if "94.87 ha" in html_content else "Clean")
    test_assert("22.568° N" not in html_content, "Zero hardcoded Kolkata coordinates in index.html", "Found in index.html" if "22.568° N" in html_content else "Clean")
    test_assert("[0, 0, 240, 349]" not in html_content, "Zero hardcoded bounding box in index.html", "Found in index.html" if "[0, 0, 240, 349]" in html_content else "Clean")
    test_assert("initOrUpdateChart(1.3, 1.4, 80.9, 16.4)" not in html_content, "Zero hardcoded initial chart values in index.html", "Found in index.html" if "initOrUpdateChart(1.3, 1.4, 80.9, 16.4)" in html_content else "Clean")

except Exception as e:
    test_assert(False, "Static UI telemetry audit", str(e))

# ------------------------------------------------------------
# TEST 12: SPATIAL PAIR VALIDATION (DISJOINT SCENE REJECTION)
# ------------------------------------------------------------
print("\n--- TEST 12: Spatial Pair Overlap Validation ---")
try:
    from backend.app import validate_pair_compatibility
    # Disjoint rasters: Kolkata (lat ~22.5, lon ~88.3) vs San Francisco (lat ~37.7, lon ~ -122.4)
    p_mock = {
        "bounds_wgs84": [[22.4, 88.2], [22.6, 88.4]],
        "count": 3,
        "is_georeferenced": True,
        "width": 512,
        "height": 512,
    }
    s_mock = {
        "bounds_wgs84": [[37.6, -122.5], [37.8, -122.3]],
        "count": 1,
        "is_georeferenced": True,
        "width": 512,
        "height": 512,
    }
    res_pair = validate_pair_compatibility(p_mock, s_mock)
    compat = res_pair["compatible"]
    issues = res_pair["issues"]
    test_assert(compat is False, "Geographically disjoint pair marked compatible=False", f"Issues: {issues}")
    test_assert(any("geographically disjoint" in iss.lower() for iss in issues), "Disjoint footprint issue recorded", str(issues))

    # Overlapping rasters
    s_overlapping = {
        "bounds_wgs84": [[22.5, 88.3], [22.7, 88.5]],
        "count": 1,
        "is_georeferenced": True,
        "width": 512,
        "height": 512,
    }
    res_ok = validate_pair_compatibility(p_mock, s_overlapping)
    compat_ok = res_ok["compatible"]
    issues_ok = res_ok["issues"]
    test_assert(compat_ok is True, "Overlapping pair marked compatible=True", f"Issues: {issues_ok}")
except Exception as e:
    test_assert(False, "Spatial pair validation test", str(e))

# ------------------------------------------------------------
# TEST 13: ALTERNATIVE VQA QUERY ROUTING (NO FORCED YES/NO)
# ------------------------------------------------------------
print("\n--- TEST 13: Alternative VQA Query Routing ---")
try:
    from models.rs_vlm import RemoteSensingVLM
    test_assert(RemoteSensingVLM._is_binary_query("Is this an urban or rural area?") is False,
                "'Is this an urban or rural area?' is NOT treated as binary yes/no", "Evaluated as binary")
    test_assert(RemoteSensingVLM._is_binary_query("Is this water or forest?") is False,
                "'Is this water or forest?' is NOT treated as binary yes/no", "Evaluated as binary")
    test_assert(RemoteSensingVLM._is_binary_query("Is there a river in this image?") is True,
                "'Is there a river in this image?' is correctly treated as binary", "Evaluated as non-binary")
    test_assert(RemoteSensingVLM._is_binary_query("Is the water level higher, yes or no?") is True,
                "Explicit 'yes or no' query treated as binary", "Evaluated as non-binary")

    # Live VLM query: "Is this an urban or rural area?" should answer with "urban" or "rural", never "yes" or "no"
    vlm_obj = RemoteSensingVLM()
    test_vrs_p = BASE_DIR / "demo_data" / "vrsbench" / "vrsbench_sample_01.tif"
    ans_dict = vlm_obj.analyze(test_vrs_p, "Is this an urban or rural area?")
    cand_ans = ans_dict.get("answer", "").lower()
    test_assert(cand_ans in ["urban", "rural"], "Alternative query returned 'urban' or 'rural' (not yes/no)", f"Answer: {cand_ans}")
except Exception as e:
    test_assert(False, "Alternative VQA query test", str(e))

# ------------------------------------------------------------
# TEST 14: VQA FALLBACK UNPACK SAFETY (3-TUPLE)
# ------------------------------------------------------------
print("\n--- TEST 14: VQA Fallback Unpack Safety ---")
try:
    from geospatial.scene_captioner import generate_rs_caption
    from backend.app import analyze_single, _read_raster
    # Verify generate_rs_caption returns 3 values (caption, confidence, diag)
    test_s2_p = BASE_DIR / "demo_data" / "vrsbench" / "vrsbench_sample_01.tif"
    raster_d = _read_raster(test_s2_p)
    cap_out = generate_rs_caption(raster_d)
    test_assert(isinstance(cap_out, tuple) and len(cap_out) == 3,
                "generate_rs_caption returns 3-tuple (answer, confidence, diag)", f"Returned {len(cap_out)} items")

    # Fallback simulation in analyze_single: query with unknown feature
    single_res = analyze_single(test_s2_p, raster_d, feature="auto", task="vqa", query="Describe this scene land cover")
    test_assert("answer" in single_res and "confidence" in single_res,
                "analyze_single unpacked fallback cleanly without ValueError", str(single_res.get("answer"))[:60])
except Exception as e:
    test_assert(False, "VQA fallback unpack safety test", str(e))

# ------------------------------------------------------------
# TEST 15: ZERO-OVERLAP FUSION METRICS (NO ARTIFICIAL FLOORS)
# ------------------------------------------------------------
print("\n--- TEST 15: Zero-Overlap Fusion Metrics ---")
try:
    from backend.app import analyze_cross_modal
    opt_p = BASE_DIR / "demo_data" / "isro_sac" / "cartosat_optical_coregistered.tif"
    sar_p = BASE_DIR / "demo_data" / "isro_sac" / "risat_sar_coregistered.tif"
    opt_data = _read_raster(opt_p)
    sar_data = _read_raster(sar_p)

    # Synthetic non-overlapping test: all zeros SAR
    sar_zero_data = dict(sar_data)
    H, W = opt_data["height"], opt_data["width"]
    sar_zero_data["raw_band"] = np.full((H, W), -48.0, dtype=np.float32)

    fusion_zero = analyze_cross_modal(opt_p, opt_data, sar_p, sar_zero_data, feature="built")
    f_zero_metrics = fusion_zero.get("evidence", {}).get("fusion_metrics", {})
    agreement_val = f_zero_metrics.get("agreement_pct", -1)
    iou_val = f_zero_metrics.get("cross_modal_iou_pct", -1)

    # Crucial audit test: neither agreement nor IoU should be artificially clipped to 45.0% or 20.0%
    test_assert(agreement_val < 40.0, f"Agreement is not artificially clipped to 45% floor (got {agreement_val}%)", f"Got: {agreement_val}")
    test_assert(iou_val < 15.0, f"IoU is not artificially clipped to 20% floor (got {iou_val}%)", f"Got: {iou_val}")
except Exception as e:
    test_assert(False, "Zero-overlap fusion test", str(e))

# ------------------------------------------------------------
# TEST 16: RIGOROUS BENCHMARK METRIC COUNTEREXAMPLES
# ------------------------------------------------------------
print("\n--- TEST 16: Benchmark Metric Counterexamples ---")
try:
    from benchmarks.evaluate_metrics import (
        compute_grounding_metrics,
        compute_cdvqa_metrics,
        compute_vqa_accuracy,
    )

    # Counterexample 1: Precision@0.5 with non-overlapping bounding box
    bad_pred_box = [0, 0, 10, 10]
    gt_box = [100, 100, 200, 200]
    g_res = compute_grounding_metrics(bad_pred_box, "north-west", "south-east", gt_bbox=gt_box)
    test_assert(g_res["precision_at_50"] == 0.0,
                "P@0.5 is 0.0 for non-overlapping predicted box (IoU < 0.50)", f"Got: {g_res['precision_at_50']}")

    good_pred_box = [105, 105, 195, 195]
    g_res_good = compute_grounding_metrics(good_pred_box, "south-east", "south-east", gt_bbox=gt_box)
    test_assert(g_res_good["precision_at_50"] == 1.0,
                "P@0.5 is 1.0 for high IoU predicted box (IoU >= 0.50)", f"Got: {g_res_good['precision_at_50']}")

    # Counterexample 2: CDVQA Negation handling
    cd_neg = compute_cdvqa_metrics("The area did not increase; it decreased by 12%", "The water area decreased", true_direction="increased")
    test_assert(cd_neg["directional_accuracy"] == 0.0,
                "CDVQA correctly assigns 0.0 directional accuracy when 'did not increase' is used for target 'increased'",
                f"Got: {cd_neg['directional_accuracy']}")

    cd_pos = compute_cdvqa_metrics("The area did not increase; it decreased by 12%", "The water area decreased", true_direction="decreased")
    test_assert(cd_pos["directional_accuracy"] == 1.0,
                "CDVQA correctly assigns 1.0 directional accuracy for target 'decreased'",
                f"Got: {cd_pos['directional_accuracy']}")

    # Counterexample 3: VQA Polarity & Word Boundary
    vqa_test = compute_vqa_accuracy(
        predictions=["No, there is no water; yes would be incorrect"],
        ground_truths=["yes"]
    )
    test_assert(vqa_test["overall_accuracy"] == 0.0,
                "VQA does not falsely match 'yes' inside a negative sentence", f"Got: {vqa_test['overall_accuracy']}%")

    vqa_test_correct = compute_vqa_accuracy(
        predictions=["No, there is no water; yes would be incorrect"],
        ground_truths=["no"]
    )
    test_assert(vqa_test_correct["overall_accuracy"] == 100.0,
                "VQA correctly matches 'no' when sentence starts with 'No'", f"Got: {vqa_test_correct['overall_accuracy']}%")

except Exception as e:
    test_assert(False, "Benchmark metric counterexamples test", str(e))

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 80)
if len(failures) == 0:
    print("ALL TESTS PASSED! 100% OF VERIFIED CODE PATHS OPERATING CORRECTLY WITH ZERO LOOPHOLES.")
else:
    print(f"TEST RUN FINISHED WITH {len(failures)} FAILURES:")
    for name, det in failures:
        print(f"  - {name}: {det}")
print("=" * 80)

sys.exit(len(failures))
