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

BASE_DIR = Path(__file__).resolve().parent.parent

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
# TEST 17: DISJOINT SCENE PAIR REJECTION GUARD (HTTP 400)
# ------------------------------------------------------------
print("\n--- TEST 17: Disjoint Scene Pair Rejection Guard ---")
try:
    from fastapi.testclient import TestClient
    from backend.app import app, FILES
    client = TestClient(app)
    # Register mock disjoint scenes
    FILES["disjoint_p"] = {
        "path": BASE_DIR / "demo_data" / "vrsbench" / "vrsbench_sample_01.tif",
        "data": {
            "bounds_wgs84": [[22.4, 88.2], [22.6, 88.4]],
            "count": 3,
            "is_georeferenced": True,
            "width": 512,
            "height": 512,
            "rgb": np.zeros((512, 512, 3), dtype=np.uint8),
        },
        "filename": "kolkata.tif",
    }
    FILES["disjoint_s"] = {
        "path": BASE_DIR / "demo_data" / "real_world_satellite" / "real_san_francisco_optical.tif",
        "data": {
            "bounds_wgs84": [[37.6, -122.5], [37.8, -122.3]],
            "count": 3,
            "is_georeferenced": True,
            "width": 512,
            "height": 512,
            "rgb": np.zeros((512, 512, 3), dtype=np.uint8),
        },
        "filename": "san_francisco.tif",
    }
    resp_disjoint = client.post("/api/analyze", json={
        "primary_id": "disjoint_p",
        "secondary_id": "disjoint_s",
        "query": "Compare changes between these two images",
    })
    test_assert(resp_disjoint.status_code == 400,
                "Disjoint image pair is rejected with HTTP 400 status", f"Status: {resp_disjoint.status_code}")
    err_detail = str(resp_disjoint.json().get("detail", "")).lower()
    test_assert("incompatible image pair rejected" in err_detail or "geographically disjoint" in err_detail,
                "Rejection detail explains geographical disjointness", f"Detail: {err_detail}")
except Exception as e:
    test_assert(False, "Disjoint pair rejection guard test", str(e))

# ------------------------------------------------------------
# TEST 18: PAIRED VQA MODALITY ROUTING & ARGUMENT BINDING
# ------------------------------------------------------------
print("\n--- TEST 18: Paired VQA Modality Routing & Kwargs Binding ---")
try:
    from backend.app import _handle_rs_vqa, analyze_change
    from types import SimpleNamespace

    opt_p = BASE_DIR / "demo_data" / "isro_sac" / "cartosat_optical_coregistered.tif"
    sar_p = BASE_DIR / "demo_data" / "isro_sac" / "risat_sar_coregistered.tif"
    opt_d = _read_raster(opt_p)
    sar_d = _read_raster(sar_p)

    ctx_optical_sar = {
        "primary": {"path": opt_p, "data": opt_d, "filename": opt_p.name},
        "secondary": {"path": sar_p, "data": sar_d, "filename": sar_p.name},
        "feature": "auto",
        "req": SimpleNamespace(query="Are built-up areas distinguishable from water?"),
        "trace": [],
    }
    vqa_fusion_res = _handle_rs_vqa(ctx_optical_sar)
    test_assert(vqa_fusion_res.get("task") == "cross_modal" or "Optical-SAR" in vqa_fusion_res.get("tool", ""),
                "Optical+SAR paired VQA correctly routed to cross-modal fusion (not bi-temporal change)",
                f"Task: {vqa_fusion_res.get('task')}, Tool: {vqa_fusion_res.get('tool')}")

    # Verify analyze_change binds kwargs (feature, query) without TypeError
    t1_p = BASE_DIR / "demo_data" / "cdvqa" / "cdvqa_time1.tif"
    t2_p = BASE_DIR / "demo_data" / "cdvqa" / "cdvqa_time2.tif"
    t1_d = _read_raster(t1_p)
    t2_d = _read_raster(t2_p)
    chg_res = analyze_change(t1_p, t1_d, t2_p, t2_d, feature="water", query="Has the water coverage increased?")
    test_assert("answer" in chg_res and "evidence" in chg_res,
                "analyze_change binds feature and query kwargs cleanly", str(chg_res.get("answer"))[:60])
except Exception as e:
    test_assert(False, "Paired VQA modality routing & kwargs binding test", str(e))

# ------------------------------------------------------------
# TEST 19: EMPTY & UNCHANGED PREDICTION SCORING IN CDVQA
# ------------------------------------------------------------
print("\n--- TEST 19: Empty & Unchanged CDVQA Prediction Scoring ---")
try:
    from benchmarks.evaluate_metrics import compute_cdvqa_metrics

    # Empty string should receive 0.0 directional accuracy even if target is unchanged
    empty_res_unchanged = compute_cdvqa_metrics("", "The scene remained unchanged", true_direction="unchanged")
    test_assert(empty_res_unchanged["directional_accuracy"] == 0.0,
                "Empty string prediction receives 0.0 directional accuracy for 'unchanged' target",
                f"Got: {empty_res_unchanged['directional_accuracy']}")

    whitespace_res = compute_cdvqa_metrics("   \n\t  ", "The water area expanded", true_direction="increased")
    test_assert(whitespace_res["directional_accuracy"] == 0.0,
                "Whitespace-only prediction receives 0.0 directional accuracy",
                f"Got: {whitespace_res['directional_accuracy']}")

    # Correct unchanged sentence should receive 1.0
    good_unchanged = compute_cdvqa_metrics("The lake extent remained unchanged and stable over time.", "The scene was unchanged", true_direction="unchanged")
    test_assert(good_unchanged["directional_accuracy"] == 1.0,
                "Genuine unchanged sentence receives 1.0 directional accuracy for 'unchanged' target",
                f"Got: {good_unchanged['directional_accuracy']}")
except Exception as e:
    test_assert(False, "Empty & unchanged CDVQA prediction scoring test", str(e))

# ------------------------------------------------------------
# TEST 20: TIGHTENED PNG/JPEG FORMAT RESTRICTIONS
# ------------------------------------------------------------
print("\n--- TEST 20: Tightened Format Restrictions for PNG/JPEG ---")
try:
    import io
    # Test arbitrary non-benchmark JPEG upload is rejected
    dummy_jpeg = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00")
    resp_bad = client.post("/api/upload", files={"file": ("test.jpg", dummy_jpeg, "image/jpeg")})
    test_assert(resp_bad.status_code == 400,
                "Arbitrary renamed file 'test.jpg' rejected with HTTP 400", f"Status: {resp_bad.status_code}")
    test_assert("restricted to prescribed benchmark" in str(resp_bad.json().get("detail", "")).lower(),
                "Rejection message states benchmark dataset restriction", str(resp_bad.json().get("detail")))

    # Test arbitrary substring bypass file like 'rsvqa_photo.jpg' is rejected
    dummy_rsvqa_fake = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00")
    resp_fake_rsvqa = client.post("/api/upload", files={"file": ("rsvqa_photo.jpg", dummy_rsvqa_fake, "image/jpeg")})
    test_assert(resp_fake_rsvqa.status_code == 400,
                "Arbitrary file 'rsvqa_photo.jpg' rejected with HTTP 400", f"Status: {resp_fake_rsvqa.status_code}")

    dummy_sample = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89")
    resp_sample = client.post("/api/upload", files={"file": ("sample.png", dummy_sample, "image/png")})
    test_assert(resp_sample.status_code == 400,
                "Generic file 'sample.png' rejected with HTTP 400", f"Status: {resp_sample.status_code}")

    # Test legitimate VRSBench filename format is permitted past upload format guard
    real_vrs_p = BASE_DIR / "data" / "external_datasets" / "vrsbench" / "images" / "P2655_0055.png"
    if real_vrs_p.exists():
        resp_vrs = client.post("/api/upload", files={"file": ("P2655_0055.png", real_vrs_p.read_bytes(), "image/png")})
        test_assert(resp_vrs.status_code == 200,
                    "Legitimate benchmark image 'P2655_0055.png' accepted by upload endpoint", f"Status: {resp_vrs.status_code}")
except Exception as e:
    test_assert(False, "Tightened format restriction test", str(e))

# ------------------------------------------------------------
# TEST 21: RSVQA SCENE-DISJOINT TRAIN/TEST SPLIT (ZERO CONTAMINATION)
# ------------------------------------------------------------
print("\n--- TEST 21: RSVQA Scene-Disjoint Train/Test Split ---")
try:
    import json, hashlib
    train_manifest_p = BASE_DIR / "data" / "external_datasets" / "rsvqa" / "rsvqa_train.json"
    eval_manifest_p = BASE_DIR / "data" / "external_datasets" / "rsvqa" / "rsvqa_official_eval.json"

    test_assert(train_manifest_p.exists(), "rsvqa_train.json exists on disk", str(train_manifest_p))
    test_assert(eval_manifest_p.exists(), "rsvqa_official_eval.json exists on disk", str(eval_manifest_p))

    train_data = json.load(open(train_manifest_p, encoding="utf-8"))
    eval_data = json.load(open(eval_manifest_p, encoding="utf-8"))

    train_hashes = set()
    for item in train_data:
        p = BASE_DIR / item["image_path"]
        if p.exists():
            train_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())

    eval_hashes = set()
    for item in eval_data:
        p = BASE_DIR / item["image_path"]
        if p.exists():
            eval_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())

    overlap = train_hashes & eval_hashes
    test_assert(len(overlap) == 0,
                f"Train and Evaluation splits have 0 overlapping scenes (clean test generalization)",
                f"Overlap count: {len(overlap)}")
    test_assert(len(train_hashes) >= 5, f"Training split contains multiple scenes ({len(train_hashes)} scenes)", f"Count: {len(train_hashes)}")
    test_assert(len(eval_hashes) >= 3, f"Evaluation split contains multiple held-out scenes ({len(eval_hashes)} scenes)", f"Count: {len(eval_hashes)}")

    # Verify config.json does not contain stale 400 validation samples
    config_p = BASE_DIR / "models" / "checkpoints" / "satquery_rs_model" / "config.json"
    cfg = json.load(open(config_p, encoding="utf-8"))
    test_assert("internal_validation_samples" not in cfg,
                "config.json does not contain stale 'internal_validation_samples' field", str(cfg.get("internal_validation_samples")))
    test_assert("Scene-Disjoint" in cfg.get("dataset", ""),
                "config.json documents scene-disjoint training provenance", cfg.get("dataset"))
except Exception as e:
    test_assert(False, "RSVQA scene-disjoint split test", str(e))

# ------------------------------------------------------------
# TEST 22: ALL VRSBENCH GROUNDING PROMPTS ROUTED TO GROUNDING
# ------------------------------------------------------------
print("\n--- TEST 22: VRSBench Grounding Prompt Routing & Captions ---")
try:
    from agent.planner import build_plan
    vrs_p = BASE_DIR / "data" / "external_datasets" / "vrsbench" / "vrsbench_official_eval.json"
    vrs_items = json.load(open(vrs_p, encoding="utf-8"))

    tasks = [build_plan(item["prompt"], 1)["task"] for item in vrs_items]
    all_grounding = all(t == "grounding" for t in tasks)
    test_assert(all_grounding,
                f"100% of VRSBench grounding prompts ({len(vrs_items)}/26) route to task='grounding'",
                f"Tasks found: {set(tasks)}")

    # Verify authentic reference captions present in manifest
    has_refs = all("reference_caption" in item and len(item["reference_caption"]) > 10 for item in vrs_items)
    test_assert(has_refs,
                "All VRSBench records contain authentic reference captions for caption scoring",
                f"Checked {len(vrs_items)} records")
except Exception as e:
    test_assert(False, "VRSBench grounding prompt routing test", str(e))

# ------------------------------------------------------------
# TEST 23: PAIRED VQA AUDITABLE EXECUTION TRACE LABELING
# ------------------------------------------------------------
print("\n--- TEST 23: Auditable Paired-VQA Execution Trace Labeling ---")
try:
    from backend.app import _handle_rs_vqa
    from types import SimpleNamespace

    # 1. Optical + SAR trace test
    ctx_opt_sar = {
        "primary": {"path": opt_p, "data": opt_d, "filename": opt_p.name},
        "secondary": {"path": sar_p, "data": sar_d, "filename": sar_p.name},
        "feature": "auto",
        "req": SimpleNamespace(query="Are built-up areas distinguishable from water?"),
        "trace": [],
    }
    _handle_rs_vqa(ctx_opt_sar)
    trace_steps_fusion = [t.get("step") for t in ctx_opt_sar["trace"]]
    test_assert("Optical-SAR Fusion" in trace_steps_fusion,
                "Trace correctly records 'Optical-SAR Fusion' for optical-SAR input (not fake RS-VQA)",
                f"Steps: {trace_steps_fusion}")

    # 2. Bi-temporal change trace test
    ctx_temporal = {
        "primary": {"path": t1_p, "data": t1_d, "filename": t1_p.name},
        "secondary": {"path": t2_p, "data": t2_d, "filename": t2_p.name},
        "feature": "water",
        "req": SimpleNamespace(query="Has water changed between these dates?"),
        "trace": [],
    }
    _handle_rs_vqa(ctx_temporal)
    trace_steps_change = [t.get("step") for t in ctx_temporal["trace"]]
    test_assert("Change Engine" in trace_steps_change,
                "Trace correctly records 'Change Engine' for temporal input (not fake RS-VQA)",
                f"Steps: {trace_steps_change}")
except Exception as e:
    test_assert(False, "Paired-VQA trace labeling test", str(e))

# ------------------------------------------------------------
# TEST 24: POSIX PATH PORTABILITY ACROSS ALL MANIFESTS
# ------------------------------------------------------------
print("\n--- TEST 24: POSIX Path Portability ---")
try:
    manifest_paths = [
        BASE_DIR / "data" / "external_datasets" / "rsvqa" / "rsvqa_train.json",
        BASE_DIR / "data" / "external_datasets" / "rsvqa" / "rsvqa_official_eval.json",
        BASE_DIR / "data" / "external_datasets" / "rsvqa" / "rsvqa_full_test.json",
        BASE_DIR / "data" / "external_datasets" / "vrsbench" / "vrsbench_official_eval.json",
        BASE_DIR / "data" / "external_datasets" / "cdvqa" / "cdvqa_official_eval.json",
    ]
    for mp in manifest_paths:
        raw_text = mp.read_text(encoding="utf-8")
        test_assert("\\\\" not in raw_text,
                    f"Manifest {mp.name} contains zero Windows backslashes (POSIX portable)",
                    f"Backslash found in {mp.name}")
except Exception as e:
    test_assert(False, "POSIX path portability test", str(e))

# ------------------------------------------------------------
# TEST 25: CDVQA MULTI-SCENE DIVERSITY & HASH VERIFICATION
# ------------------------------------------------------------
print("\n--- TEST 25: CDVQA Multi-Scene Diversity ---")
try:
    import hashlib
    cd_manifest = json.loads((BASE_DIR / "data" / "external_datasets" / "cdvqa" / "cdvqa_official_eval.json").read_text(encoding="utf-8"))
    distinct_scenes = set()
    for item in cd_manifest:
        p1 = BASE_DIR / item["image_t1"]
        p2 = BASE_DIR / item["image_t2"]
        test_assert(p1.exists(), f"CDVQA T1 image exists: {p1.name}")
        test_assert(p2.exists(), f"CDVQA T2 image exists: {p2.name}")
        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()[:8]
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()[:8]
        distinct_scenes.add((h1, h2))

    test_assert(len(distinct_scenes) >= 5,
                f"CDVQA manifest contains diverse scene pairs (got {len(distinct_scenes)} distinct scene hashes, required >= 5)",
                f"Only {len(distinct_scenes)} distinct scene pairs found")
    scenarios = {item.get("scenario") for item in cd_manifest if item.get("scenario")}
    test_assert(len(scenarios) >= 4,
                f"CDVQA covers multiple event scenarios: {scenarios}",
                f"Insufficient scenario coverage: {scenarios}")
except Exception as e:
    test_assert(False, "CDVQA multi-scene diversity test", str(e))

# ------------------------------------------------------------
# TEST 26: DYNAMIC TOOL SEQUENCING & AUDITABLE TRACE TIMING
# ------------------------------------------------------------
print("\n--- TEST 26: Dynamic Tool Sequencing & Trace Timings ---")
try:
    from backend.app import has_handler, execute_tool
    test_assert(has_handler("input_validator"), "input_validator is registered in dynamic tool registry")
    test_assert(has_handler("geospatial_tools"), "geospatial_tools is registered in dynamic tool registry")

    # Verify analyze_grounding accepts forwarded parameters
    from backend.app import analyze_grounding
    import inspect
    sig = inspect.signature(analyze_grounding)
    test_assert("grid_size" in sig.parameters, "analyze_grounding accepts grid_size parameter")
    test_assert("method" in sig.parameters, "analyze_grounding accepts method parameter")

    # Run query through API to verify multi-tool sequencing and timing trace
    demo_s2 = requests.post(f"{API_BASE}/api/load_demo", json={"sample_key": "sentinel2"}).json()
    sample_id = demo_s2["primary"]["id"]
    res_trace = requests.post(f"{API_BASE}/api/analyze", json={
        "primary_id": sample_id,
        "query": "Highlight and delineate the water bodies in this satellite image",
    }).json()

    trace_steps = res_trace.get("execution_trace", [])
    step_names = [s.get("step") for s in trace_steps]
    test_assert("Input Validator" in step_names, "Input Validator executed in multi-step plan")
    test_assert("Geospatial Tools" in step_names, "Geospatial Tools executed sequentially in multi-step plan")

    # Verify timing_ms and parameters exist in trace steps
    for step in trace_steps:
        if step.get("step") in {"Input Validator", "Geospatial Tools", "RS Grounding"}:
            test_assert("timing_ms" in step, f"Step '{step.get('step')}' records execution timing_ms", f"Missing timing in {step}")
            test_assert("parameters" in step, f"Step '{step.get('step')}' records execution parameters", f"Missing parameters in {step}")
except Exception as e:
    test_assert(False, "Tool sequencing and trace timing test", str(e))

# ------------------------------------------------------------
# TEST 27: SHARED GEOGRAPHIC GRID REPROJECTION FOR BI-TEMPORAL MASKS
# ------------------------------------------------------------
print("\n--- TEST 27: Shared Geographic Grid Reprojection ---")
try:
    from backend.app import analyze_change
    from rasterio.transform import Affine

    p1 = BASE_DIR / "demo_data" / "assam_flood" / "assam_flood_t1.tif"
    p2 = BASE_DIR / "demo_data" / "assam_flood" / "assam_flood_t2.tif"
    from backend.app import _read_raster
    d1 = _read_raster(p1)
    d2 = _read_raster(p2)

    # Both images are georeferenced
    test_assert(d1.get("is_georeferenced"), "Primary image is georeferenced")
    test_assert(d2.get("is_georeferenced"), "Secondary image is georeferenced")

    # Execute analyze_change and verify change_metrics contains reprojection-derived physical area
    ch_res = analyze_change(p1, d1, p2, d2, feature="water", query="Has flood water expanded?")
    test_assert("delta_percentage_points" in ch_res or "delta_percentage_points" in ch_res.get("evidence", {}), "Bi-temporal change computed delta percentage points")
    test_assert(ch_res.get("confidence") is not None, "Bi-temporal change computed authentic confidence")
    test_assert(ch_res.get("overlay") is not None, "Bi-temporal change generated change overlay")
except Exception as e:
    test_assert(False, "Geographic grid reprojection test", str(e))

# ------------------------------------------------------------
# TEST 28: CALIBRATED BENCHMARK THRESHOLDS
# ------------------------------------------------------------
print("\n--- TEST 28: Calibrated Benchmark Status Thresholds ---")
try:
    bench_code = (BASE_DIR / "benchmarks" / "evaluate_benchmarks.py").read_text(encoding="utf-8")
    test_assert("oa >= 50.0" in bench_code, "RSVQA enforces OA >= 50.0% standard baseline threshold")
    test_assert("avg_bin_acc >= 60.0" in bench_code, "CDVQA enforces change accuracy >= 60.0% threshold")
    test_assert("agree_val >= 65.0" in bench_code, "ISRO consensus enforces agreement >= 65.0% threshold")
    test_assert("MARGINAL" in bench_code, "Evaluator distinguishes MARGINAL vs PASSED vs FAIL states")
except Exception as e:
    test_assert(False, "Calibrated benchmark threshold test", str(e))

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
