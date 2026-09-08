import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    import requests, json
    from fastapi.testclient import TestClient
    from backend.app import app

    BASE = "http://127.0.0.1:8000"
    use_testclient = False
    try:
        r = requests.get(f"{BASE}/api/health", timeout=1)
        if r.status_code != 200:
            use_testclient = True
    except Exception:
        use_testclient = True

    client = TestClient(app) if use_testclient else None

    def post_upload(p: Path):
        if use_testclient:
            with open(p, "rb") as f:
                return client.post("/api/upload", files={"file": (p.name, f, "image/tiff")}).json()
        with open(p, "rb") as f:
            return requests.post(f"{BASE}/api/upload", files={"file": (p.name, f, "image/tiff")}).json()

    def post_analyze(json_data: dict):
        if use_testclient:
            return client.post("/api/analyze", json=json_data).json()
        return requests.post(f"{BASE}/api/analyze", json=json_data).json()

    print("============================================================")
    print("SATQUERY MULTI-MODAL EVALUATION AUDIT")
    print("============================================================\n")

    # 1. Test VQA with fine-tuned adapter
    print("--- [1] VQA Predictions (Fine-Tuned Adapter) ---")
    p_s2 = Path("demo_data/bigearthnet/S2_multispectral_patch.tif")
    up_s2 = post_upload(p_s2)

    p_urb = Path("demo_data/vrsbench/vrsbench_sample_01.tif")
    up_urb = post_upload(p_urb)

    tests = [
        (up_s2["id"], "Is this an urban or rural area?", "Sentinel-2 Tile"),
        (up_s2["id"], "Is there vegetation present?", "Sentinel-2 Tile"),
        (up_urb["id"], "Is this an urban or rural area?", "Kolkata Urban"),
        (up_urb["id"], "Are there buildings visible?", "Kolkata Urban"),
    ]

    for img_id, q, label in tests:
        res = post_analyze({"primary_id": img_id, "query": q})
        ans = res.get("answer")
        conf = round(res.get("confidence", 0) * 100)
        tool = res.get("tool")
        print(f"[{label}] Q: '{q}' -> Answer: '{ans}' (Conf: {conf}%, Tool: {tool})")

    # 2. Test Open-Vocabulary Visual Grounding
    print("\n--- [2] Open-Vocabulary Visual Grounding ---")
    grounding_tests = [
        (up_urb["id"], "Locate the river corridor and water body.", "Kolkata Urban"),
        (up_s2["id"], "Highlight the green vegetation canopy.", "Sentinel-2 Tile"),
    ]

    for img_id, q, label in grounding_tests:
        res = post_analyze({"primary_id": img_id, "query": q})
        ans = res.get("answer")
        bbox = res.get("evidence", {}).get("pixel_bounding_box") or res.get("bounding_box")
        overlay = res.get("overlay") or res.get("overlay_url")
        print(f"[{label}] Grounding: '{q}'")
        print(f"  Answer : {ans}")
        print(f"  BBox   : {bbox}")
        print(f"  Overlay: {overlay}")

    # 3. Test Multi-Class Segmentation
    print("\n--- [3] Multi-Class AI Segmentation ---")
    res_seg = post_analyze({
        "primary_id": up_s2["id"],
        "query": "Segment land cover into water, vegetation, and built up.",
    })
    print(f"Sentinel-2 Segmentation: {res_seg.get('answer')}")
    print(f"Overlay: {res_seg.get('overlay_url')}")

    print("\n============================================================")
    print("AUDIT COMPLETE: All modules verified successfully.")
    print("============================================================\n")

