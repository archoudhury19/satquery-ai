import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    import requests
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

    def post_api(endpoint: str, json_data: dict):
        if use_testclient:
            return client.post(endpoint, json=json_data).json()
        return requests.post(f"{BASE}{endpoint}", json=json_data).json()

    print("============================================================")
    print("EVALUATING REAL SATELLITE IMAGES DOWNLOADED FROM INTERNET")
    print("============================================================\n")

    for sample_key in ["real_sf", "real_sentinel"]:
        load_res = post_api("/api/load_demo", {"sample_key": sample_key})
        prim_id = load_res["primary"]["id"]
        title = load_res["title"]
        print(f"[*] Analyzing: {title}")

        # 1. Captioning
        cap_res = post_api("/api/analyze", {
            "primary_id": prim_id,
            "query": "Describe the land-cover and major objects visible in this image."
        })
        print(f"    Caption : {cap_res.get('answer')}")

        # 2. Segmentation
        seg_res = post_api("/api/analyze", {
            "primary_id": prim_id,
            "query": "Segment the land cover into water, vegetation, and built up."
        })
        print(f"    Segment : {seg_res.get('answer')}")
        print(f"    Overlay : {seg_res.get('overlay_url')}\n")

