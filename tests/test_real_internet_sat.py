import requests

BASE = "http://127.0.0.1:8000"

print("============================================================")
print("EVALUATING REAL SATELLITE IMAGES DOWNLOADED FROM INTERNET")
print("============================================================\n")

for sample_key in ["real_sf", "real_sentinel"]:
    load_res = requests.post(f"{BASE}/api/load_demo", json={"sample_key": sample_key}).json()
    prim_id = load_res["primary"]["id"]
    title = load_res["title"]
    print(f"[*] Analyzing: {title}")

    # 1. Captioning
    cap_res = requests.post(f"{BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Describe the land-cover and major objects visible in this image."
    }).json()
    print(f"    Caption : {cap_res.get('answer')}")

    # 2. Segmentation
    seg_res = requests.post(f"{BASE}/api/analyze", json={
        "primary_id": prim_id,
        "query": "Segment the land cover into water, vegetation, and built up."
    }).json()
    print(f"    Segment : {seg_res.get('answer')}")
    print(f"    Overlay : {seg_res.get('overlay_url')}\n")
