from __future__ import annotations
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

    def post_upload(p: Path, mime: str):
        if use_testclient:
            with open(p, "rb") as f:
                return client.post("/api/upload", files={"file": (p.name, f, mime)}).json()
        with open(p, "rb") as f:
            return requests.post(f"{BASE}/api/upload", files={"file": (p.name, f, mime)}).json()

    def post_analyze(json_data: dict):
        if use_testclient:
            return client.post("/api/analyze", json=json_data).json()
        return requests.post(f"{BASE}/api/analyze", json=json_data).json()

    print("=== Segmentation on ALL benchmark images ===\n")
    images = {
        "Kolkata RGB (3-band optical)":        "demo_data/vrsbench/vrsbench_sample_01.tif",
        "Sentinel-2 Multispectral (4-band T1)": "demo_data/bigearthnet/S2_multispectral_patch.tif",
        "ISRO Optical (Cartosat)":             "demo_data/isro_sac/cartosat_optical_coregistered.tif",
        "Kolkata JPEG":                        "demo_data/vrsbench/vrsbench_sample_01.jpg",
    }

    for label, img_path in images.items():
        p = Path(img_path)
        if not p.exists():
            continue
        mime = "image/jpeg" if p.suffix == ".jpg" else "image/tiff"
        up = post_upload(p, mime)

        res = post_analyze({
            "primary_id": up["id"],
            "query": "Identify the green fields, buildings, and water in different colours.",
        })

        ev = res.get("evidence", {})
        print(f"[{label}]")
        print(f"  Tool   : {res.get('tool')}")
        print(f"  Answer : {res.get('answer')}")
        print(f"  Water  : {ev.get('water_percent')}%")
        print(f"  Veg    : {ev.get('vegetation_percent')}%")
        print(f"  Built  : {ev.get('built_up_percent')}%")
        print(f"  Other  : {ev.get('unclassified_percent')}%")
        print()

