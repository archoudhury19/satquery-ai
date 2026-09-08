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

    images = [
        ("demo_data/vrsbench/vrsbench_sample_01.tif", "Kolkata High-Res Optical"),
        ("demo_data/bigearthnet/S2_multispectral_patch.tif", "Sentinel-2 Multispectral"),
    ]

    print("============================================================")
    print("TESTING ADVANCED REMOTE-SENSING SCENE CAPTIONING")
    print("============================================================\n")

    for img_path, label in images:
        p = Path(img_path)
        if not p.exists():
            continue
        up = post_upload(p)

        res = post_analyze({
            "primary_id": up["id"],
            "query": "Describe the land-cover and major objects visible in this image.",
        })

        print(f"[{label}]")
        print(f"  Tool       : {res.get('tool')}")
        print(f"  Caption    : {res.get('answer')}")
        print(f"  Confidence : {round(res.get('confidence', 0) * 100)}%")
        print(f"  Overlay    : {res.get('overlay_url')}")
        print()

