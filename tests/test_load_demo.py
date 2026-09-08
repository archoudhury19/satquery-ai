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

    for key in ["sentinel2", "kolkata", "optical_sar", "bitemporal"]:
        res = post_api("/api/load_demo", {"sample_key": key})
        title = res.get("title")
        prim = res.get("primary", {}).get("filename")
        sec = res.get("secondary", {}).get("filename") if res.get("secondary") else "None"
        print(f"[Demo: {key}] -> Title: {title}")
        print(f"  Primary   : {prim}")
        print(f"  Secondary : {sec}")
        print()

