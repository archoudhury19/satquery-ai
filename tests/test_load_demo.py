import requests

BASE = "http://127.0.0.1:8000"

for key in ["sentinel2", "kolkata", "optical_sar", "bitemporal"]:
    res = requests.post(f"{BASE}/api/load_demo", json={"sample_key": key}).json()
    title = res.get("title")
    prim = res.get("primary", {}).get("filename")
    sec = res.get("secondary", {}).get("filename") if res.get("secondary") else "None"
    print(f"[Demo: {key}] -> Title: {title}")
    print(f"  Primary   : {prim}")
    print(f"  Secondary : {sec}")
    print()
