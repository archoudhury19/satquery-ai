from __future__ import annotations

import sys
import json
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"

def test_segmentation():
    img = Path("demo_data/vrsbench/vrsbench_sample_01.tif")
    with open(img, "rb") as f:
        up = requests.post(f"{BASE}/api/upload", files={"file": (img.name, f, "image/tiff")}).json()

    print("Uploaded:", up["id"], up["filename"])

    for query in [
        "Identify the green fields, buildings, and water in different colours.",
        "Segment the land cover into water, vegetation, and built-up areas.",
        "Can it identify the green fields and buildings in different colours?",
        "Show me a colour-coded map of all land-cover classes.",
        "classify land use",
    ]:
        t0 = time.time()
        res = requests.post(f"{BASE}/api/analyze", json={
            "primary_id": up["id"],
            "query": query,
        }).json()
        elapsed = round(time.time() - t0, 3)

        print(f'\nQuery: "{query}"')
        print(f"  Task    : {res.get('task')}")
        print(f"  Tool    : {res.get('tool')}")
        print(f"  Answer  : {res.get('answer')}")
        print(f"  Overlay : {res.get('overlay_url')}")
        print(f"  Evidence: {json.dumps(res.get('evidence', {}), indent=4)}")
        print(f"  Latency : {elapsed}s")

if __name__ == "__main__":
    test_segmentation()
