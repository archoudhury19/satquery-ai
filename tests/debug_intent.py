import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.planner import understand_query

tests = [
    "Segment the land cover into water, vegetation, and built-up areas.",
    "Show me a colour-coded map of all land-cover classes.",
    "Identify the green fields, buildings, and water in different colours.",
]
for q in tests:
    intent = understand_query(q, 1)
    print(f'Q: "{q}"')
    print(f'  captioning={intent["captioning"]}, segmentation={intent["segmentation"]}')
