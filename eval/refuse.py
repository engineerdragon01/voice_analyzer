"""Re-apply the current fusion rules to stored LLM candidates (no API calls).

    python -m eval.refuse results/synthetic_hybrid.meta.json data/synthetic/labels.csv
"""
from __future__ import annotations

import csv
import json
import sys

from analyzer import fusion
from analyzer.config import Config
from analyzer.metrics import score


def refuse(meta_path: str) -> dict[str, dict]:
    cfg = Config()
    out = {}
    for r in json.load(open(meta_path)):
        m = r["meta"]
        if m.get("candidates"):
            out[r["name"]], _ = fusion.fuse(m["candidates"], m["features"], cfg)
    return out


if __name__ == "__main__":
    preds = refuse(sys.argv[1])
    labels = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(open(sys.argv[2])) if r["result_json"]}
    print(json.dumps(score([(labels[n], p) for n, p in preds.items() if n in labels]), indent=1)[:1500])
