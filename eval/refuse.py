"""Re-apply the current fusion rules to stored LLM candidates (no API calls).

    python -m eval.refuse results/synthetic_hybrid.meta.json data/synthetic/labels.csv [audio_dir]

With audio_dir, DSP features are recomputed with the current code (e.g. after adding a detector).
"""
from __future__ import annotations

import csv
import json
import sys

from analyzer import fusion
from analyzer.config import Config
from analyzer.metrics import score


def refuse(meta_path: str, audio_dir: str | None = None) -> dict[str, dict]:
    from pathlib import Path

    from analyzer import audio_io, dsp

    cfg = Config()
    out = {}
    for r in json.load(open(meta_path)):
        m = r["meta"]
        if not m.get("candidates"):
            continue
        feats = m["features"]
        if audio_dir:
            a = audio_io.load(Path(audio_dir) / r["name"])
            feats = dsp.features(a.stereo, a.mono)
        out[r["name"]], _ = fusion.fuse(m["candidates"], feats, cfg)
    return out


if __name__ == "__main__":
    preds = refuse(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
    labels = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(open(sys.argv[2])) if r["result_json"]}
    s = score([(labels[n], p) for n, p in preds.items() if n in labels and len(labels[n]) > 3])
    if s.get("n"):
        print(json.dumps({"mean_field_accuracy": s["mean_field_accuracy"], "tone_accuracy": s["tone_accuracy"],
                          **{k: v["accuracy"] for k, v in s["fields"].items()}}, indent=1))
    for n, p in preds.items():
        print(n, p["background_noise_present"], p["background_noise_type"], p["background_noise_severity"], p["confidence"])
