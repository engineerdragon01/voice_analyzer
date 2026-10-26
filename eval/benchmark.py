"""Summarize latency and cost from a CLI run's <out>.meta.json.

    python -m eval.benchmark results/hybrid.meta.json
"""
from __future__ import annotations

import json
import sys
from statistics import mean, median

from analyzer.config import COST_CEILING_PER_MIN, Config, choose_samples, estimate_call_cost


def summarize(meta_path: str) -> dict:
    rows = [r["meta"] for r in json.load(open(meta_path)) if r["meta"]]
    mins = sum(r["duration_s"] for r in rows) / 60
    cost = sum(r.get("cost_usd", 0) for r in rows)
    out = {
        "files": len(rows),
        "audio_minutes": round(mins, 2),
        "total_cost_usd": round(cost, 5),
        "cost_per_audio_min": round(cost / mins, 5) if mins else None,
        "max_cost_per_min_single_file": round(max(r.get("cost_per_min_usd", 0) for r in rows), 5),
        "latency_per_file_s": {"mean": round(mean(r["total_s"] for r in rows), 2),
                               "median": round(median(r["total_s"] for r in rows), 2),
                               "max": round(max(r["total_s"] for r in rows), 2)},
        "processing_s_per_audio_min": round(sum(r["total_s"] for r in rows) / mins, 2) if mins else None,
        "stage_share": {k: round(sum(r.get(k, 0) for r in rows) / max(1e-9, sum(r["total_s"] for r in rows)), 3)
                        for k in ("decode_s", "dsp_s", "llm_latency_s")},
        "tokens": {k: sum(r.get("usage", {}).get(k, 0) for r in rows) for k in ("audio_in", "text_in", "out")},
        "under_ceiling": (cost / mins) <= COST_CEILING_PER_MIN if mins else None,
    }
    return out


def projection_table(model: str = "gemini-2.5-flash-lite") -> str:
    cfg = Config(model=model)
    lines = ["| clip length | samples (k) | est. cost/clip | est. cost/audio-min |", "|---|---|---|---|"]
    for d in (15, 30, 60, 120, 300, 900):
        k = choose_samples(cfg, d)
        c = estimate_call_cost(cfg, d, k)
        lines.append(f"| {d}s | {k} | ${c:.5f} | ${c / (d / 60):.5f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--projection":
        print(projection_table(sys.argv[2] if len(sys.argv) > 2 else "gemini-2.5-flash-lite"))
    else:
        print(json.dumps(summarize(sys.argv[1]), indent=2))
