"""Batch analysis from the command line.

    python cli.py data/provided -o predictions/provided_calls --backend hybrid
writes <out>.csv (name,result_json), <out>.json, <out>.meta.json and prints metrics if labels exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analyzer import metrics
from analyzer.config import Config
from analyzer.manifest import plan_batch
from analyzer.pipeline import analyze_file


def run_batch(folder: Path, cfg: Config, workers: int = 4) -> tuple[list[dict], dict]:
    plan = plan_batch(folder)
    rows: list[dict] = []

    def work(name: str) -> dict:
        try:
            pred, meta = analyze_file(plan.root / name, cfg)
            return {"name": name, "status": "done", "prediction": pred, "meta": meta, "error": None}
        except Exception as e:  # noqa: BLE001 — one bad file must not fail the batch
            return {"name": name, "status": "error", "prediction": None, "meta": {}, "error": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(work, plan.to_process):
            r["label"] = plan.labels.get(r["name"])
            rows.append(r)
            status = "ok " if r["status"] == "done" else "ERR"
            print(f"[{status}] {r['name']} {r['error'] or ''}", file=sys.stderr)
    return rows, plan.report()


def write_outputs(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json", "error"])
        for r in rows:
            w.writerow([r["name"], json.dumps(r["prediction"]) if r["prediction"] else "", r["error"] or ""])
    out.with_suffix(".json").write_text(json.dumps(
        [{"name": r["name"], "result_json": r["prediction"], "error": r["error"]} for r in rows], indent=2))
    out.with_suffix(".meta.json").write_text(json.dumps(
        [{"name": r["name"], "meta": r["meta"]} for r in rows], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results/out"))
    ap.add_argument("--backend", choices=["hybrid", "gemini", "dsp"])
    ap.add_argument("--model")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = Config()
    if args.backend:
        cfg.backend = args.backend
    if args.model:
        cfg.model = args.model
    if args.samples:
        cfg.max_samples = args.samples
    t0 = time.time()
    rows, report = run_batch(args.folder, cfg, args.workers)
    wall = time.time() - t0
    write_outputs(rows, args.out)
    done = [r for r in rows if r["status"] == "done"]
    audio_min = sum(r["meta"]["duration_s"] for r in done) / 60
    cost = sum(r["meta"].get("cost_usd", 0) for r in done)
    print(json.dumps({"validation": report, "processed": len(done), "failed": len(rows) - len(done),
                      "audio_minutes": round(audio_min, 2), "wall_s": round(wall, 1),
                      "cost_usd": round(cost, 5),
                      "cost_per_audio_min": round(cost / audio_min, 5) if audio_min else None}, indent=2))
    pairs = [(r["label"], r["prediction"]) for r in done if r["label"]]
    if pairs:
        s = metrics.score(pairs)
        print(metrics.to_markdown(s, f"{cfg.backend} / {cfg.model if cfg.backend != 'dsp' else 'rules'}"))
        args.out.with_suffix(".metrics.json").write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
