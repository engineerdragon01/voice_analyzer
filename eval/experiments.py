"""Run the comparison matrix on a labeled folder and write a markdown table.

    python -m eval.experiments data/provided docs/results/experiments.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from analyzer import metrics
from analyzer.config import Config
from cli import run_batch, write_outputs

EXPERIMENTS = [
    # name, backend, model, prompt, max_samples
    ("A. DSP rules only", "dsp", None, None, 1),
    ("B. Gemini 3.5 Flash-Lite, prompt v1, k=1", "gemini", "gemini-3.5-flash-lite", "v1", 1),
    ("C. Gemini 3.5 Flash-Lite, prompt v2, k=1", "gemini", "gemini-3.5-flash-lite", "v2", 1),
    ("D. Hybrid 3.5 Flash-Lite, v2, k=1", "hybrid", "gemini-3.5-flash-lite", "v2", 1),
    ("E. Hybrid 3.5 Flash-Lite, v2, adaptive k<=3 (final)", "hybrid", "gemini-3.5-flash-lite", "v2", 3),
    ("F. Hybrid 3.1 Flash-Lite, v2, k=1", "hybrid", "gemini-3.1-flash-lite", "v2", 1),
]
FIELDS_SHORT = {"emotional_tone": "tone", "emotional_intensity": "int.", "background_noise_present": "noise",
                "background_noise_severity": "sev.", "audio_quality": "qual.", "speaker_overlap_present": "overlap",
                "long_silence_present": "silence"}


def main(folder: Path, out_md: Path, repeats: int = 2) -> None:
    rows = []
    details = {}
    for name, backend, model, prompt, k in EXPERIMENTS:
        for rep in range(repeats if backend != "dsp" else 1):
            cfg = Config()
            cfg.backend = backend
            if model:
                cfg.model = model
            if prompt:
                cfg.prompt_version = prompt
            cfg.max_samples = k
            t0 = time.time()
            res, _ = run_batch(folder, cfg, workers=4)
            wall = time.time() - t0
            tag = name.split(".")[0] + str(rep + 1)
            write_outputs(res, Path("results/exp") / tag)
            done = [r for r in res if r["status"] == "done"]
            s = metrics.score([(r["label"], r["prediction"]) for r in done if r["label"]])
            mins = sum(r["meta"]["duration_s"] for r in done) / 60
            cost = sum(r["meta"].get("cost_usd", 0) for r in done)
            rows.append((name, rep + 1, s, cost / mins if mins else 0, wall, len(res) - len(done)))
            details[f"{name} (run {rep + 1})"] = {r["name"]: r["prediction"] for r in done}
            print(f"{name} run {rep + 1}: tone {s['tone_accuracy']:.2f} mean {s['mean_field_accuracy']:.3f}", file=sys.stderr)

    lines = ["# Experiment matrix — provided labeled calls (n=3)", "",
             "Zero-shot: labels are never shown to the model. Each Gemini config run twice to expose run-to-run variance.",
             "", "| Config | run | tone acc | tone macro-F1 | " + " | ".join(FIELDS_SHORT.values())
             + " | mean field acc | $/audio-min | wall s (3 files, 4 workers) |",
             "|---|---|---|---|" + "---|" * len(FIELDS_SHORT) + "---|---|---|"]
    for name, rep, s, cpm, wall, failed in rows:
        f = s["fields"]
        lines.append(f"| {name} | {rep} | {s['tone_accuracy']:.2f} | {s['tone_macro_f1']:.2f} | "
                     + " | ".join(f"{f[k]['accuracy']:.2f}" for k in FIELDS_SHORT)
                     + f" | **{s['mean_field_accuracy']:.3f}** | {cpm:.5f} | {wall:.1f} |")
    lines += ["", "## Per-call predictions", "", "```json", json.dumps(details, indent=1), "```"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text().split("## Per-call")[0])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
