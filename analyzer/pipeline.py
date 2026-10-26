"""Single entry point used by the CLI and the web app."""
from __future__ import annotations

import time
from pathlib import Path

from . import audio_io, dsp, fusion, llm
from .config import COMPACT_BELOW_S, Config, choose_samples


def analyze_file(path: str | Path, cfg: Config | None = None) -> tuple[dict, dict]:
    """Return (prediction in the required schema, meta with timings/cost/diagnostics).
    Raises audio_io.AudioError / llm.LLMError with a readable reason on failure."""
    cfg = cfg or Config()
    t0 = time.time()
    audio = audio_io.load(path)
    t_decode = time.time() - t0

    t1 = time.time()
    feats = dsp.features(audio.stereo, audio.mono)
    t_dsp = time.time() - t1

    meta: dict = {
        "duration_s": round(audio.duration_s, 2),
        "decode_s": round(t_decode, 3),
        "dsp_s": round(t_dsp, 3),
        "backend": cfg.backend,
        "features": feats,
        "cost_usd": 0.0,
    }

    if cfg.backend == "dsp":
        pred = fusion.dsp_only(feats, cfg)
    else:
        k = choose_samples(cfg, audio.duration_s)
        api_audio, mime = audio_io.encode_for_api(audio.mono)
        candidates, lmeta = llm.classify(cfg, api_audio, feats, k, compact=audio.duration_s < COMPACT_BELOW_S,
                                         mime_type=mime)
        lmeta["api_audio_format"] = mime
        pred, diag = fusion.fuse(candidates, feats, cfg, use_dsp_overrides=(cfg.backend == "hybrid"))
        meta.update(lmeta)
        meta["diagnostics"] = diag
        meta["candidates"] = candidates  # raw model outputs: lets fusion changes be re-scored offline
        meta["evidence"] = [{"customer": c.get("customer_evidence", ""), "noise": c.get("noise_evidence", "")}
                            for c in candidates]

    meta["total_s"] = round(time.time() - t0, 2)
    minutes = max(audio.duration_s, 1e-6) / 60
    meta["cost_per_min_usd"] = round(meta["cost_usd"] / minutes, 6)
    return pred, meta
