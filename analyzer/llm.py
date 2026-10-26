"""Gemini audio backend: one request per clip, k candidates sampled in-request (input billed once)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import Config
from .schema import LLM_RESPONSE_SCHEMA

PROMPT_VERSION = "v1"
_PROMPT = (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.txt").read_text()
_client = None


class LLMError(Exception):
    pass


def _get_client(cfg: Config):
    global _client
    if _client is None:
        from google import genai

        if cfg.use_vertex:
            _client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                   location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
        else:
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY is not set")
            _client = genai.Client(api_key=key)
    return _client


def available(cfg: Config) -> bool:
    return cfg.use_vertex or bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def dsp_summary(f: dict) -> str:
    lines = [
        f"- Duration {f['duration_s']:.0f}s; speech present in {f['speech_ratio']*100:.0f}% of frames.",
        f"- Longest pause between speech inside the call: {f['longest_interior_gap_s']:.1f}s "
        f"(leading {f['leading_silence_s']:.1f}s, trailing {f['trailing_silence_s']:.1f}s).",
        f"- Speech-to-background level ratio (SNR): {f['snr_db']:.0f} dB "
        f"(>35 dB is a very quiet background; <20 dB means a loud background).",
        f"- Clipping {'SUSPECTED' if f['clipping_suspected'] else 'not detected'}; "
        f"possible brief dropouts: {f['dropouts']} (unreliable on compressed audio); "
        f"overall level {f['rms_dbfs']:.0f} dBFS{' (very quiet)' if f['rms_dbfs'] < -45 else ''}.",
    ]
    if f.get("stereo_split"):
        lines.append(f"- Separate speaker channels: simultaneous speech {f['overlap_s']:.1f}s in "
                     f"{f['overlap_events']} events (longest {f['overlap_longest_s']:.1f}s).")
    else:
        lines.append("- Recording is mono (speakers share one channel); judge overlap by ear.")
    return "\n".join(lines)


def _usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {}
    audio_in = text_in = 0
    for d in (getattr(u, "prompt_tokens_details", None) or []):
        mod = str(getattr(d, "modality", "")).upper()
        if "AUDIO" in mod:
            audio_in += d.token_count or 0
        else:
            text_in += d.token_count or 0
    prompt = u.prompt_token_count or 0
    if audio_in + text_in == 0:
        text_in = prompt
    out = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return {"audio_in": audio_in, "text_in": text_in, "out": out}


def cost_from_usage(cfg: Config, usage: dict) -> float:
    p = cfg.price()
    return (usage.get("audio_in", 0) * p["audio_in"] + usage.get("text_in", 0) * p["text_in"]
            + usage.get("out", 0) * p["out"]) / 1e6


def classify(cfg: Config, audio_bytes: bytes, features: dict, samples: int) -> tuple[list[dict], dict]:
    """Return (list of candidate label dicts, meta{usage, cost, latency, attempts})."""
    from google.genai import types

    client = _get_client(cfg)
    prompt = _PROMPT.replace("{dsp_summary}", dsp_summary(features))
    contents = [types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"), prompt]

    def make_cfg(n: int, temp: float):
        kw = dict(
            response_mime_type="application/json",
            response_schema=LLM_RESPONSE_SCHEMA,
            temperature=temp,
            candidate_count=n,
            max_output_tokens=600,
            seed=1234,
            http_options=types.HttpOptions(timeout=cfg.timeout_s * 1000),
        )
        if "2.5" in cfg.model or "3" in cfg.model:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kw)

    t0 = time.time()
    usage_total = {"audio_in": 0, "text_in": 0, "out": 0}
    candidates: list[dict] = []
    last_err: Exception | None = None
    attempts = 0
    multi_ok = True
    while len(candidates) < samples and attempts < 6:
        attempts += 1
        n = samples - len(candidates) if multi_ok else 1
        temp = 0.0 if samples == 1 else cfg.temperature
        try:
            resp = client.models.generate_content(model=cfg.model, contents=contents, config=make_cfg(n, temp))
        except Exception as e:  # noqa: BLE001 — SDK raises many types
            last_err = e
            msg = str(e).lower()
            if multi_ok and n > 1 and ("candidate" in msg or "invalid_argument" in msg or "400" in msg):
                multi_ok = False  # model rejects candidate_count>1: fall back to sequential calls
                continue
            time.sleep(min(2 ** attempts, 20))
            continue
        for k, v in _usage(resp).items():
            usage_total[k] += v
        for c in (resp.candidates or []):
            try:
                text = "".join(p.text or "" for p in c.content.parts)
                candidates.append(json.loads(text))
            except Exception as e:  # noqa: BLE001
                last_err = e
    if not candidates:
        raise LLMError(f"Gemini call failed: {last_err}")
    meta = {
        "model": cfg.model,
        "prompt_version": PROMPT_VERSION,
        "samples": len(candidates),
        "usage": usage_total,
        "cost_usd": round(cost_from_usage(cfg, usage_total), 6),
        "llm_latency_s": round(time.time() - t0, 2),
        "attempts": attempts,
    }
    return candidates[:samples], meta
