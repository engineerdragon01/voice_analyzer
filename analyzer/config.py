"""Runtime configuration (env-driven) and the pricing table used for cost accounting."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# USD per 1M tokens (paid tier, standard). Verify at https://ai.google.dev/gemini-api/docs/pricing
PRICING = {
    "gemini-3.5-flash-lite": {"audio_in": 0.30, "text_in": 0.30, "out": 2.50},
    "gemini-3.1-flash-lite": {"audio_in": 0.50, "text_in": 0.25, "out": 1.50},
    "gemini-2.5-flash-lite": {"audio_in": 0.30, "text_in": 0.10, "out": 0.40},  # closed to new users
}
# Conservative assumption for models without a verified price (keeps the budget guard pessimistic).
UNKNOWN_MODEL_PRICE = {"audio_in": 1.00, "text_in": 0.50, "out": 3.00}
# Measured on the provided calls: 774 audio tokens for 30.9 s (~25/s). Prompt ~970 text tokens, ~135 output.
AUDIO_TOKENS_PER_SEC = 25
PROMPT_TOKENS = 1000
OUTPUT_TOKENS_PER_SAMPLE = 150
OUTPUT_TOKENS_COMPACT = 90
COMPACT_BELOW_S = 20  # clips shorter than this skip the free-text evidence fields
COST_CEILING_PER_MIN = 0.003


@dataclass
class Config:
    backend: str = field(default_factory=lambda: os.getenv("ANALYZER_BACKEND", "hybrid"))  # hybrid|gemini|dsp
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    max_samples: int = field(default_factory=lambda: int(os.getenv("GEMINI_SAMPLES", "3")))
    # Budget the per-clip sample count against this (leaves headroom under the $0.003 ceiling).
    target_cost_per_min: float = field(default_factory=lambda: float(os.getenv("TARGET_COST_PER_MIN", "0.0025")))
    temperature: float = field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.7")))
    timeout_s: int = field(default_factory=lambda: int(os.getenv("GEMINI_TIMEOUT_S", "120")))
    prompt_version: str = field(default_factory=lambda: os.getenv("PROMPT_VERSION", "v2"))
    long_silence_s: float = field(default_factory=lambda: float(os.getenv("LONG_SILENCE_S", "10")))
    use_vertex: bool = field(default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"))

    def price(self) -> dict:
        return PRICING.get(self.model, UNKNOWN_MODEL_PRICE)


def estimate_call_cost(cfg: Config, duration_s: float, samples: int,
                       prompt_tokens: int = PROMPT_TOKENS, out_tokens_per_sample: int | None = None) -> float:
    """Current Flash-Lite models reject candidate_count>1, so each sample is a separate request and
    the audio + prompt are billed once per sample."""
    if out_tokens_per_sample is None:
        out_tokens_per_sample = OUTPUT_TOKENS_COMPACT if duration_s < COMPACT_BELOW_S else OUTPUT_TOKENS_PER_SAMPLE
    p = cfg.price()
    per_sample = (duration_s * AUDIO_TOKENS_PER_SEC * p["audio_in"] + prompt_tokens * p["text_in"]
                  + out_tokens_per_sample * p["out"]) / 1e6
    return samples * per_sample


def choose_samples(cfg: Config, duration_s: float) -> int:
    """Largest k <= max_samples whose projected cost/min stays under target; at least 1."""
    minutes = max(duration_s, 1.0) / 60
    for k in range(cfg.max_samples, 0, -1):
        if estimate_call_cost(cfg, duration_s, k) / minutes <= cfg.target_cost_per_min:
            return k
    return 1
