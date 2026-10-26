"""Runtime configuration (env-driven) and the pricing table used for cost accounting."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# USD per 1M tokens (paid tier, standard). Verify at https://ai.google.dev/gemini-api/docs/pricing
PRICING = {
    "gemini-2.5-flash-lite": {"audio_in": 0.30, "text_in": 0.10, "out": 0.40},
    "gemini-3.5-flash-lite": {"audio_in": 0.30, "text_in": 0.30, "out": 2.50},
    "gemini-3.1-flash-lite": {"audio_in": 0.50, "text_in": 0.25, "out": 1.50},
    "gemini-2.5-flash": {"audio_in": 1.00, "text_in": 0.30, "out": 2.50},
}
AUDIO_TOKENS_PER_SEC = 32  # Gemini audio tokenization rate
COST_CEILING_PER_MIN = 0.003


@dataclass
class Config:
    backend: str = field(default_factory=lambda: os.getenv("ANALYZER_BACKEND", "hybrid"))  # hybrid|gemini|dsp
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    max_samples: int = field(default_factory=lambda: int(os.getenv("GEMINI_SAMPLES", "3")))
    # Budget the per-clip sample count against this (leaves headroom under the $0.003 ceiling).
    target_cost_per_min: float = field(default_factory=lambda: float(os.getenv("TARGET_COST_PER_MIN", "0.0025")))
    temperature: float = field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.6")))
    timeout_s: int = field(default_factory=lambda: int(os.getenv("GEMINI_TIMEOUT_S", "120")))
    long_silence_s: float = field(default_factory=lambda: float(os.getenv("LONG_SILENCE_S", "10")))
    use_vertex: bool = field(default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"))

    def price(self) -> dict:
        return PRICING.get(self.model, PRICING["gemini-2.5-flash"])  # unknown model: assume pricier


def estimate_call_cost(cfg: Config, duration_s: float, samples: int,
                       prompt_tokens: int = 1400, out_tokens_per_sample: int = 160) -> float:
    p = cfg.price()
    audio = duration_s * AUDIO_TOKENS_PER_SEC * p["audio_in"] / 1e6
    text = prompt_tokens * p["text_in"] / 1e6
    out = samples * out_tokens_per_sample * p["out"] / 1e6
    return audio + text + out


def choose_samples(cfg: Config, duration_s: float) -> int:
    """Largest k <= max_samples whose projected cost/min stays under target; at least 1."""
    minutes = max(duration_s, 1.0) / 60
    for k in range(cfg.max_samples, 0, -1):
        if estimate_call_cost(cfg, duration_s, k) / minutes <= cfg.target_cost_per_min:
            return k
    return 1
