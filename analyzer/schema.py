"""Output schema defined by the AutoAce trial spec, plus a tolerant coercer."""
from __future__ import annotations

from typing import Any

TONES = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
INTENSITIES = ["low", "medium", "high"]
SEVERITIES = ["none", "low", "medium", "high"]
QUALITIES = ["clear", "slightly_impaired", "severely_impaired"]

FIELDS = [
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
    "confidence",
]

ENUMS = {
    "emotional_tone": TONES,
    "emotional_intensity": INTENSITIES,
    "background_noise_severity": SEVERITIES,
    "audio_quality": QUALITIES,
}
BOOLS = ["background_noise_present", "speaker_overlap_present", "long_silence_present"]

DEFAULTS: dict[str, Any] = {
    "emotional_tone": "neutral",
    "emotional_intensity": "low",
    "background_noise_present": False,
    "background_noise_type": "",
    "background_noise_severity": "none",
    "audio_quality": "clear",
    "speaker_overlap_present": False,
    "long_silence_present": False,
    "confidence": 0.5,
}

# JSON schema handed to Gemini as response_schema (OpenAPI subset).
LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_evidence": {"type": "string"},
        "emotional_tone": {"type": "string", "enum": TONES},
        "emotional_intensity": {"type": "string", "enum": INTENSITIES},
        "noise_evidence": {"type": "string"},
        "background_noise_present": {"type": "boolean"},
        "background_noise_type": {"type": "string"},
        "background_noise_severity": {"type": "string", "enum": SEVERITIES},
        "audio_quality": {"type": "string", "enum": QUALITIES},
        "speaker_overlap_present": {"type": "boolean"},
        "long_silence_present": {"type": "boolean"},
    },
    "required": [
        "customer_evidence",
        "emotional_tone",
        "emotional_intensity",
        "noise_evidence",
        "background_noise_present",
        "background_noise_type",
        "background_noise_severity",
        "audio_quality",
        "speaker_overlap_present",
        "long_silence_present",
    ],
    "propertyOrdering": [
        "customer_evidence",
        "emotional_tone",
        "emotional_intensity",
        "noise_evidence",
        "background_noise_present",
        "background_noise_type",
        "background_noise_severity",
        "audio_quality",
        "speaker_overlap_present",
        "long_silence_present",
    ],
}


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "y"):
            return True
        if s in ("false", "no", "0", "n", ""):
            return False
    return None


def _to_enum(v: Any, allowed: list[str]) -> str | None:
    if not isinstance(v, str):
        return None
    s = v.strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in allowed else None


def coerce(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a dict with exactly the schema fields, valid types, and internally consistent noise fields."""
    out: dict[str, Any] = {}
    for f, allowed in ENUMS.items():
        out[f] = _to_enum(raw.get(f), allowed) or DEFAULTS[f]
    for f in BOOLS:
        b = _to_bool(raw.get(f))
        out[f] = DEFAULTS[f] if b is None else b
    t = raw.get("background_noise_type")
    out["background_noise_type"] = t.strip() if isinstance(t, str) else ""
    try:
        c = float(raw.get("confidence", DEFAULTS["confidence"]))
    except (TypeError, ValueError):
        c = DEFAULTS["confidence"]
    out["confidence"] = round(min(1.0, max(0.0, c)), 2)

    # Noise fields must agree with each other.
    if not out["background_noise_present"]:
        out["background_noise_severity"] = "none"
        out["background_noise_type"] = ""
    elif out["background_noise_severity"] == "none":
        out["background_noise_severity"] = "low"
    if out["background_noise_present"] and not out["background_noise_type"]:
        out["background_noise_type"] = "unspecified noise"
    return {f: out[f] for f in FIELDS}


def parse_label(result_json: str | None) -> dict[str, Any] | None:
    """Parse a manifest result_json cell; None when empty or malformed."""
    import json

    if not result_json or not str(result_json).strip():
        return None
    try:
        obj = json.loads(result_json)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
