"""Audio decoding / encoding via ffmpeg. All functions raise AudioError with a human-readable reason."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".flac", ".webm", ".wma", ".amr"}
SR = 16000
MAX_DURATION_S = 3 * 3600


class AudioError(Exception):
    pass


def _clean(stderr: str, path: Path) -> str:
    """ffmpeg messages without server paths or memory addresses, truncated."""
    import re

    msg = stderr.replace(str(path), path.name).replace(str(path.parent) + "/", "")
    msg = re.sub(r"\[[\w-]+ @ 0x[0-9a-f]+\] ?", "", msg)
    return " ".join(msg.split())[:200]


@dataclass
class Audio:
    path: Path
    duration_s: float
    channels: int
    source_sr: int
    codec: str
    stereo: np.ndarray  # (n, 2) float32 at SR; duplicated mono when source is mono
    mono: np.ndarray  # (n,) float32 at SR


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTS


def probe(path: Path) -> dict:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioError("ffprobe timed out") from e
    if r.returncode != 0:
        raise AudioError(f"unreadable or corrupt audio ({_clean(r.stderr, path) or 'ffprobe failed'})")
    info = json.loads(r.stdout or "{}")
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioError("file contains no audio stream")
    return {"format": info.get("format", {}), "stream": streams[0]}


def load(path: str | Path) -> Audio:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise AudioError("file is empty or missing")
    if not is_supported(path.name):
        raise AudioError(f"unsupported file extension '{path.suffix}'")
    meta = probe(path)
    st = meta["stream"]
    try:
        dur = float(meta["format"].get("duration") or st.get("duration") or 0)
    except ValueError:
        dur = 0.0
    if dur > MAX_DURATION_S:
        raise AudioError(f"audio too long ({dur:.0f}s > {MAX_DURATION_S}s)")
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
            capture_output=True, timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioError("decode timed out") from e
    if r.returncode != 0 or not r.stdout:
        raise AudioError(f"decode failed ({_clean(r.stderr.decode(errors='ignore'), path)})")
    x = np.frombuffer(r.stdout, dtype=np.float32)
    x = x[: len(x) // 2 * 2].reshape(-1, 2).copy()
    if len(x) < SR * 0.5:
        raise AudioError("audio shorter than 0.5 s")
    return Audio(
        path=path,
        duration_s=len(x) / SR,
        channels=int(st.get("channels", 1)),
        source_sr=int(st.get("sample_rate", 0) or 0),
        codec=str(st.get("codec_name", "")),
        stereo=x,
        mono=x.mean(axis=1),
    )


INLINE_LIMIT_BYTES = 15 * 1024 * 1024  # Gemini inline requests are capped at 20 MB


def encode_for_api(mono: np.ndarray) -> tuple[bytes, str]:
    """Encode 16 kHz mono audio for the API. Lossless FLAC by default: token cost depends only on
    duration, and re-encoding already-compressed calls with a lossy codec adds artifacts that the
    model can mistake for line static (and that differ between libopus builds). Falls back to
    Opus only when FLAC would exceed the inline request size (roughly > 30 min of audio)."""
    pcm = np.clip(mono, -1, 1).astype(np.float32).tobytes()

    def run(args: list[str]) -> bytes:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                            *args, "-"], input=pcm, capture_output=True, timeout=300)
        if r.returncode != 0:
            raise AudioError(f"encode failed ({r.stderr.decode(errors='ignore')[:200]})")
        return r.stdout

    data = run(["-c:a", "flac", "-sample_fmt", "s16", "-f", "flac"])
    if len(data) <= INLINE_LIMIT_BYTES:
        return data, "audio/flac"
    return run(["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"]), "audio/ogg"
