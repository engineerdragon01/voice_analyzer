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
        raise AudioError(f"unreadable or corrupt audio ({r.stderr.strip()[:200] or 'ffprobe failed'})")
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
        raise AudioError(f"decode failed ({r.stderr.decode(errors='ignore').strip()[:200]})")
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


def encode_for_api(mono: np.ndarray, bitrate: str = "32k") -> bytes:
    """Compress 16 kHz mono float audio to Ogg/Opus to cut upload size (token cost is duration-based)."""
    pcm = np.clip(mono, -1, 1).astype(np.float32).tobytes()
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:a", "libopus", "-b:a", bitrate, "-application", "voip", "-f", "ogg", "-"],
        input=pcm, capture_output=True, timeout=300,
    )
    if r.returncode != 0:
        raise AudioError(f"encode failed ({r.stderr.decode(errors='ignore')[:200]})")
    return r.stdout
