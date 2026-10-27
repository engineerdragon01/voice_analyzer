"""Audio decoding / encoding via ffmpeg. All functions raise AudioError with a human-readable reason."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".flac", ".webm", ".wma", ".amr"}
SR = 16000
# Peak memory is roughly 10-15 MB per audio-minute; 15 min keeps a single file well inside a 512 MB
# instance. Raise MAX_AUDIO_MINUTES on bigger instances.
MAX_DURATION_S = int(float(os.getenv("MAX_AUDIO_MINUTES", "15")) * 60)
CHUNK = SR * 30  # samples processed per chunk when deriving arrays (bounds temporary memory)


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
    mono: np.ndarray  # (n,) float32 at SR
    channel_stats: dict  # correlation / difference level between L and R (see channel_stats())
    stereo: np.ndarray | None  # (n, 2) float32, kept only when the channels carry different speakers


class _ChannelAccumulator:
    """Streaming L/R statistics (float64 sums) so stereo never has to be held in full."""

    def __init__(self) -> None:
        self.n = 0
        self.sums = np.zeros(9)  # L, R, LL, RR, LR, D, DD, M, MM

    def add(self, lr: np.ndarray, mono: np.ndarray) -> None:
        lr = lr.astype(np.float64)
        L, R = lr[:, 0], lr[:, 1]
        d = L - R
        m = mono.astype(np.float64)
        self.sums += [L.sum(), R.sum(), (L * L).sum(), (R * R).sum(), (L * R).sum(),
                      d.sum(), (d * d).sum(), m.sum(), (m * m).sum()]
        self.n += len(m)

    def result(self) -> dict:
        n = max(self.n, 1)
        sl, sr_, sll, srr, slr, sd, sdd, sm, smm = self.sums
        var_l = sll / n - (sl / n) ** 2
        var_r = srr / n - (sr_ / n) ** 2
        cov = slr / n - (sl / n) * (sr_ / n)
        corr = float(cov / np.sqrt(var_l * var_r)) if var_l > 0 and var_r > 0 else 1.0
        std_d = np.sqrt(max(sdd / n - (sd / n) ** 2, 0.0))
        std_m = np.sqrt(max(smm / n - (sm / n) ** 2, 0.0))
        diff_db = float(20 * np.log10(std_d + 1e-10) - 20 * np.log10(std_m + 1e-10))
        return {"correlation": corr, "diff_db": diff_db, "split": bool(corr < 0.6 and diff_db > -20)}


def channel_stats(stereo: np.ndarray, mono: np.ndarray) -> dict:
    """L/R correlation and the level of L-R relative to the mix, accumulated in chunks (float64)."""
    acc = _ChannelAccumulator()
    for i in range(0, len(mono), CHUNK):
        acc.add(stereo[i:i + CHUNK], mono[i:i + CHUNK])
    return acc.result()


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


def _decode_stream(path: Path, timeout_s: int = 300) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Decode to 16 kHz float32 while streaming ffmpeg's output in 30 s chunks: builds the mono mix and the
    channel statistics on the fly, and keeps stereo samples only when the channels actually differ
    (dual-mono calls are detected on the first chunk). Peak memory is ~mono size instead of 3x."""
    import tempfile
    import time

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
            stdout=subprocess.PIPE, stderr=err,
        )
        t0 = time.time()
        acc = _ChannelAccumulator()
        mono_chunks: list[np.ndarray] = []
        stereo_chunks: list[np.ndarray] | None = []
        want = CHUNK * 8  # bytes per chunk (2 channels x float32)
        pending = b""
        while True:
            if time.time() - t0 > timeout_s:
                proc.kill()
                raise AudioError("decode timed out")
            data = proc.stdout.read(want - len(pending))
            if data:
                pending += data
                if len(pending) < want:
                    continue
            if not pending:
                break
            usable = len(pending) // 8 * 8
            lr = np.frombuffer(pending[:usable], dtype=np.float32).reshape(-1, 2)
            pending = pending[usable:] if not data else b""
            m = lr.mean(axis=1)
            acc.add(lr, m)
            mono_chunks.append(m)
            if stereo_chunks is not None:
                if len(mono_chunks) == 1 and len(m) >= SR * 10 and acc.result()["diff_db"] < -40:
                    stereo_chunks = None  # L == R: no point keeping stereo
                else:
                    stereo_chunks.append(lr)
            if not data:
                break
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0 or not mono_chunks:
            err.seek(0)
            raise AudioError(f"decode failed ({_clean(err.read().decode(errors='ignore'), path)})")
    mono = np.concatenate(mono_chunks) if len(mono_chunks) > 1 else mono_chunks[0].copy()
    del mono_chunks
    if len(mono) < SR * 0.5:
        raise AudioError("audio shorter than 0.5 s")
    stats = acc.result()
    stereo = None
    if stats["split"] and stereo_chunks:
        stereo = np.concatenate(stereo_chunks)
    return mono, stereo, stats


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
    mono, stereo, stats = _decode_stream(path)
    return Audio(
        path=path,
        duration_s=len(mono) / SR,
        channels=int(st.get("channels", 1)),
        source_sr=int(st.get("sample_rate", 0) or 0),
        codec=str(st.get("codec_name", "")),
        mono=mono,
        channel_stats=stats,
        stereo=stereo,
    )


INLINE_LIMIT_BYTES = 15 * 1024 * 1024  # Gemini inline requests are capped at 20 MB


def encode_for_api(mono: np.ndarray) -> tuple[bytes, str]:
    """Encode 16 kHz mono audio for the API. Lossless FLAC by default: token cost depends only on
    duration, and re-encoding already-compressed calls with a lossy codec adds artifacts that the
    model can mistake for line static (and that differ between libopus builds). Falls back to
    Opus only when FLAC would exceed the inline request size (roughly > 30 min of audio)."""
    # 16-bit PCM (what FLAC stores anyway), converted in chunks: half the memory of a float32 copy.
    s16 = np.empty(len(mono), dtype=np.int16)
    for i in range(0, len(mono), CHUNK):
        s16[i:i + CHUNK] = np.rint(np.clip(mono[i:i + CHUNK] * 32768.0, -32768, 32767))
    pcm = s16.tobytes()
    del s16

    def run(args: list[str]) -> bytes:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                            *args, "-"], input=pcm, capture_output=True, timeout=300)
        if r.returncode != 0:
            raise AudioError(f"encode failed ({r.stderr.decode(errors='ignore')[:200]})")
        return r.stdout

    data = run(["-c:a", "flac", "-f", "flac"])
    if len(data) <= INLINE_LIMIT_BYTES:
        return data, "audio/flac"
    return run(["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"]), "audio/ogg"
