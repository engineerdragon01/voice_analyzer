import subprocess
from pathlib import Path

import numpy as np
import pytest

SR = 16000


def tone(sec, f=220.0, amp=0.3, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    # Speech-like: harmonic stack with 4 Hz amplitude modulation
    s = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 6))
    return (amp * s / 2.3 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)).astype(np.float32)


def silence(sec, level=1e-4, sr=SR, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(sec * sr)) * level).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr=SR, channels=1):
    data = x if channels == 1 else np.repeat(x[:, None], 2, axis=1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(channels),
                    "-i", "-", str(path)], input=data.astype(np.float32).tobytes(), check=True)
    return path


@pytest.fixture
def make_wav(tmp_path):
    def _make(name, x, channels=1):
        return write_wav(tmp_path / name, x, channels=channels)
    return _make
