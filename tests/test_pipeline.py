import pytest

from analyzer import audio_io, fusion
from analyzer.config import Config, choose_samples, estimate_call_cost
from analyzer.pipeline import analyze_file
from tests.conftest import silence, tone
import numpy as np


def test_corrupt_file_raises_readable_error(tmp_path):
    p = tmp_path / "bad.mp3"
    p.write_bytes(b"not audio at all" * 100)
    with pytest.raises(audio_io.AudioError):
        audio_io.load(p)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(audio_io.AudioError, match="unsupported"):
        audio_io.load(p)


def test_dsp_backend_end_to_end(make_wav):
    p = make_wav("s.wav", np.concatenate([tone(3), silence(12), tone(3)]), channels=2)
    pred, meta = analyze_file(p, Config(backend="dsp"))
    assert pred["long_silence_present"] is True
    assert meta["cost_usd"] == 0


def test_budget_guard_never_exceeds_ceiling():
    cfg = Config(model="gemini-2.5-flash-lite", max_samples=3)
    for d in (5, 15, 30, 60, 180, 900):
        k = choose_samples(cfg, d)
        assert estimate_call_cost(cfg, d, k) / (max(d, 1) / 60) <= 0.003 or k == 1


def test_fusion_majority_and_tiebreak():
    cfg = Config()
    feats = {"longest_interior_gap_s": 2.0, "clipping_frac": 0, "dropouts": 0, "stereo_split": False}
    c = [{"emotional_tone": "upset", "emotional_intensity": "high", "background_noise_present": True,
          "background_noise_type": "television", "background_noise_severity": "medium"},
         {"emotional_tone": "frustrated", "emotional_intensity": "medium", "background_noise_present": True,
          "background_noise_type": "TV", "background_noise_severity": "low"},
         {"emotional_tone": "upset", "emotional_intensity": "high", "background_noise_present": False}]
    pred, diag = fusion.fuse(c, feats, cfg)
    assert pred["emotional_tone"] == "upset" and pred["emotional_intensity"] == "high"
    assert pred["background_noise_type"] == "TV"
    assert diag["tone_agreement"] == pytest.approx(0.67, abs=0.01)
    assert 0.05 <= pred["confidence"] <= 0.95
