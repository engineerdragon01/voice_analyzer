import numpy as np

from analyzer import dsp
from tests.conftest import silence, tone


def feats(x):
    return dsp.features(np.stack([x, x], 1), x)


def test_long_interior_silence_measured():
    x = np.concatenate([tone(4), silence(12), tone(4)])
    f = feats(x)
    assert 11 <= f["longest_interior_gap_s"] <= 13
    assert not f["stereo_split"]


def test_short_pauses_not_long():
    x = np.concatenate([tone(3), silence(1.5), tone(3), silence(2), tone(3)])
    assert feats(x)["longest_interior_gap_s"] < 3


def test_leading_silence_excluded_from_interior():
    x = np.concatenate([silence(15), tone(5)])
    f = feats(x)
    assert f["longest_interior_gap_s"] < 1
    assert f["leading_silence_s"] > 13


def test_snr_drops_with_noise():
    clean = np.concatenate([tone(4), silence(2), tone(4)])
    noisy = clean + np.random.default_rng(1).standard_normal(len(clean)).astype(np.float32) * 0.03
    assert feats(clean)["snr_db"] > feats(noisy)["snr_db"] + 15


def test_clipping_detected():
    x = np.clip(tone(5, amp=3.0), -1, 1)
    f = feats(x)
    assert f["clipping_frac"] > 0.05 and f["clipping_suspected"]


def test_dropouts_detected():
    x = tone(6)
    for s in (1.0, 2.0, 3.0, 4.0):
        i = int(s * 16000)
        x[i:i + 800] = 0
    assert feats(x)["dropouts"] >= 4


def test_hum_detected():
    t = np.arange(16000 * 6) / 16000
    x = np.concatenate([tone(3), silence(6), tone(3)])
    hum = np.zeros_like(x)
    hum[3 * 16000:9 * 16000] = 0.02 * np.sin(2 * np.pi * 60 * t)
    assert feats(x + hum)["noise_hum_db"] > 6


def test_stereo_split_overlap():
    a = np.concatenate([tone(3, 200), silence(3), tone(3, 200)])
    b = np.concatenate([silence(2), tone(3, 300), silence(4, seed=2)])
    f = dsp.features(np.stack([a, b], 1), (a + b) / 2)
    assert f["stereo_split"]
    assert 0.5 <= f["overlap_s"] <= 1.6


def test_click_rate_flags_crackle_not_clean():
    rng = np.random.default_rng(3)
    clean = np.concatenate([tone(3), silence(6), tone(3)])
    crackle = clean.copy()
    idx = rng.integers(3 * 16000, 9 * 16000, 120)
    crackle[idx] += rng.choice([-0.03, 0.03], len(idx))  # impulses well below speech level
    assert (feats(clean)["click_rate_per_min"] or 0) < 90
    assert feats(crackle)["click_rate_per_min"] > 90
