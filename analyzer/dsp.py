"""Deterministic acoustic features (numpy/scipy only, CPU, ~real-time x200).

Everything here is objective measurement; interpretation happens in fusion.py and in the LLM prompt.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 16000
FRAME = 320  # 20 ms
HOP = 320
EPS = 1e-10


def _frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    return x[: n * HOP].reshape(n, HOP)


def frame_db(x: np.ndarray) -> np.ndarray:
    f = _frames(x)
    return 10 * np.log10(np.mean(f**2, axis=1) + EPS)


def _otsu_threshold(v: np.ndarray) -> float:
    """Two-class split of frame energies (speech vs non-speech)."""
    hist, edges = np.histogram(v, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist)
    mu = np.cumsum(hist * centers)
    total = w[-1]
    w2 = total - w
    valid = (w > 0) & (w2 > 0)
    m1 = np.where(valid, mu / np.maximum(w, 1), 0)
    m2 = np.where(valid, (mu[-1] - mu) / np.maximum(w2, 1), 0)
    between = np.where(valid, w * w2 * (m1 - m2) ** 2, 0)
    return float(centers[int(np.argmax(between))])


def _smooth_bool(mask: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    """Close gaps shorter than min_off, then drop on-runs shorter than min_on (frames)."""
    m = mask.copy()
    for target, min_len in ((False, min_off), (True, min_on)):
        runs = _runs(m, target)
        for s, e in runs:
            if e - s < min_len:
                m[s:e] = not target
    return m


def _runs(mask: np.ndarray, value: bool = True) -> list[tuple[int, int]]:
    m = (mask == value).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def vad(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Energy VAD with adaptive threshold. Returns per-frame speech mask and level stats."""
    db = frame_db(x)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    thr = _otsu_threshold(db)
    # Keep threshold sane: at least 9 dB above floor, at most 25 dB below speech peak.
    thr = float(np.clip(thr, floor + 9, max(floor + 9, peak - 6)))
    if peak - floor < 15:
        # Little dynamic range: either continuous speech or continuous silence. Use an absolute gate.
        thr = max(peak - 15, -55.0)
    raw = db > thr
    mask = _smooth_bool(raw, min_on=5, min_off=15)  # 100 ms min speech, bridge 300 ms gaps
    return mask, {"floor_db": floor, "peak_db": peak, "threshold_db": thr, "frame_db": db}


def _spectral_stats(x: np.ndarray, mask_frames: np.ndarray) -> dict:
    """Spectral features over selected frames (e.g. non-speech) for noise characterization."""
    f = _frames(x)
    if mask_frames.sum() < 10:
        return {"flatness": None, "hum_ratio_db": None, "centroid_hz": None, "rolloff95_hz": None}
    sel = f[mask_frames[: len(f)]]
    win = np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(sel * win, n=1024, axis=1)) ** 2 + EPS
    freqs = np.fft.rfftfreq(1024, 1 / SR)
    mean_spec = spec.mean(axis=0)
    flat = float(np.exp(np.mean(np.log(mean_spec[5:]))) / np.mean(mean_spec[5:]))
    centroid = float((freqs * mean_spec).sum() / mean_spec.sum())
    cum = np.cumsum(mean_spec) / mean_spec.sum()
    rolloff = float(freqs[np.searchsorted(cum, 0.95)])
    # Mains hum: energy at 50/60 Hz harmonics relative to neighbours (longer FFT for resolution).
    sig = sel.reshape(-1)[: SR * 20]
    hum_db = None
    if len(sig) >= SR:
        fz, pz = signal.welch(sig, SR, nperseg=SR)
        def peak_ratio(base):
            vals = []
            for h in (1, 2, 3):
                fc = base * h
                i = int(round(fc))
                band = pz[max(i - 8, 0): i + 9]
                vals.append(10 * np.log10(pz[i] / (np.median(band) + EPS) + EPS))
            return float(np.mean(vals))
        hum_db = max(peak_ratio(50), peak_ratio(60))
    return {"flatness": flat, "hum_ratio_db": hum_db, "centroid_hz": centroid, "rolloff95_hz": rolloff}


def _harmonicity(x: np.ndarray, mask_frames: np.ndarray) -> float | None:
    """Mean normalized autocorrelation peak (80-400 Hz lag) in selected 40 ms windows.
    High values in non-speech regions suggest music/TV/other voices rather than stationary hiss."""
    f = _frames(x)
    idx = np.where(mask_frames[: len(f) - 1])[0]
    if len(idx) < 10:
        return None
    idx = idx[:: max(1, len(idx) // 400)]
    vals = []
    for i in idx:
        w = np.concatenate([f[i], f[i + 1]])
        w = w - w.mean()
        if np.std(w) < 1e-5:
            continue
        ac = np.correlate(w, w, "full")[len(w) - 1:]
        ac = ac / (ac[0] + EPS)
        vals.append(float(ac[40:200].max()))
    return float(np.mean(vals)) if vals else None


def features(stereo: np.ndarray, mono: np.ndarray) -> dict:
    dur = len(mono) / SR
    L, R = stereo[:, 0], stereo[:, 1]
    corr = float(np.corrcoef(L, R)[0, 1]) if np.std(L) > 0 and np.std(R) > 0 else 1.0
    diff_db = float(20 * np.log10(np.std(L - R) + EPS) - 20 * np.log10(np.std(mono) + EPS))
    stereo_split = bool(corr < 0.6 and diff_db > -20)

    speech, lv = vad(mono)
    db = lv["frame_db"]
    n = len(speech)
    fps = SR / HOP
    speech_ratio = float(speech.mean()) if n else 0.0

    # Silence / dead air: non-speech runs that are also quiet (near the floor), excluding call edges.
    quiet = db < lv["floor_db"] + 6
    nonspeech_runs = _runs(speech, False)
    interior = [(s, e) for s, e in nonspeech_runs if s > 0 and e < n]
    longest_gap = max(((e - s) / fps for s, e in interior), default=0.0)
    quiet_interior = [(s, e) for s, e in _runs(quiet & ~speech, True) if s > 0 and e < n]
    longest_quiet = max(((e - s) / fps for s, e in quiet_interior), default=0.0)
    lead = (nonspeech_runs[0][1] / fps) if nonspeech_runs and nonspeech_runs[0][0] == 0 else 0.0
    trail = ((n - nonspeech_runs[-1][0]) / fps) if nonspeech_runs and nonspeech_runs[-1][1] == n else 0.0

    # Levels & SNR
    speech_db = float(np.median(db[speech])) if speech.any() else float(lv["peak_db"])
    noise_db = float(np.median(db[~speech])) if (~speech).any() else float(lv["floor_db"])
    snr = speech_db - noise_db
    nonspeech = ~speech
    # Burstiness of the background: fraction of non-speech frames >10 dB above the noise median
    # (clicks, crackle, bangs) — the VAD's min-duration smoothing leaves these in the non-speech set.
    burst_frac = float(np.mean(db[nonspeech] > noise_db + 10)) if nonspeech.sum() > 10 else 0.0
    noise_var_db = float(np.std(db[nonspeech])) if nonspeech.sum() > 10 else 0.0

    # Quality (codec-robust: lossy re-encoding smears clipped plateaus and digital zeros)
    clip_frac = _clipping_fraction(mono)
    kurt = speech_kurtosis(mono, speech)
    dropouts = _dropouts(mono, speech, noise_db)
    rms_dbfs = float(20 * np.log10(np.sqrt(np.mean(mono**2)) + EPS))

    ns = _spectral_stats(mono, nonspeech)
    sp = _spectral_stats(mono, speech)
    harm_ns = _harmonicity(mono, nonspeech)

    out = {
        "duration_s": round(dur, 2),
        "stereo_split": stereo_split,
        "channel_correlation": round(corr, 3),
        "speech_ratio": round(speech_ratio, 3),
        "longest_interior_gap_s": round(longest_gap, 2),
        "longest_interior_quiet_s": round(longest_quiet, 2),
        "leading_silence_s": round(lead, 2),
        "trailing_silence_s": round(trail, 2),
        "speech_level_db": round(speech_db, 1),
        "noise_level_db": round(noise_db, 1),
        "snr_db": round(snr, 1),
        "noise_burst_frac": round(burst_frac, 3),
        "noise_level_std_db": round(noise_var_db, 2),
        "noise_flatness": None if ns["flatness"] is None else round(ns["flatness"], 3),
        "noise_hum_db": None if ns["hum_ratio_db"] is None else round(ns["hum_ratio_db"], 1),
        "noise_centroid_hz": None if ns["centroid_hz"] is None else round(ns["centroid_hz"]),
        "noise_harmonicity": None if harm_ns is None else round(harm_ns, 3),
        "speech_rolloff95_hz": None if sp["rolloff95_hz"] is None else round(sp["rolloff95_hz"]),
        "clipping_frac": round(clip_frac, 5),
        "speech_kurtosis": None if kurt is None else round(kurt, 2),
        "dropouts": int(dropouts),
        "rms_dbfs": round(rms_dbfs, 1),
        "peak_p999": round(float(np.percentile(np.abs(mono), 99.9)), 3),
    }
    out["clipping_suspected"] = bool(
        out["clipping_frac"] > 0.01
        or (kurt is not None and kurt < 2.5 and out["peak_p999"] > 0.5)
    )
    if stereo_split:
        out.update(_overlap_stereo(L, R))
    return out


def _overlap_stereo(L: np.ndarray, R: np.ndarray) -> dict:
    """Overlap when each speaker has their own channel: both channels voiced at once (>=300 ms runs)."""
    mL, _ = vad(L)
    mR, _ = vad(R)
    n = min(len(mL), len(mR))
    both = _smooth_bool(mL[:n] & mR[:n], min_on=15, min_off=5)
    fps = SR / HOP
    runs = _runs(both, True)
    total = sum(e - s for s, e in runs) / fps
    return {
        "overlap_s": round(total, 2),
        "overlap_events": len(runs),
        "overlap_longest_s": round(max(((e - s) / fps for s, e in runs), default=0.0), 2),
    }


def _clipping_fraction(x: np.ndarray) -> float:
    """Share of samples piled up near the peak (direct measure, reliable on lossless input)."""
    peak = float(np.percentile(np.abs(x), 99.99))
    if peak < 0.3:  # quiet recordings cannot be clipped
        return 0.0
    return float(np.mean(np.abs(x) >= 0.9 * peak))


def speech_kurtosis(x: np.ndarray, speech_mask: np.ndarray) -> float | None:
    """Kurtosis of speech samples. Natural speech is peaky (kurtosis ~5-30); hard clipping flattens
    the distribution toward ~1.5-3. Unlike plateau counting, this survives Opus/MP3 re-encoding."""
    m = np.repeat(speech_mask, HOP)
    sp = x[: len(m)][m]
    if len(sp) < SR:
        return None
    v = np.mean(sp**2)
    return float(np.mean(sp**4) / (v**2 + EPS))


def _dropouts(x: np.ndarray, speech_mask: np.ndarray, noise_db: float) -> int:
    """Packet-loss style gaps: 20-250 ms stretches inside speech regions that fall below the
    background level and are >=35 dB under the speech immediately before and after (abrupt
    drop-and-recover). Natural pauses decay gradually or stay near the background level."""
    hop = 160
    n = len(speech_mask) * (HOP // hop)
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + EPS)
    inside = np.repeat(speech_mask, HOP // hop)
    low = (e < noise_db - 6) & inside
    count = 0
    for s, t in _runs(low, True):
        if not 2 <= t - s <= 25 or s < 5 or t + 5 > n:
            continue
        gap = float(np.mean(e[s:t]))
        if e[s - 5:s].max() - gap >= 35 and e[t:t + 5].max() - gap >= 35:
            count += 1
    return count
