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
CHUNK_FRAMES = 1500  # frames per chunk (30 s at 20 ms) — bounds temporary arrays for long calls


def _frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    return x[: n * HOP].reshape(n, HOP)


def _frame_power(x: np.ndarray, win: int) -> np.ndarray:
    """Mean square per window of `win` samples, computed chunk by chunk in float64."""
    n = len(x) // win
    out = np.empty(n)
    step = max(1, (CHUNK_FRAMES * HOP) // win)
    for i in range(0, n, step):
        j = min(n, i + step)
        seg = x[i * win: j * win].reshape(j - i, win).astype(np.float64)
        out[i:j] = np.mean(seg * seg, axis=1)
    return out


def frame_db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(_frame_power(x, HOP) + EPS)


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
    idx = np.flatnonzero(mask_frames[: len(f)])
    win = np.hanning(FRAME)
    acc = np.zeros(513)
    for i in range(0, len(idx), CHUNK_FRAMES):
        sel_chunk = f[idx[i:i + CHUNK_FRAMES]].astype(np.float64)
        acc += (np.abs(np.fft.rfft(sel_chunk * win, n=1024, axis=1)) ** 2).sum(axis=0)
    freqs = np.fft.rfftfreq(1024, 1 / SR)
    mean_spec = acc / len(idx) + EPS
    flat = float(np.exp(np.mean(np.log(mean_spec[5:]))) / np.mean(mean_spec[5:]))
    centroid = float((freqs * mean_spec).sum() / mean_spec.sum())
    cum = np.cumsum(mean_spec) / mean_spec.sum()
    rolloff = float(freqs[np.searchsorted(cum, 0.95)])
    # Mains hum: energy at 50/60 Hz harmonics relative to neighbours (longer FFT for resolution).
    sig = f[idx[: (SR * 20) // HOP]].reshape(-1)
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


def features_for(audio) -> dict:
    """Features for an audio_io.Audio (uses the channel statistics computed while loading)."""
    return features(audio.stereo, audio.mono, audio.channel_stats)


def features(stereo: np.ndarray | None, mono: np.ndarray, channel_stats: dict | None = None) -> dict:
    from .audio_io import channel_stats as _channel_stats

    dur = len(mono) / SR
    cs = channel_stats or _channel_stats(stereo, mono)
    corr = cs["correlation"]
    stereo_split = cs["split"] and stereo is not None

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
    abs_mono = np.abs(mono)
    p999, p9999 = np.percentile(abs_mono, [99.9, 99.99], overwrite_input=True)
    clip_frac = _clipping_fraction(abs_mono, float(p9999))
    kurt = speech_kurtosis(mono, speech)
    dropouts = _dropouts(mono, speech, noise_db)
    sq_sum = sum(float(np.dot(c, c)) for c in (mono[i:i + CHUNK_FRAMES * HOP].astype(np.float64)
                                              for i in range(0, len(mono), CHUNK_FRAMES * HOP)))
    rms_dbfs = float(20 * np.log10(np.sqrt(sq_sum / len(mono)) + EPS))

    ns = _spectral_stats(mono, nonspeech)
    sp = _spectral_stats(mono, speech)
    harm_ns = _harmonicity(mono, nonspeech)
    clicks = click_rate(mono, speech)

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
        "click_rate_per_min": None if clicks is None else round(clicks, 1),
        "speech_rolloff95_hz": None if sp["rolloff95_hz"] is None else round(sp["rolloff95_hz"]),
        "clipping_frac": round(clip_frac, 5),
        "speech_kurtosis": None if kurt is None else round(kurt, 2),
        "dropouts": int(dropouts),
        "rms_dbfs": round(rms_dbfs, 1),
        "peak_p999": round(float(p999), 3),
    }
    out["clipping_suspected"] = bool(
        out["clipping_frac"] > 0.01
        or (kurt is not None and kurt < 2.5 and out["peak_p999"] > 0.5)
    )
    del abs_mono
    if stereo_split:
        out.update(_overlap_stereo(np.ascontiguousarray(stereo[:, 0]), np.ascontiguousarray(stereo[:, 1])))
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


def _clipping_fraction(abs_x: np.ndarray, peak: float) -> float:
    """Share of samples piled up near the peak (direct measure, reliable on lossless input).
    abs_x is |signal| (any order); peak is its 99.99th percentile."""
    if peak < 0.3:  # quiet recordings cannot be clipped
        return 0.0
    return float(np.count_nonzero(abs_x >= 0.9 * peak) / len(abs_x))


def speech_kurtosis(x: np.ndarray, speech_mask: np.ndarray) -> float | None:
    """Kurtosis of speech samples. Natural speech is peaky (kurtosis ~5-30); hard clipping flattens
    the distribution toward ~1.5-3. Unlike plateau counting, this survives Opus/MP3 re-encoding."""
    f = _frames(x)[: len(speech_mask)]
    s2 = s4 = 0.0
    count = 0
    for i in range(0, len(f), CHUNK_FRAMES):
        sel = f[i:i + CHUNK_FRAMES][speech_mask[i:i + CHUNK_FRAMES]].astype(np.float64)
        sq = sel * sel
        s2 += sq.sum()
        s4 += (sq * sq).sum()
        count += sel.size
    if count < SR:
        return None
    v = s2 / count
    return float((s4 / count) / (v**2 + EPS))


def _dropouts(x: np.ndarray, speech_mask: np.ndarray, noise_db: float) -> int:
    """Packet-loss style gaps: 20-250 ms stretches inside speech regions that fall below the
    background level and are >=35 dB under the speech immediately before and after (abrupt
    drop-and-recover). Natural pauses decay gradually or stay near the background level."""
    hop = 160
    n = len(speech_mask) * (HOP // hop)
    e = 10 * np.log10(_frame_power(x[: n * hop], hop) + EPS)
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


def click_rate(x: np.ndarray, speech_mask: np.ndarray) -> float | None:
    """Line static / crackle detector: impulsive high-frequency (>3 kHz) events per minute of non-speech.
    Each 2 ms window is compared with a running median (~100 ms); jumps of >12 dB in non-speech regions
    count as clicks. Clean calls measured <=35/min, the static call 180/min, synthetic static 940+/min.
    Deterministic and codec-stable, unlike the audio LLM's judgement of faint static."""
    win = 32  # 2 ms at 16 kHz
    sos = signal.butter(4, 3000, "hp", fs=SR, output="sos")
    n = len(x) // win
    if n < 200:
        return None
    zi = np.zeros((sos.shape[0], 2))
    step = CHUNK_FRAMES * HOP  # multiple of win
    power = np.empty(n)
    for i in range(0, n * win, step):
        y, zi = signal.sosfilt(sos, x[i:min(i + step, n * win)].astype(np.float64), zi=zi)
        k = i // win
        power[k:k + len(y) // win] = np.mean((y * y).reshape(-1, win), axis=1)
    e = 10 * np.log10(power + EPS)
    nonspeech = ~np.repeat(speech_mask, HOP // win)[:n]
    if len(nonspeech) < n:
        nonspeech = np.pad(nonspeech, (0, n - len(nonspeech)), constant_values=True)
    ns_minutes = nonspeech.mean() * len(x) / SR / 60
    if ns_minutes < 2 / 60:  # under ~2 s of non-speech: not enough evidence
        return None
    jumps = (e - signal.medfilt(e, 51) > 12) & nonspeech
    events = int((np.diff(np.concatenate([[0], jumps.astype(np.int8), [0]])) == 1).sum())
    return events / ns_minutes
