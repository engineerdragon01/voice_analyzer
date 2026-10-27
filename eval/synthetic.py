"""Synthetic validation for the objective fields (silence, noise presence, audio quality, overlap).

Derives clips from the provided calls with known, controlled transformations so DSP rules and the
hybrid system can be measured on more than 3 examples. Output stays under data/ (derived from
confidential audio, never committed).

    python -m eval.synthetic build data/provided data/synthetic
    python cli.py data/synthetic -o results/synthetic_dsp --backend dsp
    python -m eval.synthetic score data/synthetic/labels.csv results/synthetic_dsp.csv
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from analyzer import audio_io, dsp

SR = 16000
rng = np.random.default_rng(7)


def write(path: Path, x: np.ndarray) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-c:a", "libopus", "-b:a", "32k", str(path)],
                   input=np.clip(x, -1, 1).astype(np.float32).tobytes(), check=True)


def rms(x):
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


def pink(n):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    return np.fft.irfft(f, n).astype(np.float32)


def static_bursts(n):
    """Crackle: sparse high-amplitude impulses + short broadband bursts."""
    x = np.zeros(n, np.float32)
    for _ in range(int(n / SR * 6)):
        i = rng.integers(0, n - 800)
        x[i:i + rng.integers(40, 800)] += rng.standard_normal(1) * rng.standard_normal(1)[0] * 0.5
    x += (rng.random(n) < 0.002) * rng.standard_normal(n) * 0.8
    return x.astype(np.float32)


def hum(n):
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * 60 * h * t) / h for h in (1, 2, 3)).astype(np.float32)


def babble(n, sources):
    """Distant chatter/TV-like: several unrelated speech segments summed, low-passed slightly."""
    x = np.zeros(n, np.float32)
    for s in sources:
        st = rng.integers(0, max(1, len(s) - n))
        seg = s[st:st + n]
        x[: len(seg)] += seg
    return x


def mix_at_snr(speech, noise, snr_db, speech_mask):
    m = np.repeat(speech_mask, dsp.HOP)
    s_level = rms(speech[: len(m)][m]) if speech_mask.any() else rms(speech)
    gain = s_level / (rms(noise) * 10 ** (snr_db / 20))
    return speech + noise * gain


def build(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    calls = {p.name: audio_io.load(p).mono for p in sorted(src.glob("*.ogg"))}
    clean = calls["call_001.ogg"]  # labeled: no background noise, clear audio
    others = [v for k, v in calls.items() if k != "call_001.ogg"]
    rows = []

    def add(name, x, labels):
        write(out / f"{name}.ogg", x)
        rows.append((f"{name}.ogg", labels))

    # 1) Silence: insert near-floor dead air into each call at mid-point.
    for cname, x in calls.items():
        floor = x[np.abs(x) < np.percentile(np.abs(x), 5)]
        for gap in (0, 4, 15, 25):
            mid = len(x) // 2
            pad = rng.choice(floor, int(gap * SR)).astype(np.float32) if gap else np.zeros(0, np.float32)
            y = np.concatenate([x[:mid], pad, x[mid:]])
            add(f"sil_{gap:02d}s_{cname[:-4]}", y, {"long_silence_present": gap >= 10})

    # 2) Background noise on the clean call at several SNRs.
    mask, _ = dsp.vad(clean)
    n = len(clean)
    gens = {"white": lambda: rng.standard_normal(n).astype(np.float32), "pink": lambda: pink(n),
            "hum": lambda: hum(n), "static": lambda: static_bursts(n), "babble": lambda: babble(n, others)}
    for kind, g in gens.items():
        for snr in (5, 15, 25):
            add(f"noise_{kind}_{snr:02d}db", mix_at_snr(clean, g(), snr, mask),
                {"background_noise_present": True})
    for i in range(3):  # negatives: clean segments
        st = int(rng.integers(0, n // 3))
        add(f"noise_none_{i}", clean[st:], {"background_noise_present": False})

    # 3) Technical quality impairments on the clean call.
    y = np.clip(clean * 12, -1, 1)
    add("qual_clipped", y, {"audio_quality_impaired": True})
    y = clean.copy()
    for s in range(SR, n - SR, SR):  # 80 ms dropouts every second (packet loss)
        y[s:s + int(0.08 * SR)] = 0
    add("qual_dropouts", y, {"audio_quality_impaired": True})
    add("qual_quiet", clean * 0.01, {"audio_quality_impaired": True})
    add("qual_clean", clean, {"audio_quality_impaired": False})

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        for name, lab in rows:
            w.writerow([name, json.dumps(lab)])
    print(f"wrote {len(rows)} clips to {out}")


def score(labels_csv: Path, pred_csv: Path) -> dict:
    labels = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(open(labels_csv))}
    preds = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(open(pred_csv)) if r["result_json"]}
    groups: dict[str, list[tuple[bool, bool, str]]] = {}
    for name, lab in labels.items():
        p = preds.get(name)
        if p is None:
            continue
        for k, v in lab.items():
            if k == "audio_quality_impaired":
                pv = p["audio_quality"] != "clear"
            else:
                pv = p[k]
            groups.setdefault(k, []).append((v, pv, name))
    out = {}
    for k, items in groups.items():
        tp = sum(1 for t, p, _ in items if t and p)
        tn = sum(1 for t, p, _ in items if not t and not p)
        fp = [n for t, p, n in items if not t and p]
        fn = [n for t, p, n in items if t and not p]
        out[k] = {"n": len(items), "accuracy": round((tp + tn) / len(items), 3),
                  "recall": round(tp / max(1, tp + len(fn)), 3), "false_positives": fp, "false_negatives": fn}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "score":
        print(json.dumps(score(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
