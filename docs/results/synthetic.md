# Synthetic validation — objective fields (n = 34 clips, 29.1 audio-min)

Built by `python -m eval.synthetic build` from the provided calls with controlled, known transformations
(audio stays local; derived clips are never committed):

- **Silence (12 clips):** 0 / 4 / 15 / 25 s of near-floor dead air inserted mid-call into each of the 3 calls
  (label: long silence iff ≥ 10 s).
- **Background noise (18 clips):** white, pink, 60 Hz hum, static crackle, and babble (other calls' speech) mixed
  into call_001 (labeled noise-free) at 5 / 15 / 25 dB SNR, plus 3 clean negatives.
- **Audio quality (4 clips):** hard clipping (×12 gain), 80 ms dropouts every second, −40 dB attenuation, clean.

All clips are re-encoded to Ogg/Opus, as real uploads would be, so detectors are tested after lossy coding.

| Field | n | DSP rules only | Hybrid, LLM noise only (intermediate) | **Final hybrid** |
|---|---|---|---|---|
| long_silence_present | 12 | 1.00 | 1.00 | **1.00** |
| background_noise_present | 18 | 0.944 (misses the loudest static: no non-speech left to measure clicks in) | 0.667 (misses pink ×2, hum ×2, babble; 1 FP) | **1.00** (0 FN, 0 FP) |
| audio_quality impaired vs clear | 4 | 0.75 (misses dropouts) | 0.75 | **0.75** (misses dropouts) |

The detectors are complementary:
- **Steady noise** (hiss, hum, distant babble): SNR < 30 dB or a mains-hum peak. The audio LLM under-reports these.
- **Intermittent line static / crackle:** high-frequency click rate in non-speech ≥ 90 per minute (clean calls
  ≤ 35, the labeled static call 180, synthetic static 940+).
- **Voices / TV / music and anything else audible:** the audio LLM.
- The LLM's own claims of *static* are accepted only when measured clicks support them (or cannot be measured),
  because those claims flipped between audio codecs on identical calls.

Cost on the final run: $0.0523 for 29.1 audio-min = **$0.0018 / audio-min** (worst single file $0.0022/min);
24 s wall time with 4 workers.
