# Technical Memo — Voice Tone & Background Noise Classifier

**Hosted dashboard:** https://autoace-voice-analyzer.onrender.com · **Final config:** hybrid DSP + Gemini 3.5
Flash-Lite (prompt v2, lossless audio, adaptive self-consistency k ≤ 3) · **Measured cost:** $0.0018–0.0019 per
audio-minute (ceiling $0.003) · **Throughput on the hosted 0.5-CPU instance:** ≈ 16 audio-minutes per wall-clock
minute (≈ 8 s of processing per audio-minute per file) · **Provided calls:** 14 of 21 scored fields correct (0.667),
both tone misses adjacent-class.

## 0. Where each evaluation criterion is addressed

| Criterion (weight) | What this submission does | Evidence |
|---|---|---|
| Hidden-set performance (45%) | Field-by-field fusion of three detectors and the LLM; spec definitions verbatim in the prompt; enum-constrained output; DSP owns objective fields | §1–§3; synthetic set: silence 1.00, noise 1.00, quality 0.75; provided calls 0.667 |
| Cost efficiency (15%) | Flash-Lite, lossless-but-small audio, per-clip budget guard, compact schema for short clips, real token usage recorded per file | §4: $0.0018–0.0019/min measured; worst single file $0.0022/min |
| Technical rigor (15%) | Six configurations compared (DSP-only, LLM-only ×2 prompts, hybrid k=1/k≤3, alternative model), each Gemini configuration run twice; zero-shot (no label leakage); thresholds set in wide data gaps and confirmed on a 34-clip synthetic set; offline re-scoring; calibration analysis | §2, §3, `results/` |
| Production practicality (10%) | Per-file isolation, retries, fail-fast on permanent errors, memory budget for a 512 MB instance, crash-loop guard, audio deleted after analysis, single container | §5, §6, §7 |
| Communication (5%) | This memo, a step-by-step README, per-call prediction table, explicit limitations | README, `../predictions/README.md`, §8 |
| Hosted dashboard (10%) | Login; folder or ZIP upload; manifest validation report; live progress; per-file error reasons; results with label comparison; CSV/JSON download; batch history and delete | Verified in a browser against the hosted URL with five purpose-built batches (§6) |

## 1. Summary

The system pairs deterministic signal processing (DSP) with an audio-capable LLM. Each field is decided by the
component that measured best on it:

| Field | Decided by | Evidence |
|---|---|---|
| emotional_tone, emotional_intensity | Gemini (vote over k samples) | needs words, prosody, behaviour; DSP can only guess a prior |
| background_noise_* | union of three detectors: SNR/hum (steady noise), high-frequency click rate (line static), Gemini (voices, TV, music, other) | synthetic noise presence: LLM alone 0.667 → final 1.00 |
| audio_quality | Gemini, floored by DSP clipping (speech kurtosis) and low-level detectors | objective impairments that survive lossy coding |
| speaker_overlap_present | DSP when speakers are on separate channels, else Gemini | provided calls are dual-mono, so overlap is judged by ear |
| long_silence_present | DSP only (interior gap ≥ 10 s) | 1.00 on synthetic; letting the LLM flip borderline gaps hurt |
| confidence | sample agreement + DSP/LLM noise agreement + silence margin | logprobs are not available on these models |

## 2. Approaches tested

All runs are zero-shot: the labeled calls are never shown to the model as examples. Full tables are in
[`results/experiments.md`](results/experiments.md) (3 provided calls, each Gemini config run twice) and
[`results/synthetic.md`](results/synthetic.md) (34 synthetic clips).

| Config (3 provided calls) | Tone acc | Mean acc, 7 scored fields | $/audio-min |
|---|---|---|---|
| A. DSP rules only (tone fixed to "neutral / medium") | 0.33 | 0.714 | 0 |
| B. Gemini 3.5 Flash-Lite alone, prompt v1 | 0.33 | 0.381 | 0.00088 |
| C. Gemini 3.5 Flash-Lite alone, prompt v2 | 0.33 | 0.428 | 0.00092 |
| D. Hybrid, k = 1 | 0.33 | 0.667 | 0.00092 |
| **E. Hybrid, adaptive k ≤ 3 (final)** | 0.33 | 0.667 | 0.00187 |
| F. Hybrid with Gemini 3.1 Flash-Lite, k = 1 | 0.00 | 0.714 | 0.00109 |

**How to read this.** On 3 calls one call moves a field by 0.33, so A, D, E and F are statistically
indistinguishable. A's 0.714 comes from its constant "medium" intensity matching all three labels; it can never
output a non-neutral tone, so its tone macro-F1 on a realistic hidden set is capped near 0.2. The Gemini-only rows
(B, C) show why the DSP detectors matter: the LLM alone is unreliable on noise, silence and severity. Larger models
(Gemini 3.5 Flash, 3.8 Flash; single runs) made **the same tone choices** at ~2× the cost, and 3.5 Flash
hallucinated noise on call_001, so model size was not the bottleneck. Gemini 2.5 models are closed to new API users.

**Iterations**, each justified by the spec or by measurement, never by one call alone:
1. **Prompt v2.** AutoAce calls often have an AI voice agent on the business side, so v2 tells the model that
   customer reactions to a bot (demanding a human, repeated "hello?", insults) are emotional signals. It anchors
   intensity to the spec ("medium = clear and sustained"), asks the model to listen to pauses for intermittent
   noise, and counts TV or other voices over a participant as overlap.
2. **DSP-owned silence:** silence accuracy 0.67 → 1.00.
3. **Steady-noise fusion (LLM OR SNR/hum):** synthetic noise 0.667 → 0.944.
4. **Click-rate static detector plus a guard on LLM static claims:** synthetic noise 0.944 → 1.00; noise fields on
   the provided calls became identical across audio codecs (§3).
5. **Bugs caught by inspecting diagnostics:** a fixed request seed made all k samples identical (3× cost, no
   information); now each sample has its own seed. The budget estimate assumed one billing of the input per clip,
   but these models reject multiple candidates, so the input is billed per sample; the estimate now reflects that.

## 3. Validation

**Metrics.** Per-field accuracy; macro-F1 for tone (skipping classes absent from both labels and predictions);
tone confusion matrix; Brier score of confidence against tone correctness. Implemented in `analyzer/metrics.py`
and shown in the dashboard whenever a batch includes labels.

**Final config on the provided calls (n = 3):** mean field accuracy 0.667.

| Field | Accuracy | | Field | Accuracy |
|---|---|---|---|---|
| emotional_tone | 0.33 (macro-F1 0.17) | | audio_quality | 1.00 |
| emotional_intensity | 0.33 | | speaker_overlap_present | 0.33 |
| background_noise_present | 1.00 | | long_silence_present | 1.00 |
| background_noise_severity | 0.67 | | | |

Per-class tone performance (classes that occur in labels or predictions):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.50 | 1.00 | 0.67 | 1 |
| satisfied | 0.00 | 0.00 | 0.00 | 1 |
| frustrated | 0.00 | — | 0.00 | 0 (predicted once) |
| upset | — | 0.00 | 0.00 | 1 |

Tone confusion matrix (rows = label, columns = prediction):

| label \ pred | neutral | satisfied | frustrated | upset | distressed |
|---|---|---|---|---|---|
| neutral | **1** | 0 | 0 | 0 | 0 |
| satisfied | 1 | **0** | 0 | 0 | 0 |
| frustrated | 0 | 0 | **0** | 0 | 0 |
| upset | 0 | 0 | 1 | **0** | 0 |
| distressed | 0 | 0 | 0 | 0 | **0** |

Both tone errors are **adjacent-class** (upset → frustrated, satisfied → neutral): the model is consistently one
step more conservative than the labeler, at every model size tried.

**Synthetic set (n = 34, objective fields):** silence 1.00, noise presence 1.00 (0 false positives), audio quality
0.75 (packet-loss dropouts are missed after lossy coding).

**Robustness finding.** The hosted container and my workstation produced different noise answers for the same calls.
Identical DSP features and token counts traced it to the audio sent to Gemini: different libopus builds encode
differently, and the model's verdict on faint static flipped with the codec (Opus here, Opus there, FLAC). Fixes:
(1) send lossless FLAC, so every environment sends the same samples; (2) a deterministic click-rate static detector;
(3) accept the LLM's static claims only with click evidence. Re-scoring the stored LLM outputs from all three codec
runs gives identical noise fields. Intensity also moved on one call between codecs, a reminder that the LLM's
borderline calls are fragile. Repeatability on identical input: labels were identical across repeated runs
(experiment table, each configuration run twice); confidence can move by a few hundredths because the hosted model
is not bit-for-bit deterministic even with fixed seeds (call_003: 0.75 locally, 0.81 on a hosted run).

**Limits of this validation.**
- n = 3 can't rank close configurations. The synthetic set is the evidence for objective fields; tone has no
  equivalent without more labeled audio.
- The prompt states that recurring line static is background noise, not an audio-quality impairment. That
  convention, and the click-rate threshold, came from call_003's label, so call_003's noise result is not
  independent. The threshold sits in a wide gap (clean ≤ 35 vs 180 clicks/min) and was confirmed on synthetic static.
- The other thresholds also sit in wide gaps: SNR 30 dB (TV call 25.6 vs clean 44–45) and silence 10 s (longest
  natural pause 7.3 s).
- All three labels carry `confidence: 0.82`, identical to the spec's example output, which suggests an LLM produced
  them from the same spec. If the hidden set was labeled the same way, matching that labeler's conventions (e.g.
  calling polite customers "satisfied", neutral calls "medium") matters as much as acoustic accuracy. I did not hard-code
  those conventions from 3 examples; they are the first thing to settle with more labels.

**Confidence.** Logprobs are not enabled and multiple candidates per request are rejected for these models, so
confidence combines (a) agreement across k independently seeded samples when the budget allows k > 1, (b) agreement
between the DSP and LLM noise detectors, and (c) the silence-threshold margin. Output range 0.05–0.95; perfect
agreement maps to 0.92. With n = 3, calibration is only loosely checkable (tone Brier 0.41, mean confidence 0.69, i.e.
overconfident on tone). The lowest-confidence call (call_002, 0.60) is the one where the model and DSP disagreed on
noise and the model was wrong.

## 4. Cost

Gemini Developer API, paid tier, `gemini-3.5-flash-lite`: audio in $0.30, text in $0.30, output $2.50 per 1M tokens.
Measured: **~25 audio tokens/s**, ~970 prompt tokens, ~130–150 output tokens per sample; input is re-billed per sample.

Budget guard (`analyzer/config.py:choose_samples`): per clip, the largest k ≤ 3 whose projected cost is ≤
$0.0025/min. Clips under 20 s use a compact schema without the free-text evidence fields.

| Clip length | k | Est. cost/clip | Est. cost/audio-min |
|---|---|---|---|
| 15 s | 1 | $0.00064 | $0.00255 |
| 30 s | 1 | $0.00090 | $0.00180 |
| 60 s | 2 | $0.00225 | $0.00225 |
| 120 s | 3 | $0.00473 | $0.00236 |
| 300 s | 3 | $0.00877 | $0.00176 |
| 900 s | 3 | $0.02227 | $0.00148 |

**Measured:** provided calls $0.0074 / 3.96 min = **$0.00187/min** (worst file $0.0020); synthetic set $0.0523 /
29.1 min = **$0.0018/min** (worst file $0.0022); a 15 s clip $0.0024/min. Every request's real token usage is shown
per file and per batch in the dashboard. `GEMINI_SAMPLES=1` halves cost on longer calls (~$0.0009/min) with no
measured accuracy loss, at the price of a less informative confidence score.

Not included: hosting (Render Starter ~$7/month flat, about $0.00002/min at 10k min/day) and DSP compute (< 0.3 s
CPU per audio-minute). Clips under ~12 s can't meet $0.003/min alone because of the fixed prompt overhead; packing
several short clips into one request would fix that.

## 5. Latency

Per file, time splits ~11% decode, ~5% DSP, ~75% Gemini request(s), the rest FLAC encoding and bookkeeping. Median
1.8–2.6 s per file, max ~8 s (the 172 s call with k = 3 sequential samples). That is ≈ 3.3 s of processing per
audio-minute per worker; 4 workers finished 29 audio-minutes in 24 s. Throughput scales with `WORKERS` up to the
API rate limit; running a clip's samples concurrently would cut long-clip latency about 3×.

**On the hosted Render Starter instance (0.5 CPU), measured:** the three provided calls (3.96 audio-min, run
concurrently) finished in **14.6 s wall-clock, ≈ 16 audio-minutes per minute**; per file 8.7 s, 9.0 s and 14.4 s,
of which decode + DSP (CPU-bound, shared across workers on 0.5 CPU) was 3.7–6.6 s and Gemini 2–7 s. Estimate for
larger batches: the CPU-bound part costs ≈ 2 s per audio-minute on this instance, so a 100-audio-minute batch
should take roughly 4–7 minutes. A 1–2 CPU instance roughly halves the CPU-bound part.

## 6. Production design

- **Memory on a 512 MB instance:** audio is decoded as a stream (mono mix and channel statistics built on the
  fly; stereo kept only for genuinely split channels) and every DSP statistic is computed in 30 s chunks. Peak
  memory fell from ~45 to ~10–15 MB per audio-minute with byte-identical features and API audio. Files reserve
  their duration against a shared audio-minute budget (default 15), so short calls run 4-wide while long ones
  queue; `MALLOC_ARENA_MAX=2` and a post-file `malloc_trim` return memory between files. Measured worst case
  (four 11.5-min calls + a 14.5-min call + the three provided calls): **343 MB peak**, back to 130 MB idle.
  Files over 15 min are rejected with a clear message (configurable), and a file that was mid-analysis during two
  restarts is failed rather than retried, so a single file can never crash-loop the service.
- **Batch handling:** folder or ZIP upload (a wrapper folder is fine, `__MACOSX` is ignored, path traversal and zip
  bombs are rejected); manifest validation before processing (missing audio, unlisted audio, unsupported files,
  malformed labels); per-file isolation with retries and timeouts, and permanent API errors fail fast; live progress
  and per-file error reasons; SQLite job store, so results survive refreshes, and interrupted files are re-queued
  after a restart.
- **Outputs:** CSV (`name,result_json`, one column per field, status/error) and JSON, with original filenames.
- **Reproducibility:** pinned dependencies; versioned prompts (`analyzer/prompts/v1.txt`, `v2.txt`); per-sample
  fixed seeds; lossless audio to the API; the CLI shares the web code path (`analyzer/pipeline.py`);
  `eval/experiments.py` regenerates every table; `eval/refuse.py` re-scores fusion changes from stored model
  outputs without API calls.
- **Verification:** 32 pytest cases (DSP on synthetic signals, schema, manifest, budget guard, fusion, API end to
  end with a corrupt file, unmatched rows, a ZIP traversal attempt, batch deletion, the memory budget, the restart guard and partial-label scoring), plus Playwright browser tests
  against a local server and the hosted Render deployment. The hosted audit covered the spec's example batch (wav +
  mp3 + labels), a folder upload, a manifest without `result_json`, a batch with missing / unlisted / unsupported /
  corrupt / empty / malformed-label / duplicate entries and m4a / flac / uppercase extensions, and a batch with no CSV.

## 7. Data handling and external API disclosure

- **External API:** Google Gemini Developer API, model `gemini-3.5-flash-lite`, paid tier, prices as above.
- **Does audio leave AutoAce infrastructure?** Yes. Each clip is converted to 16 kHz mono FLAC and sent to Google's
  API. Under the paid-tier terms, prompts and responses are not used to train Google's models; Google may keep them
  for a limited period for abuse monitoring.
- **Inside the service:** uploaded audio is deleted right after analysis (`KEEP_AUDIO=0`); only predictions,
  measurements and labels persist in SQLite on the service's disk. No audio, and no model text quoting callers, is
  committed to the repository.
- **Recommended production path:** Vertex AI in AutoAce's own GCP project and region (same code; set
  `GOOGLE_GENAI_USE_VERTEXAI=true`), so audio stays under AutoAce's cloud account and residency controls. The
  DSP-only backend needs no third party at all (no tone model).

## 8. Failure modes and limitations

- **Tone conservatism:** the model under-calls "satisfied" (polite = neutral) and "upset" (called frustrated).
  This is the largest error source; it needs labeled data to calibrate.
- **Customer attribution:** on mono audio the model infers which voice is the customer; an angry agent or a
  recorded prompt could be mis-attributed.
- **Overlap on mono audio** is judged by ear and was under-called (0.33). Dual-channel recordings automatically
  switch to the measured overlap detector.
- **Dropouts / packet loss** are not detectable reliably after lossy coding; only severe cases are heard.
- **Steady-noise threshold** assumes telephone-like levels; a very quiet talker over a modest floor can read as
  noisy. Music-on-hold or IVR audio can read as background noise. Loud continuous static masks the click detector
  (the LLM covers it).
- **Codec sensitivity** of the LLM's borderline judgements (§3); mitigated for noise, not eliminated for intensity.
- **Language:** prompts are in English; the Spanish call was handled, but non-English emotion cues are less tested.
- **Model churn:** Gemini 2.5 closed to new users mid-project. The model is one env var, and unknown models are
  priced pessimistically so the budget guard fails safe.

## 9. Next steps

1. **Labels:** 200–500 labeled calls with agreed conventions, so tone thresholds can be calibrated and a small
   stacking classifier trained on LLM outputs plus acoustic features, with grouped cross-validation by caller.
2. **Self-hosted tone model:** speech-emotion embeddings (emotion2vec / wav2vec2) plus a logistic-regression head,
   ensembled with the LLM; near-zero marginal cost, and audio stays in AutoAce's infrastructure.
3. **Diarization** (customer vs agent) before tone; use dual-channel recordings where the telephony stack has them.
4. **Cost/latency:** pack short clips per request, run samples concurrently, cache the static prompt.
5. **Calibration:** isotonic calibration of confidence once labels exist; route low-confidence calls to human review.
