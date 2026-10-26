# Technical Memo — Voice Tone & Background Noise Classifier

**Hosted dashboard:** https://autoace-voice-analyzer.onrender.com · **Repo:** this repository · **Final config:**
hybrid DSP + Gemini 3.5 Flash-Lite, prompt v2, adaptive self-consistency (k ≤ 3) · **Measured cost:**
$0.0018–0.0019 per audio-minute (ceiling $0.003) · **Latency:** ~4 s processing per audio-minute per worker;
~1 s wall per audio-minute with 4 workers.

## 1. Summary

The system combines deterministic signal processing with an audio-capable LLM, each assigned the fields it is
measurably better at:

| Field | Decided by | Why |
|---|---|---|
| emotional_tone, emotional_intensity | Gemini (vote over k samples) | needs words, prosody and behaviour |
| background_noise_present / type / severity | Gemini **OR** DSP steady-noise detector | LLM hears intermittent sounds (static, TV, voices); DSP catches steady noise the LLM under-reports (synthetic: 0.667 → 0.944) |
| audio_quality | Gemini, floored by DSP clipping / low-level detectors | objective impairments the LLM can miss |
| speaker_overlap_present | DSP when speakers are on separate channels, else Gemini | provided calls are dual-mono, so overlap must be judged by ear |
| long_silence_present | DSP only (interior gap ≥ 10 s) | 100% on synthetic set; LLM override hurt (exp. D vs earlier rule) |
| confidence | sample agreement + DSP/LLM agreement + silence margin | logprobs are unavailable on these models |

## 2. Approaches tested

All runs are zero-shot: the labeled calls are never shown to the model as examples. Full tables:
[`results/experiments.md`](results/experiments.md) (3 provided calls, each Gemini config run twice) and
[`results/synthetic.md`](results/synthetic.md) (34 synthetic clips).

| Config | Tone acc | Mean field acc (7 scored fields) | $/audio-min |
|---|---|---|---|
| A. DSP rules only (tone = neutral prior) | 0.33 | 0.619 | 0 |
| B. Gemini 3.5 Flash-Lite, prompt v1 | 0.33 | 0.428 | 0.00088 |
| C. Gemini 3.5 Flash-Lite, prompt v2 | 0.33 | 0.571 | 0.00092 |
| D. Hybrid (DSP + Gemini v2), k = 1 | 0.33 | **0.667** | 0.00092 |
| **E. Hybrid, adaptive k ≤ 3 (final)** | 0.33 | **0.667** | 0.00188 |
| F. Hybrid with Gemini 3.1 Flash-Lite, k = 1 | 0.00 | 0.619 | 0.00109 |

Also probed (single runs, not in the table): Gemini 3.5 Flash and 3.8 Flash. They made the **same tone choices**
on all three calls as Flash-Lite at ~2× the cost (3.5 Flash also hallucinated noise on call_001), so model size
was not the bottleneck. Gemini 2.5 Flash/Flash-Lite are closed to new API users.

Key iterations, each motivated by the spec text rather than by a specific call:
1. **Prompt v2:** the business side of AutoAce calls is often an AI voice agent; v2 tells the model that customer
   reactions to a bot (demanding a human, repeated "hello?", insults) are emotional signals; anchors intensity
   to the spec ("medium = clear and sustained", which includes a steady neutral manner); tells the model to listen
   to pauses for intermittent noise; counts TV/other voices over a participant as overlap. Mean field accuracy
   0.428 → 0.571; intensity 0.00 → 0.67.
2. **DSP-owned silence:** removed a rule that let the model flip borderline gaps; silence 0.67 → 1.00.
3. **Noise fusion:** LLM OR DSP steady-noise; noise presence on provided calls 0.67 → 1.00, synthetic 0.667 → 0.944.
4. **Bug found by inspecting diagnostics:** a fixed request seed made all k samples identical (3× cost for no
   information). Fixed with a per-sample seed (still reproducible).

## 3. Validation

**Metric.** Per-field accuracy; macro-F1 for tone (classes absent from both labels and predictions are skipped);
tone confusion matrix; Brier score of confidence vs. tone correctness. Implemented in `analyzer/metrics.py` and
shown in the dashboard whenever a batch includes labels.

**Final config on the provided calls (n = 3):**

| Field | Accuracy |
|---|---|
| emotional_tone | 0.33 (macro-F1 0.17) |
| emotional_intensity | 0.67 |
| background_noise_present | 1.00 |
| background_noise_severity | 0.33 |
| audio_quality | 1.00 |
| speaker_overlap_present | 0.33 |
| long_silence_present | 1.00 |

Tone confusion matrix (rows = label, columns = prediction):

| label \ pred | neutral | satisfied | frustrated | upset | distressed |
|---|---|---|---|---|---|
| neutral | **1** | 0 | 0 | 0 | 0 |
| satisfied | 1 | **0** | 0 | 0 | 0 |
| frustrated | 0 | 0 | **0** | 0 | 0 |
| upset | 0 | 0 | 1 | **0** | 0 |
| distressed | 0 | 0 | 0 | 0 | **0** |

Both tone errors are **adjacent-class** (upset→frustrated, satisfied→neutral): the model is consistently one
step more conservative than the labels. Every model size tested made the same choices, which points at label
convention rather than model capacity.

**Honest limits of this validation.**
- n = 3: one call moves any field by 0.33, so these numbers cannot separate close configurations. The synthetic set
  (n = 34) is the main evidence for objective fields; there is no equivalent for tone without more labeled audio.
- The prompt states that recurring line static is background noise (not an audio-quality impairment). That
  convention was read from call_003's label, so call_003's noise result is not independent evidence.
- Thresholds (SNR < 30 dB for steady noise, 10 s silence) sit inside wide gaps in the observed data (noise: 25.6 dB
  vs 44–45 dB; silence: longest natural gap 7.3 s) and were confirmed on the synthetic set rather than fitted.
- All three labels carry `confidence: 0.82`, identical to the spec's example output, which suggests labels were
  produced by an LLM using the same spec. If the hidden set was labeled the same way, matching that labeler's
  conventions (e.g. its willingness to call polite customers "satisfied") matters more than acoustic accuracy.

**Confidence calibration.** Logprobs are not enabled for these models and multiple candidates per request are
rejected, so confidence comes from (a) agreement across k independently seeded samples (when budget allows k > 1),
(b) agreement between the DSP and LLM noise detectors, and (c) the silence-threshold margin. Values range 0.05–0.95;
perfect agreement maps to 0.92, never 1.0. With n = 3, calibration can only be checked loosely: the lowest-confidence
call (call_002, 0.60, noise detectors disagreed) is the one where the model's noise reading was wrong.

## 4. Cost analysis

Pricing (Gemini Developer API, paid tier, USD per 1M tokens): 3.5 Flash-Lite audio in $0.30, text in $0.30,
output $2.50. Measured usage: **~25 audio tokens per second**, ~970 prompt tokens, ~130–150 output tokens per sample.
Current Flash-Lite models reject `candidate_count > 1`, so each sample re-bills audio + prompt.

Per-clip budget guard (`analyzer/config.py:choose_samples`): picks the largest k ≤ 3 with projected cost ≤
$0.0025/min (headroom under the $0.003 ceiling). Clips under 20 s use a compact schema without the free-text
evidence fields to cut output tokens.

| Clip length | k | Est. cost / clip | Est. cost / audio-min |
|---|---|---|---|
| 15 s | 1 | $0.00064 | $0.00255 |
| 30 s | 1 | $0.00090 | $0.00180 |
| 60 s | 2 | $0.00225 | $0.00225 |
| 120 s | 3 | $0.00473 | $0.00236 |
| 300 s | 3 | $0.00877 | $0.00176 |
| 900 s | 3 | $0.02227 | $0.00148 |

**Measured:** provided calls $0.00745 for 3.96 min = **$0.00188/min** (worst single file $0.0020/min); synthetic
set $0.0521 for 29.1 min = **$0.00179/min** (worst file $0.00223/min); a 15 s clip measured $0.00244/min.
Every request's real token usage is recorded and shown per file and per batch in the dashboard.
Setting `GEMINI_SAMPLES=1` halves cost on longer calls (~$0.0009/min) at no measured accuracy loss on this data,
but the confidence score then carries less information.

**Not included:** hosting (Render Starter, ~$7/month flat, ≈ $0.00002/min at 10k min/day) and DSP compute
(< 0.5 s CPU per audio-minute). Clips under ~12 s cannot meet $0.003/min alone because of the fixed prompt overhead;
packing several short clips into one request is the fix if that traffic matters.

## 5. Latency

| Stage | Share of per-file time |
|---|---|
| ffmpeg decode | ~9% |
| DSP features | ~5% |
| Opus re-encode + Gemini request(s) | ~85% |

Per file: median 2.3–3.5 s, max ~10.7 s (a 172 s call with k = 3, sequential samples). Processing ≈ 4 s per
audio-minute per worker; the synthetic batch (29 audio-min, 34 files) finished in **30 s wall** with 4 workers.
Throughput scales with `WORKERS` up to the API rate limit. Samples within a clip could run concurrently to cut the
long-clip latency roughly 3×.

## 6. Production design

- **Batch handling:** folder or ZIP upload (one wrapper folder tolerated, `__MACOSX` ignored, path traversal and
  zip bombs rejected); manifest validation before processing (missing audio, unlisted audio, unsupported files,
  malformed labels); per-file isolation with retries and timeouts so one bad file never fails the batch; progress
  and per-file error reasons; SQLite job store so results survive refreshes and restarts (interrupted files are
  re-queued on startup).
- **Outputs:** CSV (`name,result_json` plus one column per field and status/error) and JSON, original filenames kept.
- **Reproducibility:** pinned dependencies, versioned prompts (`analyzer/prompts/v1.txt`, `v2.txt`), fixed per-sample
  seeds, CLI identical to the web path (`analyzer/pipeline.py`), `eval/experiments.py` regenerates every table.
- **Tests:** 24 pytest cases (DSP on synthetic signals, schema coercion, manifest validation, budget guard, fusion,
  API end-to-end including a corrupt file, unmatched rows and a ZIP traversal attempt) plus a Playwright browser test.

## 7. Data handling and external API disclosure

- **External API:** Google Gemini Developer API, model `gemini-3.5-flash-lite`, paid tier. Prices as above.
- **Does audio leave AutoAce infrastructure?** Yes: each clip is re-encoded to 16 kHz mono Opus and sent to
  Google's API. Under the paid-tier terms, prompts and responses are not used to train Google's models; Google may
  log them for a limited period for abuse monitoring.
- **Inside the service:** uploaded audio is deleted right after analysis (`KEEP_AUDIO=0`); only predictions,
  features and labels persist in SQLite on the service's disk. No audio is committed to the repository.
- **Recommended production path:** Vertex AI in AutoAce's own GCP project and region (same code; set
  `GOOGLE_GENAI_USE_VERTEXAI=true`), so audio stays under AutoAce's cloud account and data-residency controls.
  Without any third-party processing, the DSP-only backend still runs (lower tone accuracy).

## 8. Failure modes and limitations

- **Tone boundary conservatism:** the model under-calls "satisfied" (polite = neutral to it) and "upset" (calls it
  frustrated). This is the largest error source and is a calibration question that needs more labeled calls.
- **Who is the customer:** on mono recordings the model must infer which voice is the customer; an angry agent or a
  hold-music voice could be mis-attributed.
- **Overlap on mono audio** is judged by ear and was under-called (0.33). Dual-channel recordings switch to the
  measured overlap detector automatically.
- **Dropouts / packet loss** are not reliably detectable after lossy coding (synthetic: missed); the model hears
  severe cases only.
- **Steady-noise threshold** assumes telephone-like levels; a very quiet speaker over a modest noise floor can read
  as noisy. Music-on-hold or IVR prompts can be mistaken for background noise.
- **Language:** prompts are English; call_002 is Spanish and was handled, but non-English emotion cues are less tested.
- **Model churn:** Gemini 2.5 models closed to new users during this project; the model is one env var, and the
  pricing table and budget guard fail safe (unknown models assumed expensive).

## 9. Next steps (with more data or time)

1. **Labels:** 200–500 labeled calls with agreed conventions would let us calibrate tone thresholds and train a
   small classifier on model outputs plus acoustic features (stacking), with grouped cross-validation by caller.
2. **Self-hosted tone model:** speech-emotion embeddings (e.g. emotion2vec / wav2vec2) with a logistic-regression
   head, ensembled with the LLM. Near-zero marginal cost, and audio never leaves AutoAce.
3. **Diarization** (customer vs agent) before tone analysis; use dual-channel recordings where the telephony
   platform provides them.
4. **Cost:** pack short clips into one request; run samples concurrently; context-cache the static prompt.
5. **Calibration:** isotonic calibration of confidence once a few hundred labels exist; route low-confidence calls
   to human review.
