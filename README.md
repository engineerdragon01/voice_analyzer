# AutoAce Voice Analyzer

Classifies **emotional tone** and **background noise / audio conditions** in production call audio and returns the
AutoAce 9-field schema per clip. Ships as a hosted dashboard (login → upload folder/ZIP → validate → process →
review → download CSV/JSON) plus a CLI for reproducible offline runs.

```
upload ─► manifest validation ─► worker pool (per-file isolation, retries, timeouts)
  per file: ffmpeg decode ─► DSP features (numpy/scipy: VAD, silence, SNR, noise spectrum, clipping, level)
                         └─► Gemini audio (16 kHz mono Opus, schema-constrained JSON, k self-consistency samples,
                              DSP evidence in prompt)
            ─► fusion (votes + DSP-owned objective fields) ─► schema coercion ─► calibrated confidence
```

**Hosted:** https://autoace-voice-analyzer.onrender.com (credentials provided separately)

Design rationale, validation, cost and latency are in [`docs/MEMO.md`](docs/MEMO.md); experiment tables in
[`docs/results/`](docs/results/); predictions for the provided calls in [`predictions/`](predictions/).

## Quick start (local)

Requirements: Python 3.11+, `ffmpeg` on PATH.

```bash
pip install -r requirements-dev.txt
cp .env.example .env            # fill GEMINI_API_KEY, DASH_PASS
set -a; source .env; set +a
uvicorn app.main:app --port 8000   # open http://localhost:8000
```

### CLI

```bash
# Batch folder = audio files + optional CSV manifest (name,result_json)
python cli.py path/to/batch -o results/run1 --backend hybrid     # or: gemini | dsp
# -> results/run1.csv (name,result_json,error), .json, .meta.json (timings, cost, DSP features), .metrics.json
python -m eval.benchmark results/run1.meta.json                   # latency + cost summary
python -m eval.benchmark --projection gemini-3.5-flash-lite       # cost by clip length
python -m eval.experiments data/provided docs/results/experiments.md  # full comparison matrix
python -m eval.refuse results/run1.meta.json path/to/labels.csv    # re-score fusion changes offline (no API calls)
```

If the manifest has labels, accuracy, macro-F1 and a tone confusion matrix are printed (and shown in the dashboard).

### Tests

```bash
pytest -q        # DSP on synthetic signals, schema, manifest validation, budget guard, fusion, API end-to-end
```

### Synthetic validation (objective fields)

```bash
python -m eval.synthetic build data/provided data/synthetic
python cli.py data/synthetic -o results/synthetic_hybrid --backend hybrid
python -m eval.synthetic score data/synthetic/labels.csv results/synthetic_hybrid.csv
```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Paid-tier Gemini Developer API key (required for `hybrid`/`gemini`) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model; pricing table in `analyzer/config.py` |
| `GEMINI_SAMPLES` | `3` | Max self-consistency samples per clip (budget guard may lower it) |
| `PROMPT_VERSION` | `v2` | Prompt file in `analyzer/prompts/` |
| `TARGET_COST_PER_MIN` | `0.0025` | Budget guard: k is lowered per clip so projected cost stays below this |
| `ANALYZER_BACKEND` | `hybrid` | Default backend |
| `LONG_SILENCE_S` | `10` | Interior dead-air threshold (seconds) |
| `DASH_USER` / `DASH_PASS` | `autoace` / – | Dashboard login (login disabled until `DASH_PASS` is set) |
| `SESSION_SECRET` | random | Cookie signing key (set it so sessions survive restarts) |
| `DATA_DIR` | `./data/app` | SQLite DB + temporary uploads |
| `WORKERS` | `4` | Concurrent files |
| `KEEP_AUDIO` | `0` | Uploaded audio is deleted after analysis unless set |
| `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | – | Use Vertex AI so audio stays in AutoAce's GCP project |

## Deploy (Render, Docker)

1. render.com → **New + → Blueprint** → pick this repo/branch (uses `render.yaml`).
2. Set `GEMINI_API_KEY` and `DASH_PASS` when prompted (`SESSION_SECRET` is generated).
3. Starter plan + 1 GB disk at `/data` (configured in the blueprint). Apply, wait for the build, open the URL.

Railway works the same with `railway.json` (add a volume at `/data` and the same variables).

## Batch format

```
evaluation_batch/            (folder or ZIP; a single wrapper folder is fine)
  call_001.wav
  call_002.mp3
  labels.csv                 name,result_json   (result_json optional)
```

Supported audio: wav, mp3, ogg, opus, m4a, mp4, aac, flac, webm, wma, amr. Validation reports CSV rows with no
audio, audio not in the CSV (still processed), unsupported files, malformed `result_json`. A corrupt file is
marked as failed with the reason; the rest of the batch continues.

## Data handling

- Audio is sent only to the Gemini API (paid tier: not used to train Google models); uploaded files are deleted
  after analysis; results stay in the service's SQLite DB.
- Production option: Vertex AI in AutoAce's own GCP project/region (same code path, env switch).
- No audio is committed to this repository (`.gitignore`).

## Layout

```
analyzer/   audio_io, dsp, llm (Gemini), fusion, config (pricing + budget guard), schema, manifest, metrics, pipeline
app/        FastAPI app (auth, upload, jobs, results) + static dashboard
eval/       synthetic set builder/scorer, benchmark
tests/      pytest suite
docs/       MEMO.md and result tables
predictions/ outputs for the provided calls
```
