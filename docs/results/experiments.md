# Experiment matrix — provided labeled calls (n=3)

Zero-shot: labels are never shown to the model. Each Gemini config run twice to expose run-to-run variance (DSP-only is deterministic, run once).

**How to read this:** with 3 calls, one call moves a field by 0.33, so rows within ~0.05 of each other are indistinguishable. Row A scores well only because its fixed "medium" intensity matches all three labels; it cannot output a non-neutral tone. Columns: tone/int./noise/sev./qual./overlap/silence = per-field accuracy; mean field acc = average over those 7 fields; $/audio-min = measured from API token usage.

| Config | run | tone acc | tone macro-F1 | tone | int. | noise | sev. | qual. | overlap | silence | mean field acc | $/audio-min | wall s (3 files, 4 workers) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A. DSP rules only | 1 | 0.33 | 0.17 | 0.33 | 0.67 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | **0.714** | 0.00000 | 1.5 |
| B. Gemini 3.5 Flash-Lite, prompt v1, k=1 | 1 | 0.33 | 0.17 | 0.33 | 0.00 | 0.33 | 0.33 | 1.00 | 0.33 | 0.33 | **0.381** | 0.00088 | 3.9 |
| B. Gemini 3.5 Flash-Lite, prompt v1, k=1 | 2 | 0.33 | 0.17 | 0.33 | 0.00 | 0.33 | 0.33 | 1.00 | 0.33 | 0.33 | **0.381** | 0.00088 | 3.6 |
| C. Gemini 3.5 Flash-Lite, prompt v2, k=1 | 1 | 0.33 | 0.17 | 0.33 | 0.33 | 0.00 | 0.00 | 1.00 | 0.33 | 1.00 | **0.428** | 0.00092 | 3.4 |
| C. Gemini 3.5 Flash-Lite, prompt v2, k=1 | 2 | 0.33 | 0.17 | 0.33 | 0.33 | 0.00 | 0.00 | 1.00 | 0.33 | 1.00 | **0.428** | 0.00092 | 6.0 |
| D. Hybrid 3.5 Flash-Lite, v2, k=1 | 1 | 0.33 | 0.17 | 0.33 | 0.33 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | **0.667** | 0.00092 | 3.7 |
| D. Hybrid 3.5 Flash-Lite, v2, k=1 | 2 | 0.33 | 0.17 | 0.33 | 0.33 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | **0.667** | 0.00092 | 3.1 |
| E. Hybrid 3.5 Flash-Lite, v2, adaptive k<=3 (final) | 1 | 0.33 | 0.17 | 0.33 | 0.33 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | **0.667** | 0.00187 | 7.2 |
| E. Hybrid 3.5 Flash-Lite, v2, adaptive k<=3 (final) | 2 | 0.33 | 0.17 | 0.33 | 0.33 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | **0.667** | 0.00187 | 7.0 |
| F. Hybrid 3.1 Flash-Lite, v2, k=1 | 1 | 0.00 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | **0.714** | 0.00109 | 3.7 |
| F. Hybrid 3.1 Flash-Lite, v2, k=1 | 2 | 0.00 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | **0.714** | 0.00109 | 3.6 |

## Per-call predictions

```json
{
 "A. DSP rules only (run 1)": {
  "call_001.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.35
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "steady background noise",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.35
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.35
  }
 },
 "B. Gemini 3.5 Flash-Lite, prompt v1, k=1 (run 1)": {
  "call_001.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": true,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": true,
   "confidence": 0.72
  }
 },
 "B. Gemini 3.5 Flash-Lite, prompt v1, k=1 (run 2)": {
  "call_001.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": true,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": true,
   "confidence": 0.72
  }
 },
 "C. Gemini 3.5 Flash-Lite, prompt v2, k=1 (run 1)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  }
 },
 "C. Gemini 3.5 Flash-Lite, prompt v2, k=1 (run 2)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  }
 },
 "D. Hybrid 3.5 Flash-Lite, v2, k=1 (run 1)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": true,
   "background_noise_type": "steady background noise",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.6
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.6
  }
 },
 "D. Hybrid 3.5 Flash-Lite, v2, k=1 (run 2)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": true,
   "background_noise_type": "steady background noise",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.6
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.6
  }
 },
 "E. Hybrid 3.5 Flash-Lite, v2, adaptive k<=3 (final) (run 1)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": true,
   "background_noise_type": "steady background noise",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.6
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.75
  }
 },
 "E. Hybrid 3.5 Flash-Lite, v2, adaptive k<=3 (final) (run 2)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": true,
   "background_noise_type": "steady background noise",
   "background_noise_severity": "low",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.6
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "sharp static",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.75
  }
 },
 "F. Hybrid 3.1 Flash-Lite, v2, k=1 (run 1)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "radio or TV",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": true,
   "background_noise_type": "office chatter",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": true,
   "long_silence_present": false,
   "confidence": 0.72
  }
 },
 "F. Hybrid 3.1 Flash-Lite, v2, k=1 (run 2)": {
  "call_001.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": false,
   "background_noise_type": "",
   "background_noise_severity": "none",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_002.ogg": {
   "emotional_tone": "frustrated",
   "emotional_intensity": "medium",
   "background_noise_present": true,
   "background_noise_type": "radio or TV",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": false,
   "long_silence_present": false,
   "confidence": 0.72
  },
  "call_003.ogg": {
   "emotional_tone": "neutral",
   "emotional_intensity": "low",
   "background_noise_present": true,
   "background_noise_type": "office chatter",
   "background_noise_severity": "medium",
   "audio_quality": "clear",
   "speaker_overlap_present": true,
   "long_silence_present": false,
   "confidence": 0.72
  }
 }
}
```
