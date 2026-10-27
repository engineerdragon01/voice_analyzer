"""Scoring against labels: tone accuracy + macro-F1, per-field accuracy, confusion matrix, calibration."""
from __future__ import annotations

from .schema import ENUMS, TONES, coerce

SCORED_FIELDS = [
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
]


def macro_f1(y_true: list, y_pred: list, classes: list) -> tuple[float, dict]:
    """Macro-F1 over classes that appear in y_true or y_pred (absent-everywhere classes are skipped)."""
    per = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = tp + fn
        if support == 0 and fp == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": support}
    mf1 = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return round(mf1, 3), per


def confusion(y_true: list, y_pred: list, classes: list) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def score(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs = [(label, prediction)]. Each field is scored only over labels that actually contain it
    (a partial result_json is not penalized for fields it omits). Label values are coerced like predictions."""
    if not pairs:
        return {"n": 0}
    raw = [l for l, _ in pairs]
    labels = [coerce(l) for l in raw]
    preds = [coerce(p) for _, p in pairs]
    out: dict = {"n": len(pairs), "fields": {}}
    for f in SCORED_FIELDS:
        idx = [i for i, r in enumerate(raw) if f in r]
        if not idx:
            continue
        yt = [labels[i][f] for i in idx]
        yp = [preds[i][f] for i in idx]
        acc = sum(1 for a, b in zip(yt, yp) if a == b) / len(yt)
        classes = ENUMS[f] if f in ENUMS else [False, True]
        mf1, per = macro_f1(yt, yp, classes)
        out["fields"][f] = {"n": len(idx), "accuracy": round(acc, 3), "macro_f1": mf1,
                            "per_class": {str(k): v for k, v in per.items()}}
    tone_idx = [i for i, r in enumerate(raw) if "emotional_tone" in r]
    yt = [labels[i]["emotional_tone"] for i in tone_idx]
    yp = [preds[i]["emotional_tone"] for i in tone_idx]
    out["tone_confusion"] = {"classes": TONES, "matrix": confusion(yt, yp, TONES)}
    exact = [all(labels[i][f] == preds[i][f] for f in SCORED_FIELDS if f in raw[i]) for i in range(len(raw))]
    out["all_fields_exact"] = round(sum(exact) / len(exact), 3)
    accs = [v["accuracy"] for v in out["fields"].values()]
    out["mean_field_accuracy"] = round(sum(accs) / len(accs), 3) if accs else None
    conf = [preds[i]["confidence"] for i in tone_idx]
    if tone_idx:
        # Calibration: Brier score of confidence vs. "tone correct".
        correct = [1.0 if a == b else 0.0 for a, b in zip(yt, yp)]
        out["tone_brier"] = round(sum((c - y) ** 2 for c, y in zip(conf, correct)) / len(conf), 3)
        out["mean_confidence"] = round(sum(conf) / len(conf), 3)
        out["tone_accuracy"] = out["fields"]["emotional_tone"]["accuracy"]
        out["tone_macro_f1"] = out["fields"]["emotional_tone"]["macro_f1"]
    else:
        out["tone_brier"] = out["mean_confidence"] = out["tone_accuracy"] = out["tone_macro_f1"] = None
    return out


def to_markdown(s: dict, title: str = "Results") -> str:
    if not s.get("n"):
        return f"## {title}\n\nNo labeled items.\n"
    def fmt(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)

    lines = [f"## {title}", "", f"n = {s['n']} labeled clips", "",
             f"- Tone accuracy: **{fmt(s['tone_accuracy'])}**, tone macro-F1: **{fmt(s['tone_macro_f1'])}**",
             f"- Mean per-field accuracy: {fmt(s['mean_field_accuracy'])}; all-fields-exact: {s['all_fields_exact']:.3f}",
             f"- Confidence: mean {fmt(s['mean_confidence'], '.2f')}, tone Brier {fmt(s['tone_brier'])}", "",
             "| Field | Accuracy | Macro-F1 |", "|---|---|---|"]
    for f, v in s["fields"].items():
        lines.append(f"| {f} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    cls = s["tone_confusion"]["classes"]
    lines += ["", "Tone confusion matrix (rows = label, columns = prediction):", "",
              "| label \\ pred | " + " | ".join(cls) + " |", "|---" * (len(cls) + 1) + "|"]
    for c, row in zip(cls, s["tone_confusion"]["matrix"]):
        lines.append(f"| {c} | " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"
