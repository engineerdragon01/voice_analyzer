"""Batch folder discovery and manifest (CSV) validation, shared by CLI and web app."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from .audio_io import is_supported
from .schema import parse_label

IGNORED_PREFIXES = (".", "__MACOSX", "._")


@dataclass
class BatchPlan:
    root: Path
    manifest_name: str | None
    to_process: list[str] = field(default_factory=list)  # audio filenames that will be analyzed
    labels: dict[str, dict | None] = field(default_factory=dict)  # name -> parsed label (or None)
    missing_audio: list[str] = field(default_factory=list)  # in CSV, not in folder
    unlisted_audio: list[str] = field(default_factory=list)  # in folder, not in CSV
    unsupported: list[str] = field(default_factory=list)  # non-audio / unknown files
    malformed_labels: list[str] = field(default_factory=list)  # result_json present but unparseable
    errors: list[str] = field(default_factory=list)  # fatal problems
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "manifest": self.manifest_name,
            "to_process": len(self.to_process),
            "labeled": sum(1 for n in self.to_process if self.labels.get(n)),
            "missing_audio": self.missing_audio,
            "unlisted_audio": self.unlisted_audio,
            "unsupported": self.unsupported,
            "malformed_labels": self.malformed_labels,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def find_batch_root(root: Path) -> Path:
    """If an upload/ZIP wraps everything in a single folder, descend into it."""
    cur = root
    for _ in range(3):
        entries = [p for p in cur.iterdir() if not p.name.startswith(IGNORED_PREFIXES)]
        files = [p for p in entries if p.is_file()]
        dirs = [p for p in entries if p.is_dir()]
        if not files and len(dirs) == 1:
            cur = dirs[0]
        else:
            break
    return cur


def read_manifest(text: str) -> tuple[list[dict], list[str]]:
    """Parse CSV text. Returns (rows with name/result_json, errors)."""
    text = text.lstrip("﻿")
    try:
        reader = csv.DictReader(io.StringIO(text))
        fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    except csv.Error as e:
        return [], [f"CSV could not be parsed: {e}"]
    if "name" not in fields:
        return [], ["CSV is missing the required 'name' column"]
    rows = []
    try:
        for r in reader:
            r = {(k or "").strip().lower(): (v or "") for k, v in r.items()}
            name = r.get("name", "").strip()
            if name:
                rows.append({"name": name, "result_json": r.get("result_json", "")})
    except csv.Error as e:
        return rows, [f"CSV could not be parsed: {e}"]
    return rows, []


def plan_batch(root: Path) -> BatchPlan:
    root = find_batch_root(root)
    files = sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith(IGNORED_PREFIXES))
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    audio = [p.name for p in files if is_supported(p.name)]
    others = [p.name for p in files if p.suffix.lower() != ".csv" and not is_supported(p.name)]

    plan = BatchPlan(root=root, manifest_name=None, unsupported=others)
    if not csvs:
        plan.warnings.append("No CSV manifest found; processing every supported audio file in the folder.")
        plan.to_process = audio
        plan.labels = {n: None for n in audio}
        return plan
    manifest = next((p for p in csvs if p.name.lower() == "labels.csv"), csvs[0])
    if len(csvs) > 1:
        plan.warnings.append(f"Multiple CSV files found; using '{manifest.name}'.")
    plan.manifest_name = manifest.name
    rows, errs = read_manifest(manifest.read_text(errors="replace"))
    if errs:
        plan.errors.extend(errs)
        plan.warnings.append("Manifest unusable; processing every supported audio file in the folder.")
        plan.to_process = audio
        plan.labels = {n: None for n in audio}
        return plan

    seen = set()
    audio_set = set(audio)
    lower_map = {a.lower(): a for a in audio}
    for r in rows:
        name = r["name"]
        if name in seen:
            plan.warnings.append(f"Duplicate manifest row for '{name}' (first row used).")
            continue
        seen.add(name)
        actual = name if name in audio_set else lower_map.get(name.lower())
        if actual is None:
            if (root / name).exists():
                plan.unsupported.append(name) if name not in plan.unsupported else None
            else:
                plan.missing_audio.append(name)
            continue
        if actual != name:
            plan.warnings.append(f"'{name}' matched '{actual}' case-insensitively.")
        label = parse_label(r["result_json"])
        if r["result_json"].strip() and label is None:
            plan.malformed_labels.append(name)
        plan.labels[actual] = label
        plan.to_process.append(actual)
    listed = set(plan.to_process)
    plan.unlisted_audio = [a for a in audio if a not in listed]
    if plan.unlisted_audio:
        plan.warnings.append(f"{len(plan.unlisted_audio)} audio file(s) not in the manifest were also processed.")
        plan.to_process.extend(plan.unlisted_audio)
        for a in plan.unlisted_audio:
            plan.labels[a] = None
    if not plan.to_process:
        plan.errors.append("No processable audio files found.")
    return plan
