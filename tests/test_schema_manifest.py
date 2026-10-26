import json

from analyzer.manifest import plan_batch, read_manifest
from analyzer.schema import FIELDS, coerce
from tests.conftest import tone


def test_coerce_repairs_and_orders():
    out = coerce({"emotional_tone": "Frustrated", "background_noise_present": "yes",
                  "background_noise_severity": "none", "audio_quality": "slightly impaired",
                  "confidence": 7})
    assert list(out) == FIELDS
    assert out["emotional_tone"] == "frustrated"
    assert out["background_noise_severity"] == "low"  # present noise cannot be 'none'
    assert out["audio_quality"] == "slightly_impaired"
    assert out["confidence"] == 1.0


def test_coerce_no_noise_clears_type():
    out = coerce({"background_noise_present": False, "background_noise_type": "TV",
                  "background_noise_severity": "high"})
    assert out["background_noise_type"] == "" and out["background_noise_severity"] == "none"


def test_read_manifest_requires_name():
    rows, errs = read_manifest("file,result_json\na.wav,\n")
    assert errs and not rows


def test_plan_batch_reports_mismatches(tmp_path, make_wav):
    make_wav("a.wav", tone(1))
    make_wav("b.wav", tone(1))
    (tmp_path / "notes.txt").write_text("x")
    label = json.dumps({"emotional_tone": "neutral"})
    (tmp_path / "labels.csv").write_text(
        'name,result_json\na.wav,"' + label.replace('"', '""') + '"\nmissing.mp3,\nb.wav,{bad json\n')
    p = plan_batch(tmp_path)
    assert p.missing_audio == ["missing.mp3"]
    assert p.unsupported == ["notes.txt"]
    assert p.malformed_labels == ["b.wav"]
    assert p.labels["a.wav"]["emotional_tone"] == "neutral"
    assert sorted(p.to_process) == ["a.wav", "b.wav"]


def test_plan_batch_descends_wrapper_folder(tmp_path, make_wav):
    inner = tmp_path / "evaluation_batch"
    inner.mkdir()
    (tmp_path / "__MACOSX").mkdir()
    from tests.conftest import write_wav
    write_wav(inner / "c.wav", tone(1))
    (inner / "labels.csv").write_text("name,result_json\nc.wav,\n")
    p = plan_batch(tmp_path)
    assert p.to_process == ["c.wav"] and not p.missing_audio


def test_no_manifest_processes_all(tmp_path, make_wav):
    make_wav("a.wav", tone(1))
    p = plan_batch(tmp_path)
    assert p.to_process == ["a.wav"] and p.warnings
