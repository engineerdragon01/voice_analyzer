import importlib
import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import silence, tone, write_wav


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("DASH_USER", "u")
    monkeypatch.setenv("DASH_PASS", "p")
    import app.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        yield c


def _login(c):
    r = c.post("/api/login", data={"username": "u", "password": "p"})
    assert r.status_code == 200


def _zip(tmp_path):
    a = write_wav(tmp_path / "call_a.wav", np.concatenate([tone(3), silence(12), tone(3)]))
    buf = io.BytesIO()
    label = json.dumps({"emotional_tone": "neutral", "long_silence_present": True})
    csv = "name,result_json\ncall_a.wav,\"" + label.replace('"', '""') + "\"\nbroken.mp3,\nghost.wav,\n"
    with zipfile.ZipFile(buf, "w") as z:
        z.write(a, "evaluation_batch/call_a.wav")
        z.writestr("evaluation_batch/broken.mp3", b"garbage" * 200)
        z.writestr("evaluation_batch/labels.csv", csv)
        z.writestr("__MACOSX/evaluation_batch/._call_a.wav", b"x")
        z.writestr("../../evil.txt", b"x")
    return buf.getvalue()


def test_requires_login(client):
    assert client.get("/api/batches").status_code == 401
    assert client.get("/", follow_redirects=False).status_code in (302, 307)
    assert client.post("/api/login", data={"username": "u", "password": "bad"}).status_code == 401


def test_zip_batch_end_to_end(client, tmp_path):
    _login(client)
    r = client.post("/api/batches", files=[("files", ("batch.zip", _zip(tmp_path), "application/zip"))])
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["missing_audio"] == ["ghost.wav"]
    assert rep["to_process"] == 2
    assert not (tmp_path / "evil.txt").exists()
    bid = r.json()["batch_id"]
    assert client.post(f"/api/batches/{bid}/run", data={"backend": "dsp"}).status_code == 200
    for _ in range(100):
        s = client.get(f"/api/batches/{bid}").json()
        if s["batch"]["status"] == "completed":
            break
        time.sleep(0.1)
    assert s["batch"]["status"] == "completed"
    by = {f["name"]: f for f in s["files"]}
    assert by["call_a.wav"]["status"] == "done"
    assert by["call_a.wav"]["prediction"]["long_silence_present"] is True
    assert by["broken.mp3"]["status"] == "error" and by["broken.mp3"]["error"]
    assert s["metrics"]["n"] == 1
    csv_text = client.get(f"/api/batches/{bid}/results.csv").text
    assert csv_text.startswith("name,result_json") and "call_a.wav" in csv_text and "broken.mp3" in csv_text
    js = client.get(f"/api/batches/{bid}/results.json").json()
    assert {x["name"] for x in js} == {"call_a.wav", "broken.mp3"}


def test_folder_upload_with_relative_paths(client, tmp_path):
    _login(client)
    a = write_wav(tmp_path / "x.wav", tone(2)).read_bytes()
    r = client.post("/api/batches", files=[
        ("files", ("batch/x.wav", a, "audio/wav")),
        ("files", ("batch/labels.csv", b"name,result_json\nx.wav,\n", "text/csv")),
        ("files", ("batch/sub/y.wav", a, "audio/wav")),
    ])
    rep = r.json()["report"]
    assert rep["to_process"] == 1 and rep["manifest"] == "labels.csv"
    assert any("nested" in w for w in rep["warnings"])


def test_bad_zip_rejected(client):
    _login(client)
    r = client.post("/api/batches", files=[("files", ("b.zip", b"not a zip", "application/zip"))])
    assert r.status_code == 400


def _completed_dsp_batch(client, tmp_path):
    r = client.post("/api/batches", files=[("files", ("batch.zip", _zip(tmp_path), "application/zip"))])
    bid = r.json()["batch_id"]
    client.post(f"/api/batches/{bid}/run", data={"backend": "dsp"})
    for _ in range(100):
        if client.get(f"/api/batches/{bid}").json()["batch"]["status"] == "completed":
            return bid
        time.sleep(0.1)
    raise AssertionError("batch did not complete")


def test_delete_batch(client, tmp_path):
    _login(client)
    bid = _completed_dsp_batch(client, tmp_path)
    import app.main as m
    assert (m.DATA_DIR / "batches" / bid).exists()
    assert client.delete(f"/api/batches/{bid}").status_code == 200
    assert client.get(f"/api/batches/{bid}").status_code == 404
    assert not (m.DATA_DIR / "batches" / bid).exists()
    assert bid not in [b["id"] for b in client.get("/api/batches").json()]
    assert client.delete(f"/api/batches/{bid}").status_code == 404


def test_delete_refused_while_processing(client, tmp_path):
    _login(client)
    a = write_wav(tmp_path / "x.wav", tone(2)).read_bytes()
    bid = client.post("/api/batches", files=[("files", ("x.wav", a, "audio/wav"))]).json()["batch_id"]
    from app import store
    store.set_file(bid, "x.wav", status="queued")
    r = client.delete(f"/api/batches/{bid}")
    assert r.status_code == 409 and "cancel" in r.json()["detail"].lower()
    store.set_file(bid, "x.wav", status="error")
    assert client.delete(f"/api/batches/{bid}").status_code == 200


def test_delete_requires_login(client):
    assert client.delete("/api/batches/anything").status_code == 401
