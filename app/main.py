"""AutoAce voice analysis dashboard: login, batch upload (folder or ZIP), validation, processing, results."""
from __future__ import annotations

import csv
import hmac
import io
import json
import os
import shutil
import secrets
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from analyzer import metrics
from analyzer.config import Config
from analyzer.llm import available as llm_available
from analyzer.manifest import plan_batch
from analyzer.pipeline import analyze_file
from analyzer.schema import FIELDS

from . import store

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/app"))
DASH_USER = os.getenv("DASH_USER", "autoace")
DASH_PASS = os.getenv("DASH_PASS", "")
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
WORKERS = int(os.getenv("WORKERS", "4"))
KEEP_AUDIO = os.getenv("KEEP_AUDIO", "").lower() in ("1", "true")
STATIC = Path(__file__).parent / "static"



@asynccontextmanager
async def lifespan(_app):
    store.init(DATA_DIR / "app.db")
    # Resume work interrupted by a restart (audio is still on disk for unfinished files).
    for batch_id, name in store.interrupted():
        b = store.get_batch(batch_id)
        c = Config()
        c.backend = (b or {}).get("config", {}).get("backend", c.backend)
        executor.submit(_process, batch_id, name, c)
    yield


app = FastAPI(title="AutoAce Voice Analyzer", docs_url=None, redoc_url=None, lifespan=lifespan)
executor = ThreadPoolExecutor(max_workers=WORKERS)
_login_failures: dict[str, list[float]] = {}
_cancel: set[str] = set()


# ---------------------------------------------------------------- auth
PUBLIC = {"/login", "/api/login", "/healthz", "/static/style.css", "/static/login.js"}


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC or request.session.get("user"):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "not authenticated"}, status_code=401)
    return RedirectResponse("/login")


# Added after the auth middleware so it wraps it (outermost) and populates request.session first.
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=12 * 3600,
                   https_only=os.getenv("COOKIE_SECURE", "").lower() in ("1", "true"), same_site="lax")


@app.get("/login")
def login_page():
    return FileResponse(STATIC / "login.html")


@app.post("/api/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = request.client.host if request.client else "?"
    now = time.time()
    recent = [t for t in _login_failures.get(ip, []) if now - t < 300]
    if len(recent) >= 10:
        raise HTTPException(429, "Too many failed attempts; try again in a few minutes.")
    ok = bool(DASH_PASS) and hmac.compare_digest(username, DASH_USER) and hmac.compare_digest(password, DASH_PASS)
    if not ok:
        _login_failures[ip] = recent + [now]
        raise HTTPException(401, "Invalid username or password.")
    request.session["user"] = username
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------- upload & validation
def _safe_name(raw: str) -> str | None:
    """Keep only the basename of an uploaded path; reject traversal/hidden/empty names."""
    name = PurePosixPath(raw.replace("\\", "/")).name
    if not name or name in (".", "..") or name.startswith("."):
        return None
    return name


def _extract_zip(zpath: Path, dest: Path) -> list[str]:
    """Extract only regular files, flattening paths at the batch root level; guards against zip bombs."""
    notes = []
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        total = sum(i.file_size for i in infos)
        if total > 4 * MAX_UPLOAD_BYTES:
            raise HTTPException(400, "ZIP expands to more than the allowed size.")
        parts = [PurePosixPath(i.filename) for i in infos]
        # Common wrapper folder (e.g. evaluation_batch/) is stripped; deeper nesting is ignored.
        visible = [p for p in parts if not any(s.startswith(("__MACOSX", ".")) for s in p.parts)]
        prefix = None
        if visible and all(len(p.parts) >= 2 for p in visible) and len({p.parts[0] for p in visible}) == 1:
            prefix = visible[0].parts[0]
        for info, p in zip(infos, parts):
            if any(s.startswith(("__MACOSX", ".")) for s in p.parts) or ".." in p.parts or p.is_absolute():
                continue
            rel = p.parts[1:] if prefix else p.parts
            if len(rel) != 1:
                notes.append(f"Ignored nested file '{info.filename}' (audio must be at the batch root).")
                continue
            target = dest / rel[0]
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return notes


@app.post("/api/batches")
async def upload(files: list[UploadFile] = File(...), label: str = Form("")):
    batch_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    bdir = DATA_DIR / "batches" / batch_id / "input"
    bdir.mkdir(parents=True)
    notes: list[str] = []
    total = 0
    try:
        for uf in files:
            raw = uf.filename or ""
            # Folder uploads send relative paths: keep files directly inside the chosen folder only.
            rel = PurePosixPath(raw.replace("\\", "/"))
            if len(rel.parts) > 2:
                notes.append(f"Ignored nested file '{raw}' (audio must be at the batch root).")
                continue
            name = _safe_name(raw)
            if not name:
                continue
            dest = bdir / name
            with open(dest, "wb") as out:
                while chunk := await uf.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // 2**20} MB.")
                    out.write(chunk)
            if name.lower().endswith(".zip"):
                try:
                    notes += _extract_zip(dest, bdir)
                except zipfile.BadZipFile:
                    raise HTTPException(400, f"'{name}' is not a valid ZIP archive.")
                dest.unlink()
    except HTTPException:
        shutil.rmtree(bdir.parent, ignore_errors=True)
        raise
    plan = plan_batch(bdir)
    report = plan.report()
    report["warnings"] = notes + report["warnings"]
    report["batch_root"] = str(plan.root.relative_to(bdir)) if plan.root != bdir else ""
    store.create_batch(batch_id, label or batch_id, report, [(n, plan.labels.get(n)) for n in plan.to_process])
    return {"batch_id": batch_id, "report": report}


# ---------------------------------------------------------------- processing
def _batch_input_dir(batch_id: str) -> Path:
    base = DATA_DIR / "batches" / batch_id / "input"
    root = store.get_batch(batch_id)["report"].get("batch_root") or ""
    return base / root if root else base


def _process(batch_id: str, name: str, cfg: Config) -> None:
    if batch_id in _cancel:
        store.set_file(batch_id, name, status="error", error="cancelled by user", finished=time.time())
        return
    store.set_file(batch_id, name, status="processing", started=time.time())
    path = _batch_input_dir(batch_id) / name
    try:
        pred, meta = analyze_file(path, cfg)
        store.set_file(batch_id, name, status="done", prediction=pred, meta=meta, error=None, finished=time.time())
    except Exception as e:  # noqa: BLE001 — isolate per-file failures
        store.set_file(batch_id, name, status="error", error=f"{type(e).__name__}: {e}"[:500], finished=time.time())
    finally:
        if not KEEP_AUDIO:
            try:
                path.unlink(missing_ok=True)  # confidential audio is not retained after analysis
            except OSError:
                pass
        _maybe_finish(batch_id)


_finish_lock = threading.Lock()


def _maybe_finish(batch_id: str) -> None:
    with _finish_lock:
        files = store.get_files(batch_id)
        if all(f["status"] in ("done", "error") for f in files):
            store.set_batch(batch_id, status="completed")


@app.post("/api/batches/{batch_id}/run")
def run(batch_id: str, backend: str = Form("")):
    b = store.get_batch(batch_id)
    if not b:
        raise HTTPException(404, "Unknown batch.")
    if b["status"] not in ("validated",):
        raise HTTPException(409, f"Batch is already {b['status']}.")
    cfg = Config()
    if backend in ("hybrid", "gemini", "dsp"):
        cfg.backend = backend
    if cfg.backend != "dsp" and not llm_available(cfg):
        raise HTTPException(503, "Gemini API key is not configured on the server; choose the DSP-only backend.")
    files = store.get_files(batch_id)
    if not files:
        raise HTTPException(400, "Nothing to process.")
    store.set_batch(batch_id, status="processing", config={"backend": cfg.backend, "model": cfg.model,
                                                            "max_samples": cfg.max_samples})
    for f in files:
        store.set_file(batch_id, f["name"], status="queued")
        executor.submit(_process, batch_id, f["name"], cfg)
    return {"ok": True, "queued": len(files)}


@app.post("/api/batches/{batch_id}/cancel")
def cancel(batch_id: str):
    _cancel.add(batch_id)
    return {"ok": True}


@app.delete("/api/batches/{batch_id}")
def delete(batch_id: str):
    b = store.get_batch(batch_id)
    if not b:
        raise HTTPException(404, "Unknown batch.")
    if any(f["status"] in ("queued", "processing") for f in store.get_files(batch_id)):
        raise HTTPException(409, "Batch is still processing — cancel it first, then delete.")
    store.delete_batch(b["id"])
    # Path built from the stored id (not the raw URL segment); also removes any leftover audio.
    shutil.rmtree(DATA_DIR / "batches" / b["id"], ignore_errors=True)
    _cancel.discard(b["id"])
    return {"ok": True}


# ---------------------------------------------------------------- results
@app.get("/api/config")
def config_info():
    c = Config()
    return {"backend": c.backend, "model": c.model, "llm_available": llm_available(c),
            "max_samples": c.max_samples, "target_cost_per_min": c.target_cost_per_min}


@app.get("/api/batches")
def batches():
    return store.list_batches()


@app.get("/api/batches/{batch_id}")
def batch_status(batch_id: str):
    b = store.get_batch(batch_id)
    if not b:
        raise HTTPException(404, "Unknown batch.")
    files = store.get_files(batch_id)
    counts = {s: sum(1 for f in files if f["status"] == s) for s in ("pending", "queued", "processing", "done", "error")}
    done = [f for f in files if f["status"] == "done"]
    audio_min = sum(f["meta"]["duration_s"] for f in done) / 60
    cost = sum(f["meta"].get("cost_usd", 0) for f in done)
    proc = sum(f["meta"].get("total_s", 0) for f in done)
    summary = {
        "audio_minutes": round(audio_min, 2),
        "cost_usd": round(cost, 5),
        "cost_per_audio_min": round(cost / audio_min, 5) if audio_min else None,
        "processing_s_per_audio_min": round(proc / audio_min, 2) if audio_min else None,
    }
    pairs = [(f["label"], f["prediction"]) for f in done if f["label"]]
    return {"batch": b, "counts": counts, "total": len(files), "files": files, "summary": summary,
            "metrics": metrics.score(pairs) if pairs else None}


def _rows(batch_id: str) -> list[dict]:
    if not store.get_batch(batch_id):
        raise HTTPException(404, "Unknown batch.")
    return store.get_files(batch_id)


@app.get("/api/batches/{batch_id}/results.csv")
def results_csv(batch_id: str):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name", "result_json", "status", "error"] + FIELDS)
    for f in _rows(batch_id):
        p = f["prediction"] or {}
        w.writerow([f["name"], json.dumps(p) if p else "", f["status"], f["error"] or ""]
                   + [json.dumps(p[k]) if isinstance(p.get(k), bool) else p.get(k, "") for k in FIELDS])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="results_{batch_id}.csv"'})


@app.get("/api/batches/{batch_id}/results.json")
def results_json(batch_id: str):
    out = [{"name": f["name"], "status": f["status"], "error": f["error"], "result_json": f["prediction"]}
           for f in _rows(batch_id)]
    return Response(json.dumps(out, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="results_{batch_id}.json"'})
