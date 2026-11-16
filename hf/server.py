"""Local web server for the demo. Start it with:  python run.py"""
from __future__ import annotations

import json
import os
import re
import traceback
import warnings
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .config import DEMO_DIR, R, RUNG_IDS, STATIC_DIR
from .pipeline import AnalysisError, Engine

warnings.filterwarnings("ignore", category=RuntimeWarning)
MAX_UPLOAD = 40 * 1024 * 1024
app = FastAPI(title="HyperForensics Rev 3 demo", docs_url=None, redoc_url=None)
_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


@app.on_event("startup")
def _warm() -> None:
    engine()


def _stream(gen):
    try:
        for ev in gen:
            yield json.dumps(ev) + "\n"
    except AnalysisError as exc:
        yield json.dumps(dict(type="error", msg=str(exc))) + "\n"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        yield json.dumps(dict(type="error", msg=f"Analysis failed: {type(exc).__name__}: {exc}")) + "\n"


def _rung(rung: int) -> int:
    if not 0 <= rung < R:
        raise HTTPException(400, "rung must be 0-8")
    return rung


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/reference")
def reference():
    return JSONResponse(engine().public_reference(), headers={"Cache-Control": "no-store"})


@lru_cache(maxsize=256)
def _thumb(sid: str, rung: int, side: int) -> bytes:
    m = engine().sample_media(sid, rung)
    h, w = m.rgb.shape[:2]
    s = side / max(h, w)
    img = cv2.resize(m.rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else m.rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()


@app.get("/api/sample/{sid}/thumb")
def sample_thumb(sid: str, rung: int = 0, side: int = 240):
    try:
        return Response(_thumb(sid, _rung(rung), min(640, max(64, side))), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})
    except AnalysisError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/sample/{sid}/file")
def sample_file(sid: str, rung: int = 0):
    """The exact bytes of a demo sample at a rung, so it can be saved and uploaded again."""
    try:
        m = engine().sample_media(sid, _rung(rung))
    except AnalysisError as exc:
        raise HTTPException(404, str(exc)) from exc
    ext = "jpg" if m.data[:2] == b"\xff\xd8" else "png"
    return Response(m.data, media_type="image/jpeg" if ext == "jpg" else "image/png",
                    headers={"Content-Disposition": f'attachment; filename="{sid}_{RUNG_IDS[rung]}.{ext}"'})


@app.post("/api/analyze")
async def analyze(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file larger than 40 MB")
    name = unquote(request.headers.get("x-filename", "upload"))[:120]
    return StreamingResponse(_stream(engine().analyse(data, name)), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ demo images
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def demo_dir() -> Path:
    return Path(os.environ.get("HF_DEMO_DIR") or DEMO_DIR)


def _demo_label(name: str) -> str | None:
    n = name.lower()
    if re.match(r"(org|orig|original|real)[_\- ]", n):
        return "original"
    if re.match(r"(edited|edit|fake|tampered)[_\- ]", n):
        return "edited"
    return None


def _demo_key(name: str):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 10 ** 6, 0 if _demo_label(name) == "original" else 1, name.lower())


def _demo_path(name: str) -> Path:
    d = demo_dir()
    p = (d / name).resolve()
    if p.parent != d.resolve() or p.suffix.lower() not in IMG_EXT or not p.is_file():
        raise HTTPException(404, "no such demo image")
    return p


@app.get("/api/demo")
def demo_list():
    d = demo_dir()
    names = sorted((p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT), key=_demo_key) \
        if d.is_dir() else []
    return JSONResponse(dict(folder=str(d), images=[dict(name=n, label=_demo_label(n)) for n in names]),
                        headers={"Cache-Control": "no-store"})


@lru_cache(maxsize=64)
def _demo_thumb(name: str, mtime: float) -> bytes:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(_demo_path(name))).convert("RGB")
    im.thumbnail((320, 320))
    import io
    b = io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return b.getvalue()


@app.get("/api/demo/{name}/thumb")
def demo_thumb(name: str):
    p = _demo_path(name)
    return Response(_demo_thumb(name, p.stat().st_mtime), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=600"})


@app.post("/api/analyze_demo")
def analyze_demo(name: str):
    p = _demo_path(name)
    gen = engine().analyse(p.read_bytes(), p.name, user_label=_demo_label(p.name))
    return StreamingResponse(_stream(gen), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/analyze_sample")
def analyze_sample(id: str, rung: int = 0):
    eng = engine()
    try:
        m = eng.sample_media(id, _rung(rung))
    except AnalysisError as exc:
        raise HTTPException(404, str(exc)) from exc
    ext = "jpg" if m.data[:2] == b"\xff\xd8" else "png"
    gen = eng.analyse(m.data, f"{id}_{RUNG_IDS[rung]}.{ext}", sample=dict(id=id, rung=rung))
    return StreamingResponse(_stream(gen), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
