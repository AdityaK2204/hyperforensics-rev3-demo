"""Offline reference build: corpus -> score tensor -> measurements -> fusion arms.

    python -m hf.build            build whatever is missing
    python -m hf.build --force    rebuild everything from the source scenes

Stage 1  controlled core set (corpus.py)
Stage 2  score tensor  S[n, r, k] = s_k(x_n^(r)),  damage features F[n, r, :],
         and the probe tensor Sp[n, r, k]: the same scores after one plain re-save
Stage 3  thresholds, errors, survival, dependence, topology, damage hub, trust
Stage 4  three fusion arms, trained and evaluated per rung
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import corpus, cues, ladder
from .config import CORPUS_DIR, K, PROBE, R, REF_DIR, config_hash

_DEV = None


def _score_sample(rec: dict):
    """Worker: take one sample to all nine rungs, run the battery at each, then probe each."""
    global _DEV
    import cv2
    cv2.setNumThreads(1)
    if _DEV is None:
        _DEV = corpus.load_devices()
    m0 = ladder.Media.from_bytes((CORPUS_DIR / rec["file"]).read_bytes())
    dev = _DEV.get(rec["device_id"]) if rec["device_id"] else None
    S = np.full((R, K), np.nan, dtype=np.float32)
    Sp = np.full((R, K), np.nan, dtype=np.float32)
    F = np.zeros((R, len(cues.DAMAGE_NAMES)), dtype=np.float32)
    for r in range(R):
        m = ladder.rung(m0, r)
        out = cues.extract(m, device=dev)
        S[r], F[r] = out["scores"], out["damage"]
        Sp[r] = cues.extract(ladder.apply_ops(m, PROBE["ops"]), device=dev)["scores"]
    return rec["id"], S, F, Sp


def stage_scores(records: list[dict], log=print):
    t0 = time.time()
    n = len(records)
    S = np.zeros((n, R, K), dtype=np.float32)
    Sp = np.zeros((n, R, K), dtype=np.float32)
    F = np.zeros((n, R, len(cues.DAMAGE_NAMES)), dtype=np.float32)
    pos = {rec["id"]: i for i, rec in enumerate(records)}
    workers = max(1, min(12, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j, (sid, s, f, sp) in enumerate(pool.map(_score_sample, records, chunksize=4)):
            S[pos[sid]], F[pos[sid]], Sp[pos[sid]] = s, f, sp
            if (j + 1) % 48 == 0 or j + 1 == n:
                log(f"  score tensor: {j + 1}/{n} samples x {R} rungs  ({time.time() - t0:.0f}s)")
    stage_learned(records, S, Sp, log)
    np.savez_compressed(REF_DIR / "scores.npz", S=S, F=F, Sp=Sp,
                        ids=np.array([r["id"] for r in records]), config=config_hash())
    return S, F, Sp


def stage_learned(records: list[dict], S, Sp, log=print):
    """C9 on every state of every sample, in this process (it runs on the GPU)."""
    if K < 9 or not cues.c9_available():
        log("  C9: erasure network weights not found, cue masked")
        return
    t0 = time.time()
    for i, rec in enumerate(records):
        m0 = ladder.Media.from_bytes((CORPUS_DIR / rec["file"]).read_bytes())
        for r in range(R):
            m = ladder.rung(m0, r)
            S[i, r, 8] = cues.c9_learned(m.rgb)[0]
            Sp[i, r, 8] = cues.c9_learned(ladder.apply_ops(m, PROBE["ops"]).rgb)[0]
        if (i + 1) % 96 == 0 or i + 1 == len(records):
            log(f"  C9: {i + 1}/{len(records)} samples ({time.time() - t0:.0f}s)")


def load_scores(records: list[dict]):
    p = REF_DIR / "scores.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "Sp" not in z.files or str(z["config"]) != config_hash() \
            or list(z["ids"]) != [r["id"] for r in records]:
        return None
    return z["S"], z["F"], z["Sp"]


def build(force: bool = False, log=print) -> dict:
    from . import fusion, measure

    REF_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if force or not (CORPUS_DIR / "corpus.json").exists() or not (REF_DIR / "devices.npz").exists():
        log("[1/4] controlled core set: devices, flat fields, captures, manipulations")
        records = corpus.build_corpus(log)
        force = True
    else:
        records = corpus.load_corpus()
    cached = None if force else load_scores(records)
    if cached is None:
        log(f"[2/4] score tensor: {len(records)} samples x {R} rungs x {K} cues, plus the re-save probe")
        S, F, Sp = stage_scores(records, log)
    else:
        S, F, Sp = cached
    log("[3/4] thresholds, survival, dependence, topology, damage hub, trust")
    ref = measure.measure(records, S, F, Sp)
    log("[4/4] fusion arms: dependence graph / fully connected / independence")
    fusion.train_and_evaluate(records, S, F, Sp, ref, log)
    ref["config_hash"] = config_hash()
    ref["built_at"] = time.strftime("%Y-%m-%d %H:%M")
    ref["build_seconds"] = round(time.time() - t0, 1)
    (REF_DIR / "reference.json").write_text(json.dumps(ref))
    log(f"reference built in {time.time() - t0:.0f}s  (config {config_hash()})")
    return ref


def ready() -> bool:
    need = ["reference.json", "frozen.npz", "fusion.pt", "scores.npz", "devices.npz"]
    if not all((REF_DIR / n).exists() for n in need):
        return False
    try:
        return json.loads((REF_DIR / "reference.json").read_text()).get("config_hash") == config_hash()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    build(force="--force" in sys.argv)
