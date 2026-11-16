"""Build the erasure training set from COCO val2017 photos (Hugging Face mirror).

For every photo: one object is erased with LaMa (object-shaped mask from GrabCut inside
the COCO box, plus a brush margin). The original and the edited copy then go through
identical later steps, so the only difference between them is the erasure:
  editor save  : edited always; originals half of the time (same quality range)
  sharing step : the same for both copies of a pair (JPEG 60-85, metadata stripped,
                 sometimes resized), skipped for 20% of pairs

    python -m hf.erase.make_data [--limit N]
"""
from __future__ import annotations

import io
import json
import sys
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from ..config import DATA
from .lama import Lama

SRC = sorted((DATA / "erase" / "dl" / "coco" / "data").glob("val-*.parquet"))
OUT = DATA / "erase" / "pairs"
N_VAL = 400


def jpeg(rgb, q, sub=2, progressive=False):
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=int(q), subsampling=sub, progressive=progressive)
    return b.getvalue()


def dec(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def object_mask(rgb, box, rng):
    """Object-shaped mask from a box: GrabCut, falling back to an ellipse."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    m = np.zeros((H, W), np.uint8)
    rect = (int(x0), int(y0), max(2, int(x1 - x0)), max(2, int(y1 - y0)))
    ok = False
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        gc = np.zeros((H, W), np.uint8)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        frac = m.sum() / max(1, rect[2] * rect[3])
        ok = 0.2 < frac < 0.95
    except cv2.error:
        pass
    if not ok:
        m[:] = 0
        cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (max(2, int((x1 - x0) / 2)), max(2, int((y1 - y0) / 2))),
                    0, 0, 360, 1, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:                                    # keep the largest blob, like a single brush selection
        k = 1 + int(np.argmax(st[1:, 4]))
        m = (lab == k).astype(np.uint8)
    r = int(rng.integers(4, 16))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if rng.random() < 0.5:                       # ragged brush edge
        noise = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 6)
        edge = cv2.dilate(m, np.ones((9, 9), np.uint8)) - cv2.erode(m, np.ones((9, 9), np.uint8))
        m = np.where(edge > 0, (noise > 0).astype(np.uint8), m)
    return m.astype(bool)


def pick_object(row, H, W, rng):
    ob = row["objects"]
    cands = []
    for bb, area in zip(ob["bbox"], ob["area"]):
        x0, y0, x1, y1 = bb
        if x1 <= x0 or y1 <= y0:
            continue
        f = (x1 - x0) * (y1 - y0) / (H * W)
        if 0.015 < f < 0.30 and (x1 - x0) < 0.85 * W and (y1 - y0) < 0.85 * H:
            cands.append((x0, y0, x1, y1))
    if not cands:
        return None
    return cands[int(rng.integers(0, len(cands)))]


def main(limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train").mkdir(exist_ok=True)
    (OUT / "val").mkdir(exist_ok=True)
    lama = Lama()
    rng = np.random.default_rng(20261116)
    index, t0, n = [], time.time(), 0
    rows = []
    for p in SRC:
        rows += pq.read_table(p, columns=["image_id", "image", "width", "height", "objects"]).to_pylist()
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    for i, row in enumerate(rows):
        rgb = dec(row["image"]["bytes"])
        H, W = rgb.shape[:2]
        if min(H, W) < 256:
            continue
        box = pick_object(row, H, W, rng)
        if box is None:
            continue
        mask = object_mask(rgb, box, rng)
        if not 0.01 < mask.mean() < 0.40:
            continue
        # erase: sometimes at reduced resolution and upsampled back, as phone apps do
        s = 1.0 if rng.random() < 0.5 else float(rng.uniform(0.45, 0.8))
        if s < 1.0:
            sm = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            mm = cv2.resize(mask.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            fill = cv2.resize(lama(sm, mm), (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            fill = lama(rgb, mask)
        a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(rng.uniform(0.6, 2.5)))[..., None]
        a = np.maximum(a, mask[..., None].astype(np.float32))
        edited = np.clip(rgb * (1 - a) + fill * a, 0, 255).astype(np.uint8)

        # editor save (edited always, originals half the time), same quality range for both
        eq = int(rng.integers(88, 99))
        edited = dec(jpeg(edited, eq, sub=int(rng.choice([0, 2]))))
        orig = rgb if rng.random() < 0.5 else dec(jpeg(rgb, int(rng.integers(88, 99)), sub=int(rng.choice([0, 2]))))

        # sharing step, identical for both copies
        share = rng.random() < 0.8
        if share:
            q = int(np.clip(rng.normal(74, 7), 55, 90))
            prog = bool(rng.random() < 0.5)
            if rng.random() < 0.3:
                f = float(rng.uniform(0.6, 0.95))
                sz = (int(W * f), int(H * f))
                orig = cv2.resize(orig, sz, interpolation=cv2.INTER_AREA)
                edited = cv2.resize(edited, sz, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
            ob, eb = jpeg(orig, q, progressive=prog), jpeg(edited, q, progressive=prog)
        else:
            q = None
            ob, eb = jpeg(orig, 95), jpeg(edited, 95)
        split = "val" if n < N_VAL else "train"
        stem = f"{split}/{row['image_id']:012d}"
        (OUT / f"{stem}_o.jpg").write_bytes(ob)
        (OUT / f"{stem}_e.jpg").write_bytes(eb)
        cv2.imwrite(str(OUT / f"{stem}_m.png"), mask.astype(np.uint8) * 255)
        index.append(dict(stem=stem, split=split, share_q=q, erase_scale=s, area=float(mask.mean())))
        n += 1
        if n % 250 == 0:
            print(f"  {n} pairs ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "index.json").write_text(json.dumps(index))
    print(f"done: {n} pairs ({sum(1 for x in index if x['split'] == 'val')} val) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(lim)
