"""Calibrate C9 on data from its own operating domain.

The controlled set is 512 px frames, mostly splices and copy-moves; C9 is an erasure
detector for shared photos read at up to 1024 px. Its threshold, evidence scale and trust
table are therefore measured on the held-out COCO pairs (the val split, never used for
training), taken down the same nine-rung ladder with the same re-save probe.

Trust is indexed by the rung the damage estimator *reads* for each state (posterior
weighted), because that is how trust is looked up when a file is analysed.

    python -m hf.erase.calibrate         (after hf.build has fitted the damage hub data)
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import cues, ladder
from ..config import MAX_FPR, PROBE, R, REF_DIR
from .make_data import OUT

CAL = REF_DIR / "c9_calibration.npz"


def _damage(path: str):
    import cv2
    cv2.setNumThreads(1)
    m0 = ladder.Media.from_bytes(open(path, "rb").read())
    F = []
    for r in range(R):
        F.append(cues.extract(ladder.rung(m0, r))["damage"])
    return path, np.stack(F)


def main():
    from ..build import load_scores
    from ..corpus import load_corpus
    from ..measure import DamageHub, youden_threshold
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    val = [x for x in json.loads((OUT / "index.json").read_text()) if x["split"] == "val"]
    items = [(str(OUT / f"{x['stem']}_o.jpg"), 0) for x in val] + [(str(OUT / f"{x['stem']}_e.jpg"), 1) for x in val]
    paths = [p for p, _ in items]
    y = np.array([l for _, l in items])
    n = len(items)
    F = np.zeros((n, R, len(cues.DAMAGE_NAMES)), np.float32)
    pos = {p: i for i, p in enumerate(paths)}
    with ProcessPoolExecutor(max_workers=max(1, min(10, (os.cpu_count() or 4) - 4))) as pool:
        for j, (p, f) in enumerate(pool.map(_damage, paths, chunksize=8)):
            F[pos[p]] = f
            if (j + 1) % 200 == 0:
                print(f"  damage features {j + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    S = np.zeros((n, R), np.float32)
    Sp = np.zeros((n, R), np.float32)
    for i, p in enumerate(paths):
        m0 = ladder.Media.from_bytes(open(p, "rb").read())
        for r in range(R):
            m = ladder.rung(m0, r)
            S[i, r] = cues.c9_learned(m.rgb)[0]
            Sp[i, r] = cues.c9_learned(ladder.apply_ops(m, PROBE["ops"]).rgb)[0]
        if (i + 1) % 200 == 0:
            print(f"  C9 scores {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    # the same damage hub the analysis uses (fitted on the controlled set)
    recs = load_corpus()
    _, Fc, _ = load_scores(recs)
    split = np.array([r["split"] for r in recs])
    scene = np.array([r["scene"][:3] for r in recs])
    hub = DamageHub(Fc, scene, (split == "train") | (split == "calib"), with_oof=False)
    post = hub.posterior(F.reshape(n * R, -1)).reshape(n, R, R)       # [image, applied rung, read rung]

    tau = youden_threshold(S[:, 0], y, MAX_FPR)
    D, Dp = S >= tau, Sp >= tau
    right = (D == (y[:, None] == 1)).astype(np.float64)
    persist = (Dp == D)
    w_cls = np.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())[:, None]
    T = np.full((R, 2, 2), 0.5)
    trust = np.zeros(R)
    for rr in range(R):                                   # rung as the damage estimator reads it
        w = post[:, :, rr] * w_cls                        # (n, applied rung)
        base = float((w * right).sum() / max(w.sum(), 1e-9))
        for dv in (0, 1):
            sd = D == bool(dv)
            wd = w * sd
            rate_d = float((wd * right).sum() / wd.sum()) if wd.sum() > 0 else base
            t_d = rate_d
            for bv in (0, 1):
                sb = sd & (persist == bool(bv))
                wb = w * sb
                rate = float((wb * right).sum() / wb.sum()) if wb.sum() > 0 else t_d
                eff = wb.sum() / max(w.max(), 1e-12)      # effective sample count
                T[rr, dv, bv] = (eff * rate + 6.0 * t_d) / (eff + 6.0)
        # informedness at this read rung
        tp = (w * (D & (y[:, None] == 1))).sum() / max((w * (y[:, None] == 1)).sum(), 1e-9)
        tn = (w * (~D & (y[:, None] == 0))).sum() / max((w * (y[:, None] == 0)).sum(), 1e-9)
        trust[rr] = max(0.0, float(tp + tn - 1))
    lr = LogisticRegression(C=1.0).fit(S[:, 0, None], y)
    llr = np.array([float(lr.coef_[0, 0]), float(lr.intercept_[0])])
    scale = max(0.3, float(np.std(S[y == 0, 0])))
    from ..measure import auc
    auc_r = [auc(S[:, r], y) for r in range(R)]
    read_share = post.mean(axis=0)                        # how often each applied rung is read as each rung
    np.savez_compressed(CAL, tau=tau, T=T, trust=trust, llr=llr, scale=scale, auc=np.array(auc_r),
                        n=n, read_share=read_share)
    print("tau", round(float(tau), 3), "AUC per applied rung", [round(a, 3) for a in auc_r])
    print("trust per read rung", trust.round(2).tolist())
    print("T fired/persist per read rung", T[:, 1, 1].round(2).tolist())
    print("T quiet/persist per read rung", T[:, 0, 1].round(2).tolist())
    print(f"done in {time.time() - t0:.0f}s -> {CAL}")


if __name__ == "__main__":
    main()
