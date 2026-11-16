"""After the erasure network changes: re-score C9 on the controlled set, recalibrate C9 on
held-out COCO pairs, then re-measure and refit the fusion.

    python -m hf.erase.refresh
"""
from __future__ import annotations

import numpy as np

from .. import build, corpus
from ..config import REF_DIR, config_hash
from . import calibrate


def main():
    recs = corpus.load_corpus()
    z = np.load(REF_DIR / "scores.npz", allow_pickle=False)
    S, F, Sp = z["S"].copy(), z["F"], z["Sp"].copy()
    print("[1/3] C9 on the controlled set")
    build.stage_learned(recs, S, Sp)
    np.savez_compressed(REF_DIR / "scores.npz", S=S, F=F, Sp=Sp, ids=np.array([r["id"] for r in recs]),
                        config=config_hash())
    print("[2/3] C9 calibration on held-out COCO pairs")
    calibrate.main()
    print("[3/3] measurements and fusion")
    build.build()


if __name__ == "__main__":
    main()
