"""CFG registry.

Everything that defines a measurement lives here, not in code paths: the ladder
parameters, the lifecycle events, the cue parameters and the seeds. The hash of
this registry is stamped on every reference build, so a number can always be
traced to the configuration that produced it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE_DIR = DATA / "base"          # lossless source scenes (downloaded once)
CORPUS_DIR = DATA / "corpus"      # controlled core set (generated)
REF_DIR = DATA / "reference"      # frozen-at-R0 store + measurements
STATIC_DIR = ROOT / "static"
DEMO_DIR = ROOT / "demo_images"   # images behind the "Demo images" button (override: run.py --demo-dir)

SEED = 20260905                   # one seed registry for ladder, splits, training
FUSION_SEEDS = [11, 23, 37, 51, 73]

# --------------------------------------------------------------------------- cues
CUES = [
    dict(id="C1", name="Error level analysis", short="Error level", cls="compression",
         modality="IMG only", reads="x - JPEG_q(x) residual",
         predicted="Local recompression inconsistency; weak standalone.", breaks="R3-R4"),
    dict(id="C2", name="Block-DCT statistics", short="Block-DCT", cls="compression",
         modality="IMG + VID", reads="8x8 blocks, DCT sparsity on and off the grid",
         predicted="Outlasts C1 but fails together with it.", breaks="after C1"),
    dict(id="C3", name="Quantisation history", short="Quantisation", cls="compression",
         modality="IMG + VID", reads="DCT coefficients, prior quantisation lattice",
         predicted="Most informative substrate cue; most abrupt failure.", breaks="abruptly"),
    dict(id="C4", name="Sensor noise / PRNU", short="Sensor PRNU", cls="sensor",
         modality="IMG + VID", reads="W = x - F(x), PCE(W, K_device)",
         predicted="Survives moderate JPEG; realignment breaks it.", breaks="R7"),
    dict(id="C5", name="Generator resampling", short="Resampling", cls="synthesis",
         modality="IMG + VID", reads="|FFT| of prediction residual, periodic peaks",
         predicted="Most degradation-robust pixel-level cue.", breaks="last of all"),
    dict(id="C6", name="Illumination & geometry", short="Illumination", cls="scene",
         modality="IMG + VID", reads="surface illuminant and shading direction",
         predicted="Near compression-invariant, weak; scene-dependent.", breaks="never"),
    dict(id="C7", name="Temporal-semantic", short="Temporal", cls="temporal",
         modality="VID only", reads="frames[t]: lip-sync, blink, motion",
         predicted="Near-independent of all others: decisive H3 test.", breaks="to be measured"),
    dict(id="C8", name="Container & metadata", short="Container", cls="encapsulation",
         modality="IMG + VID", reads="file bytes: Q-tables, markers, encoder tags",
         predicted="Carried by re-saves; gone after a platform round-trip.", breaks="R8"),
    dict(id="C9", name="Learned erasure detector", short="Erasure net", cls="learned",
         modality="IMG", reads="RGB at 1024 px: texture and structure of filled regions",
         predicted="Trained on LaMa erasures after sharing; strength on other erasers unknown.",
         breaks="to be measured"),
]
CUE_IDS = [c["id"] for c in CUES]
K = len(CUES)
IMAGE_MASKED = {"C7"}             # C7 is video only; the demo is image only

# ------------------------------------------------------------------------- ladder
# Each rung is a deterministic chain of primitive operations. R0-R7 are fully
# determined by this table. R8 in the project is a real upload/re-download; the
# demo substitutes a platform emulator and says so wherever R8 is shown.
LADDER = [
    dict(id="R0", label="PNG / near-lossless JPEG, no resize", short="as shot", ops=[]),
    dict(id="R1", label="single JPEG, high Q", short="JPEG 92",
         ops=[["jpeg", dict(q=92, sub=0)]]),
    dict(id="R2", label="single JPEG, medium Q", short="JPEG 78",
         ops=[["jpeg", dict(q=78, sub=2)]]),
    dict(id="R3", label="double JPEG, aligned grid, mismatched Q-tables", short="JPEG 70 > 88",
         ops=[["jpeg", dict(q=70, sub=2)], ["jpeg", dict(q=88, sub=2)]]),
    dict(id="R4", label="double JPEG, misaligned grid", short="JPEG 80 > shift > 75",
         ops=[["jpeg", dict(q=80, sub=2)], ["shift", dict(dx=4, dy=4)], ["jpeg", dict(q=75, sub=2)]]),
    dict(id="R5", label="resize > JPEG", short="0.7x > JPEG 78",
         ops=[["resize", dict(scale=0.7)], ["jpeg", dict(q=78, sub=2)]]),
    dict(id="R6", label="aggressive JPEG, low Q", short="JPEG 35",
         ops=[["jpeg", dict(q=35, sub=2)]]),
    dict(id="R7", label="multi-generation recompression chain", short="4 generations",
         ops=[["jpeg", dict(q=80, sub=2)], ["resize", dict(scale=0.85)], ["jpeg", dict(q=65, sub=2)],
              ["shift", dict(dx=3, dy=5)], ["jpeg", dict(q=50, sub=2)], ["jpeg", dict(q=70, sub=2)]]),
    dict(id="R8", label="platform upload and re-download (emulated)", short="platform emulator",
         ops=[["resize", dict(scale=0.5)], ["strip", {}], ["jpeg", dict(q=70, sub=2, progressive=True)],
              ["jpeg", dict(q=76, sub=2)]]),
]
RUNG_IDS = [r["id"] for r in LADDER]
R = len(LADDER)

# Lifecycle events: what actually happens to a file once it is out in the world.
# The lifecycle engine applies every ordered arrangement of these up to
# LIFECYCLE_DEPTH, so order effects are measured rather than assumed.
EVENTS = [
    dict(id="save", label="Re-save", ops=[["jpeg", dict(q=85, sub=2)]]),
    dict(id="msg", label="Messenger forward",
         ops=[["resize", dict(scale=0.8)], ["strip", {}], ["jpeg", dict(q=70, sub=2)]]),
    dict(id="shot", label="Screenshot",
         ops=[["crop", dict(l=5, t=3, r=2, b=4)], ["resize", dict(scale=1.12)], ["strip", {}],
              ["jpeg", dict(q=88, sub=2)]]),
    dict(id="post", label="Social post",
         ops=[["resize", dict(scale=0.6)], ["strip", {}], ["jpeg", dict(q=62, sub=2, progressive=True)]]),
]
EVENT_IDS = [e["id"] for e in EVENTS]
LIFECYCLE_DEPTH = 3

# The stability probe: one plain re-save. Whether a cue's decision survives it is
# measured for every sample at every rung, so that trust can be conditioned on it.
PROBE = dict(id="resave", label="plain re-save (JPEG 85)", ops=[["jpeg", dict(q=85, sub=2)]])

# ------------------------------------------------------------------- cue parameters
CUE_PARAMS = dict(
    analysis_cap=1024,            # analysed region: central crop, 8-aligned, at most this side
    min_side=160,
    ela_qualities=[72, 84, 88, 92, 96],   # first entry is the reference quality
    c3_modes=15, c3_qmax=32, c3_fmin=0.45,
    prnu_block=16, prnu_sigma0=2.2, pce_match=60.0,
    c5_window=128, c5_stride=64,
    c6_block=32,
    tail_frac=0.02,
    z_mode="rank", z_clip=5.0, self_norm=True,
)

# ------------------------------------------------------------------ measurement
AUC_FLOOR = 0.60                  # pre-registered survivability floor for R*
MAX_FPR = 0.10                    # thresholds at R0 may flag at most this share of authentic files
EDGE_NULL_Q = 0.95                # theta = this quantile of the independence null at R0
ECE_BINS = 5
VERDICT = dict(fake=0.65, real=0.35, min_trust=0.60)

# Calibration domain of the pixel cues (C1-C6). Their thresholds and trust were fitted
# on 512x512 frames. A much larger frame shows content at a different scale and comes
# from a different camera pipeline, so their trust is reduced with frame size: full at
# up to 2x the reference area, none from 16x (about 4 megapixels) upward.
REF_PIXELS = 512 * 512
PIXEL_CUES = [0, 1, 2, 3, 4, 5]   # C9 is excluded: it works at a fixed 1024 px scale


def pixel_domain(h: int, w: int) -> float:
    import math
    r = (h * w) / (2.0 * REF_PIXELS)
    return float(min(1.0, max(0.0, 1.0 - math.log(max(r, 1.0)) / math.log(8.0))))


def registry() -> dict:
    return dict(seed=SEED, cues=[c["id"] for c in CUES], fusion_seeds=FUSION_SEEDS, ladder=LADDER, events=EVENTS,
                lifecycle_depth=LIFECYCLE_DEPTH, probe=PROBE, cue_params=CUE_PARAMS,
                auc_floor=AUC_FLOOR, max_fpr=MAX_FPR, edge_null_q=EDGE_NULL_Q, ece_bins=ECE_BINS,
                verdict=VERDICT)


def config_hash() -> str:
    blob = json.dumps(registry(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
