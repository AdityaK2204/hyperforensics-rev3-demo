# HyperForensics Rev 3 demo (images)

A working implementation of the Rev 3 architecture for still images: give it a file, and it returns
a verdict, the cues that decided it, where in the frame the evidence is, and how long that verdict
would survive if the file kept being shared.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python run.py --open
```

The demo opens at http://127.0.0.1:8765. The reference is already built in `data/`. If `data/` is
missing, the first run downloads the 24 Kodak lossless photographs (about 15 MB) and builds
everything, which takes roughly ten minutes once.

`python run.py --rebuild` regenerates the corpus and every measurement from scratch.

**Open demo images** on the home screen lists the photos in `demo_images/` (or the folder given
with `--demo-dir`). Files named `org_…` or `edited_…` get a label badge, and the result shows
whether the verdict agrees with that label. The label is display only: the analysis never reads
the file name.

## How a file is analysed

Two pipelines run on every file and meet at the fusion node.

**Pipeline A, lifecycle.** The file is degraded here, by us: down the nine-rung ladder R0 to R8, and
through every ordering of four everyday events (re-save, messenger forward, screenshot, social
post) up to three deep, which is 40 histories. All cues are read at every step. This gives how the
file's evidence moves, which cues move together, where the file already sits on the ladder, and
whether each cue's decision survives one plain re-save.

**Pipeline B, evidence.** The cue battery runs on the file as received: one score per cue, a
decision against a threshold frozen at R0, and a localisation map.

**Fusion.** Pipeline A's damage estimate and re-save probe select each cue's trust and the edge set
from the reference measurements. A small graph network reads pipeline B's evidence through those
edges and returns the probability of manipulation. The verdict is one of three: manipulated or
synthetic, no manipulation evidence, or inconclusive when the file is too damaged for silence to
mean anything.

## Where each block of the architecture diagram lives

| Diagram block | Code |
|---|---|
| 01 Data plane, controlled core set, splits | `hf/corpus.py` |
| 02 Degradation engine, ladder R0 to R8, lifecycle orderings | `hf/ladder.py`, parameters in `hf/config.py` |
| 03 Cue battery C1 to C8, score tensor | `hf/cues.py`, `hf/codec.py` (C8), `hf/build.py` |
| 04 Decisions, errors, survival, R* | `hf/measure.py` |
| 05 Dependence per rung (Q, disagreement, correlation) | `hf/measure.py` |
| 06 Topology rule and the three fusion arms | `hf/measure.py` (edges), `hf/fusion.py` |
| 07 Evaluation and the four reported artefacts | Measurements tab, `data/reference/reference.json` |
| Control plane (config, seeds, frozen-at-R0 store) | `hf/config.py`, `data/reference/frozen.npz`, System tab |
| Damage estimator and cue reliability model | `hf/measure.py` (`DamageHub`, `trust_table`) |
| Online path and web interface | `hf/pipeline.py`, `hf/server.py`, `static/index.html` |

## What the cues are

| Cue | Reads | Implementation |
|---|---|---|
| C1 Error level analysis | re-save residual at five qualities | regional inconsistency of the residual's shape across qualities |
| C2 Block-DCT statistics | DCT sparsity on and off the 8x8 grid | regions whose grid is missing or shifted |
| C3 Quantisation history | prior quantisation lattice in low-frequency DCT modes | blocks that do not sit on the lattice the rest of the frame obeys |
| C4 Sensor noise / PRNU | noise residual against a device fingerprint | PCE for presence, block correlation for regions |
| C5 Generator resampling | spectrum of a linear-prediction residual | periodic peaks off the JPEG lattice, per region and whole frame |
| C6 Illumination | chromaticity of near-neutral surfaces | regional disagreement in illuminant colour |
| C7 Temporal-semantic | video only | masked for images |
| C8 Container and metadata | file bytes | editor and generator tags, date mismatch, camera provenance |

## What was measured on the toy corpus

432 files (192 authentic, 240 manipulated or synthetic) from 48 scenes, each taken to all nine
rungs. Thresholds come from the calibration split, fusion weights from the training split, and the
numbers below from the 144 held-out test files. One corpus seed, five fusion seeds.

| Result | Value |
|---|---|
| R* per cue | C1 R2, C2 R2, C3 R1, C4 R7, C5 R1, C6 R0, C8 R8 |
| H1, cues break at different rungs | supported |
| H2, C1 to C3 fail together | supported: error correlation 0.58 inside the block at R0 |
| H3, dependence graph beats the other arms under heavy damage | not supported: 0.03 to 0.04 AUC worse on R5 to R8 |
| Fused AUC, dependence graph | 0.97 at R0, 0.82 at R7, 0.52 at R8 |
| Fused AUC without the container cue | 0.91 at R0, 0.56 at R7 |
| Damage estimator | right rung for 92% of held-out file states |

## AI object removal (erasing) and the learned cue C9

Photos erased on a phone and then shared through a messenger lose their metadata and are
recompressed at about quality 75. None of the eight hand-built cues, and not TruFor (tested
and dropped), separates an erased copy from its original after that. C9 is a detector trained
for exactly this case.

**Training data** (`python -m hf.erase.make_data`): 4,106 COCO val2017 photos (Hugging Face
mirror). In each, one object is erased with LaMa (object-shaped GrabCut mask from the COCO box,
plus a brush margin, sometimes inpainted at reduced resolution as phone apps do). The original
and the erased copy then get identical later steps: an editor-style re-save, then the same
sharing step (quality 55-90, metadata stripped, sometimes resized). The only difference between
the two copies is the erasure. 400 pairs are held out.

**Network** (`python -m hf.erase.train`): ResNet-34 (ImageNet) with a light FPN decoder; it
outputs a map of filled-in regions and an image score pooled from the strongest 1% of the map.
It reads photos at up to 1024 px. Held-out COCO AUC 0.95; at 5% false alarms it catches 83% of
erasures. A second run with phone-camera augmentation reached 0.936 and was discarded by a rule
fixed before its result was known (keep only if within 0.01).

**Calibration** (`python -m hf.erase.calibrate`): C9's threshold, evidence scale and trust are
measured on the held-out COCO pairs taken down the full ladder, not on the 512 px controlled set,
which is mostly splices that C9 is not built to see. Trust is indexed by the damage level the
estimator reads, as at analysis time. AUC 0.93 at R0, 0.92 at R4, 0.66 at R8.

**On the 14 demo photos** (never used for training, calibration or model selection; looked at
three times while fixing two integration bugs, so not a fully clean test):

| | Edited (7) | Original (7) |
|---|---|---|
| Overall verdict | 4 manipulated (pairs 2, 3, 5, 6), 3 inconclusive | 1 no evidence (pair 2), 6 inconclusive, 0 called manipulated |
| Erasure check (C9) | found in 4 (2, 3, 5, 6), missed in 1 (pair 1), unreliable in 2 (7, 8) | none found in 4 (1, 2, 3, 5), false alarm in 1 (pair 6), unreliable in 2 (7, 8) |

Originals stay "inconclusive" overall because no erasure found is not proof of authenticity:
a spliced photo would also show no erasure, and on large shared photos nothing else can be
checked. The verdict panel shows the erasure check separately, with how often C9 is right in
that situation. Pairs 7 and 8 are a tiny erasure in a sky and a dark scene; there C9's decision
changes under a single re-save, and it is measured as unreliable in that case.

Earlier provenance work still applies: C8 reads AI-edit provenance (IPTC Digital Source Type,
Apple "Clean Up", Google and Samsung erase tools) and catches unshared phone edits directly.

## Limits you should state when presenting this

- **The corpus is simulated.** Four simulated cameras (fingerprint, JPEG tables, EXIF) and
  scripted manipulations on Kodak photographs stand in for the team phones and consented fakes. The
  numbers show the measurement pipeline works end to end. They are not findings about real media.
- **R8 is an emulator**, not a real upload and re-download.
- **Metadata carries the fused result under damage.** Without C8 the fused AUC falls to 0.56 by R7.
- **C6 never worked.** The classical illuminant cue is at chance from R0; the report already says
  this cue needs a learned estimator.
- **C5 reads resampling, not generators.** The synthetic images come from a toy generator that
  upsamples by non-integer factors. Real generator output may leave no such trace.
- **C4 needs a registered camera.** On any upload that does not claim or match one of the four
  simulated devices it is masked.
- **Your own uploads are outside the calibrated set.** Authentic web copies are in the corpus so
  that a missing fingerprint or a resize is not read as forgery, but thresholds and trust were
  still fitted on simulated data. An editing-software tag alone is enough for C8 to fire.
- **No public datasets and no DeepfakeBench**, so there is no cross-dataset number.

## Extending to video

`ladder.Media` is one state of a file and `cues.extract` returns scores, maps and damage features
for it. A video path needs a clip-first `Media` that carries frames and the audio track, video
chains for each rung in `config.LADDER`, the C7 implementation, C1 masked, and video versions of C2,
C3 and C8. Nothing in `measure.py` or `fusion.py` depends on the modality.
