"""The online path: one file in, one verdict out.

Two pipelines run on every file and meet at the fusion node.

  Pipeline A  lifecycle     Degrade the file ourselves: the nine-rung ladder, and
                            every ordering of four real-world events. Read all cues
                            at every step. What comes out is how this file's
                            evidence moves, which cues move together, where the
                            file already sits on the ladder, and whether each
                            cue's decision survives a plain re-save.
  Pipeline B  evidence      Run the cue battery on the file as received: scores,
                            frozen decisions, localisation maps.
  Fusion                    A's damage estimate and stability probe select each
                            cue's trust and the edge set from the reference
                            measurements; the graph network reads B's evidence
                            through them and returns p(fake | x).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata

from . import build, codec, corpus, cues, fusion, ladder, measure
from .config import (CORPUS_DIR, CUE_IDS, CUE_PARAMS, CUES, EVENTS, EVENT_IDS, K, LADDER,
                     LIFECYCLE_DEPTH, PROBE, R, REF_DIR, RUNG_IDS, VERDICT, config_hash, pixel_domain,
                     registry)


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------------- helpers
def _orient(a: np.ndarray, o: int) -> np.ndarray:
    """Apply an EXIF orientation to an image or a map, for display only."""
    if o == 2:
        a = a[:, ::-1]
    elif o == 3:
        a = a[::-1, ::-1]
    elif o == 4:
        a = a[::-1]
    elif o == 5:
        a = np.swapaxes(a, 0, 1)
    elif o == 6:
        a = np.swapaxes(a, 0, 1)[:, ::-1]
    elif o == 7:
        a = np.swapaxes(a, 0, 1)[::-1, ::-1]
    elif o == 8:
        a = np.swapaxes(a, 0, 1)[::-1]
    return np.ascontiguousarray(a)


def _jpeg_url(rgb: np.ndarray, side: int, q: int = 80) -> str:
    h, w = rgb.shape[:2]
    s = side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _map_payload(m: np.ndarray, lo: float, hi: float, o: int, size: int = 64):
    """A cue map as a small 8-bit grid the interface can colour and overlay."""
    if m is None or m.size <= 1:
        return None
    v = np.clip((m.astype(np.float32) - lo) / (hi - lo), 0, 1)
    h, w = v.shape
    s = size / max(h, w)
    interp = cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA
    v = cv2.resize(v, (max(2, int(round(w * s))), max(2, int(round(h * s)))), interpolation=interp)
    v = _orient(v, o)
    return dict(w=int(v.shape[1]), h=int(v.shape[0]),
                data=base64.b64encode((v * 255).astype(np.uint8).tobytes()).decode())


def _canvas(m: np.ndarray, region, full_hw, lo: float, hi: float, size: int = 96) -> np.ndarray:
    """A cue map computed on `region`, normalised to 0-1 and placed on a grid covering the whole frame."""
    H, W = full_hw
    y0, x0, h, w = region
    s = size / max(H, W)
    gh, gw = max(2, int(round(H * s))), max(2, int(round(W * s)))
    canvas = np.zeros((gh, gw), dtype=np.float32)
    v = np.clip((m.astype(np.float32) - lo) / (hi - lo), 0, 1)
    ry, rx = min(gh - 1, int(round(y0 * s))), min(gw - 1, int(round(x0 * s)))
    rh, rw = max(1, min(gh - ry, int(round(h * s)))), max(1, min(gw - rx, int(round(w * s))))
    canvas[ry:ry + rh, rx:rx + rw] = cv2.resize(v, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return canvas


def _encode(canvas: np.ndarray, o: int):
    c = _orient(canvas, o)
    return dict(w=int(c.shape[1]), h=int(c.shape[0]),
                data=base64.b64encode((np.clip(c, 0, 1) * 255).astype(np.uint8).tobytes()).decode())


def _full_map(m: np.ndarray, region, full_hw, lo: float, hi: float, o: int, size: int = 96):
    if m is None or m.size <= 1:
        return None
    return _encode(_canvas(m, region, full_hw, lo, hi, size), o)


def _region_box(region, full_hw, o: int):
    """The analysed region as fractions [x, y, w, h] of the displayed (oriented) frame."""
    H, W = full_hw
    y0, x0, h, w = region
    g = np.zeros((200, max(2, int(200 * W / H))) if H >= W else (max(2, int(200 * H / W)), 200), dtype=bool)
    sy, sx = g.shape[0] / H, g.shape[1] / W
    g[int(y0 * sy):int(np.ceil((y0 + h) * sy)), int(x0 * sx):int(np.ceil((x0 + w) * sx))] = True
    g = _orient(g, o)
    ys, xs = np.nonzero(g)
    return [round(xs.min() / g.shape[1], 4), round(ys.min() / g.shape[0], 4),
            round((xs.max() + 1 - xs.min()) / g.shape[1], 4), round((ys.max() + 1 - ys.min()) / g.shape[0], 4)]


def _spearman(X: np.ndarray) -> np.ndarray:
    """Rank correlation between columns; NaN where a column is masked or constant."""
    k = X.shape[1]
    out = np.full((k, k), np.nan)
    ranks = {}
    for j in range(k):
        c = X[:, j]
        if np.isnan(c).any() or np.ptp(c) < 1e-9:
            continue
        ranks[j] = rankdata(c)
    for i in ranks:
        for j in ranks:
            out[i, j] = float(np.corrcoef(ranks[i], ranks[j])[0, 1])
    return out


def _f(v, nd=3):
    """JSON-safe float."""
    if v is None:
        return None
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, nd)


def _fl(a, nd=3):
    return [_f(v, nd) for v in np.asarray(a).ravel().tolist()]


# ------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, log=print):
        t0 = time.time()
        self.ref = json.loads((REF_DIR / "reference.json").read_text())
        self.records = corpus.load_corpus()
        self.by_id = {r["id"]: r for r in self.records}
        self.devices = corpus.load_devices()
        self.fuser = fusion.Fuser()
        self.frozen = self.fuser.frozen
        S, F, _ = build.load_scores(self.records)
        split = np.array([r["split"] for r in self.records])
        scene = np.array([r["scene"][:3] for r in self.records])
        self.hub = measure.DamageHub(F, scene, (split == "train") | (split == "calib"), with_oof=False)
        self.pool = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) - 2)))
        cv2.setNumThreads(1)
        log(f"engine ready in {time.time() - t0:.1f}s  (config {config_hash()}, {len(self.records)} reference samples)")

    # ------------------------------------------------------------------ public
    def public_reference(self) -> dict:
        ref = dict(self.ref)
        ref["cues"] = CUES
        ref["ladder"] = [dict(id=r["id"], label=r["label"], short=r["short"]) for r in LADDER]
        ref["events"] = [dict(id=e["id"], label=e["label"]) for e in EVENTS]
        ref["probe"] = dict(id=PROBE["id"], label=PROBE["label"])
        ref["registry"] = registry()
        ref["devices"] = [{k: d[k] for k in ("id", "make", "model", "q")} for d in self.devices.values()]
        ref["verdict_rule"] = VERDICT
        gal = []
        for g in self.ref.get("gallery", []):
            rec = self.by_id.get(g["id"])
            if rec:
                gal.append(dict(g, rung_id=RUNG_IDS[g["rung"]], label=corpus.FAMILY_LABEL[rec["family"]]))
        ref["gallery"] = gal
        return ref

    def sample_media(self, sid: str, rung: int = 0) -> ladder.Media:
        rec = self.by_id.get(sid)
        if rec is None:
            raise AnalysisError(f"unknown sample: {sid}")
        m = ladder.Media.from_bytes((CORPUS_DIR / rec["file"]).read_bytes())
        return ladder.rung(m, int(rung))

    def identify_device(self, m: ladder.Media, cont: dict):
        ex = cont["exif"]
        for dev in self.devices.values():
            if ex.get("Make") == dev["make"] and ex.get("Model") == dev["model"]:
                return dev, f"device {dev['id']} claimed by EXIF ({dev['make']} {dev['model']})"
        if min(m.rgb.shape[:2]) < CUE_PARAMS["min_side"]:
            return None, "no registered fingerprint matches this file"
        f = cues.Frame(m.rgb)
        W = cues.noise_residual(f.Y)
        best = (0.0, None)
        for dev in self.devices.values():
            pce = cues.prnu_align(f, dev, W)[0]
            if pce > best[0]:
                best = (pce, dev)
        if best[1] is not None and best[0] >= CUE_PARAMS["pce_match"]:
            return best[1], f"device {best[1]['id']} matched blind by fingerprint (PCE {best[0]:,.0f})"
        return None, "no registered device fingerprint matches this file"

    # ----------------------------------------------------------------- analyse
    def analyse(self, data: bytes, name: str, sample: dict | None = None, user_label: str | None = None):
        """Generator: yields progress events, then one result event.

        user_label is what a demo image's file name says it is ('original' or 'edited'). It is
        only echoed back for display; nothing in the analysis reads it."""
        t0 = time.time()
        yield dict(type="progress", stage="read", pct=3, msg="Reading the container")
        try:
            m0 = ladder.Media.from_bytes(data)
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError("This file could not be decoded as an image (JPEG, PNG, WebP, BMP, TIFF).") from exc
        H, W = m0.rgb.shape[:2]
        if min(H, W) < CUE_PARAMS["min_side"]:
            raise AnalysisError(f"Image is {W}x{H}; the cues need at least {CUE_PARAMS['min_side']} px on the short side.")
        cont = codec.read_container(data)
        try:
            orient = int(Image.open(io.BytesIO(data)).getexif().get(274, 1))
        except Exception:  # noqa: BLE001
            orient = 1
        if sample is not None:
            rec = self.by_id[sample["id"]]
            device = self.devices.get(rec["device_id"]) if rec["device_id"] else None
            dev_note = f"device {rec['device_id']} from the sample record" if device else \
                "the sample record names no device"
        else:
            rec = None
            device, dev_note = self.identify_device(m0, cont)

        # ---------------- Pipeline B: evidence on the file as received
        yield dict(type="progress", stage="cues", pct=10, msg=f"Pipeline B: reading {K} cues on the file as received")
        use_c9 = cues.c9_available()
        out0 = cues.extract(m0, device=device, want_maps=True, learned=use_c9)

        # ---------------- Pipeline A: degrade it ourselves
        yield dict(type="progress", stage="ladder", pct=20, msg="Pipeline A: building ladder R0-R8 and lifecycle orderings")
        # large files: the lifecycle is run on the analysed region, which is all the cues read
        ry, rx, rh, rw = cues.analysis_region(H, W)
        work = m0 if (rh >= H - 8 and rw >= W - 8) else ladder.Media(
            rgb=np.ascontiguousarray(m0.rgb[ry:ry + rh, rx:rx + rw]), data=m0.data, exif=m0.exif)
        lad = [m0] + [ladder.rung(work, r) for r in range(1, R)]
        life = ladder.lifecycle_states(work, LIFECYCLE_DEPTH)
        paths = list(life.keys())
        jobs = {}
        for r in range(1, R):
            jobs[self.pool.submit(cues.extract, lad[r], device, False, use_c9)] = ("lad", r)
        for r in range(1, R):   # the stability probe of every rung; the input's own probe is the 'save' event
            jobs[self.pool.submit(cues.extract, ladder.apply_ops(lad[r], PROBE["ops"]), device, False, use_c9)] = ("probe", r)
        for i, p in enumerate(paths):
            jobs[self.pool.submit(cues.extract, life[p], device, False, use_c9)] = ("life", i)
        nf = len(cues.DAMAGE_NAMES)
        S_lad, F_lad = np.full((R, K), np.nan, np.float32), np.zeros((R, nf), np.float32)
        S_probe = np.full((R, K), np.nan, np.float32)
        S_life, F_life = np.full((len(paths), K), np.nan, np.float32), np.zeros((len(paths), nf), np.float32)
        S_lad[0], F_lad[0] = out0["scores"], out0["damage"]
        done = 0
        for fut in as_completed(jobs):
            kind, i = jobs[fut]
            o = fut.result()
            if kind == "lad":
                S_lad[i], F_lad[i] = o["scores"], o["damage"]
            elif kind == "probe":
                S_probe[i] = o["scores"]
            else:
                S_life[i], F_life[i] = o["scores"], o["damage"]
            done += 1
            if done % 8 == 0 or done == len(jobs):
                yield dict(type="progress", stage="lifecycle", pct=20 + int(68 * done / len(jobs)),
                           msg=f"Pipeline A: cues read on {done} of {len(jobs)} degraded states")
        S_probe[0] = S_life[paths.index(("save",))]

        # ---------------- fusion
        yield dict(type="progress", stage="fuse", pct=92, msg="Fusion: damage estimate, trust, topology, verdict")
        post = self.hub.posterior(F_lad)                       # (R, R): each ladder state's rung posterior
        dom = np.array([pixel_domain(*lad[r].rgb.shape[:2]) for r in range(R)], dtype=np.float32)
        fz = self.fuser.predict(S_lad, post, S_probe, pixel_domain=dom)
        tau, scale = self.frozen.tau, self.frozen.scale
        margin = lambda S: (S - tau[None, :]) / scale[None, :]  # noqa: E731
        fired_lad = fz["decision"]
        p0 = float(fz["A1"][0])
        trust0, contrib0 = fz["trust"][0], fz["contrib"][0]
        trust_sum = float(trust0.sum())
        env = self.hub.envelope(F_lad[0])

        def label_of(p, tsum):
            if p >= VERDICT["fake"]:
                return "fake"
            if p <= VERDICT["real"] and tsum >= VERDICT["min_trust"]:
                return "real"
            return "inconclusive"

        lab = label_of(p0, trust_sum)
        labels_lad = [label_of(float(fz["A1"][r]), float(fz["trust"][r].sum())) for r in range(R)]
        holds = 0
        for r in range(1, R):
            if labels_lad[r] != lab:
                break
            holds = r

        # ---------------- relationships on this file
        S_all = np.concatenate([S_lad, S_life], axis=0)
        comove = _spearman(margin(S_all))
        rung_hat = int(np.argmax(post[0]))
        ref_rho = np.array(self.ref["dependence"]["rho_calib"][rung_hat], dtype=float)
        adj = fz["adjacency"][0]
        edges = [[i, j, _f(adj[i, j])] for i in range(K) for j in range(i + 1, K) if adj[i, j] > 0.02]

        # order effect: among depth-3 histories, how much of the variation is due to order alone
        d3 = [i for i, p in enumerate(paths) if len(p) == LIFECYCLE_DEPTH]
        M3 = margin(S_life[d3])
        groups = {}
        for row, i in enumerate(d3):
            groups.setdefault(frozenset(paths[i]), []).append(row)
        order_effect = []
        for k in range(K):
            col = M3[:, k]
            if np.isnan(col).any() or col.var() < 1e-9:
                order_effect.append(None)
                continue
            within = np.mean([col[g].var() for g in groups.values()])
            order_effect.append(_f(min(1.0, within / col.var())))

        fired_life = self.frozen.decide(S_life)
        depth_summary = []
        for d in range(0, LIFECYCLE_DEPTH + 1):
            if d == 0:
                rows = margin(S_lad[:1])
                fr = fired_lad[:1]
            else:
                idx = [i for i, p in enumerate(paths) if len(p) == d]
                rows, fr = margin(S_life[idx]), fired_life[idx]
            depth_summary.append(dict(depth=d, n=int(rows.shape[0]),
                                      median=_fl(np.nanmedian(rows, axis=0)) if not np.all(np.isnan(rows)) else [None] * K,
                                      lo=_fl(np.nanmin(np.where(np.isnan(rows), np.inf, rows), axis=0)),
                                      hi=_fl(np.nanmax(np.where(np.isnan(rows), -np.inf, rows), axis=0)),
                                      fired=_fl(fr.mean(axis=0))))

        # ---------------- per-cue report
        y0, x0, h, w = out0["region"]
        m_lad = margin(S_lad)
        cue_out, maps_for_fusion = [], []
        for k, meta in enumerate(CUES):
            cid = meta["id"]
            if np.isnan(S_lad[0, k]):
                cue_out.append(dict(id=cid, status="masked", reason=out0["masked"].get(cid, "masked")))
                continue
            fired = bool(fired_lad[0, k])
            survives = None
            if fired:
                last = 0
                for r in range(1, R):
                    if not fired_lad[r, k]:
                        break
                    last = r
                survives = RUNG_IDS[last]
            mp = out0["maps"].get(cid)
            lo, hi = (0.0, 20.0) if cid == "C5" else (0.0, 1.0) if cid == "C9" else (2.0, 10.0)
            reg = (0, 0, H, W) if cid == "C9" else out0["region"]
            has = mp is not None and mp.size > 1 and float(np.max(mp)) > lo
            payload = _full_map(mp, reg, (H, W), lo, hi, orient) if has else None
            if fired and has and trust0[k] > 0.02:
                maps_for_fusion.append((float(trust0[k]), _canvas(mp, reg, (H, W), lo, hi)))
            cue_out.append(dict(
                id=cid, status="active", score=_f(S_lad[0, k]), tau=_f(tau[k]), margin=_f(m_lad[0, k]),
                fired=fired, persists=bool(fz["persists"][0, k]), evidence=_f(fz["evidence"][0, k]),
                trust=_f(trust0[k]), contribution=_f(contrib0[k]), survives=survives,
                details=out0["details"].get(cid, []), map=payload))

        fused_map, fm = None, None
        if maps_for_fusion:
            wsum = sum(wt for wt, _ in maps_for_fusion)
            fm = sum(wt * v for wt, v in maps_for_fusion) / wsum
            fused_map = _encode(fm, orient)

        # ---------------- verdict
        order = np.argsort(-np.abs(contrib0))
        decisive = [dict(id=CUE_IDS[k], contribution=_f(contrib0[k])) for k in order
                    if abs(contrib0[k]) > 0.15 and not np.isnan(S_lad[0, k]) and trust0[k] > 0.05][:4]
        conf = abs(p0 - 0.5) * 2
        reasons = []
        fired_ids = [c["id"] for c in cue_out if c.get("fired") and (c.get("trust") or 0) > 0.05]
        if lab == "fake":
            title = "Manipulated or synthetic"
            c8 = next((c for c in cue_out if c["id"] == "C8" and c.get("fired")), None)
            reasons.append(f"{len(fired_ids)} trusted cue{'s' if len(fired_ids) != 1 else ''} above the frozen "
                           f"threshold: {', '.join(fired_ids) or 'none'}.")
            if c8 and c8["details"]:
                reasons.append("Container: " + "; ".join(c8["details"]) + ".")
        elif lab == "real":
            title = "No manipulation evidence"
            reasons.append("The cues that can still be trusted at this level of damage found nothing.")
        else:
            title = "Inconclusive, leaning authentic" if p0 <= VERDICT["real"] else "Inconclusive"
            if p0 <= VERDICT["real"]:
                reasons.append("Nothing points at manipulation, but the evidence that can be trusted here is too thin "
                               "to vouch for the file.")
            elif trust_sum < VERDICT["min_trust"] and dom[0] < 0.5:
                reasons.append("Nothing this system can trust on a photo this size decides it: the pixel cues are "
                               "outside their calibration range and the container carries no provenance.")
            elif trust_sum < VERDICT["min_trust"]:
                reasons.append("The file is too degraded for the cues to be trusted, so silence is not evidence.")
            else:
                reasons.append("The evidence is mixed: the fused probability sits between the two decision bounds.")
        reasons.append(f"Damage estimate: looks like {RUNG_IDS[rung_hat]} ({LADDER[rung_hat]['label']}), "
                       f"posterior {post[0, rung_hat]:.0%}.")
        if holds >= R - 1:
            reasons.append("This verdict holds on every rung of the ladder, including the platform emulator.")
        elif holds == 0:
            reasons.append(f"This verdict does not survive {RUNG_IDS[1]}: one high-quality re-save would change it.")
        else:
            reasons.append(f"This verdict holds through {RUNG_IDS[holds]} and changes at {RUNG_IDS[holds + 1]}.")
        if env > 1.5:
            reasons.append("The file's damage profile lies outside the calibrated envelope of the reference set: "
                           "treat the probability as indicative only.")

        # the erasure-specific answer from C9, with its measured reliability for this file
        erasure = None
        if K > 8 and not np.isnan(S_lad[0, 8]):
            d9, b9 = int(fz["decision"][0, 8]), int(fz["persists"][0, 8])
            p_right = float(post[0] @ self.frozen.T[8, :, d9, b9])
            erasure = dict(found=bool(d9), p_right=_f(p_right), stable=bool(b9), trust=_f(fz["trust"][0, 8]),
                           score=_f(S_lad[0, 8]), reliable=bool(fz["trust"][0, 8] >= 0.3))
        if dom[0] < 0.999:
            reasons.append(f"This image is {W * H / 1e6:.1f} megapixels; the pixel cues C1-C6 were calibrated on "
                           f"0.26-megapixel frames, so their trust is scaled to {dom[0]:.0%}. Provenance in the "
                           f"container (C8) is unaffected.")
        if device is None and rec is None:
            reasons.append("No registered camera matches this file, so the sensor cue is masked. Thresholds and trust "
                           "were fitted on the simulated reference set and may not transfer to it.")

        ladder_out = []
        for r in range(R):
            mm = lad[r] if r else work
            ladder_out.append(dict(
                rung=RUNG_IDS[r], label=LADDER[r]["label"], short=LADDER[r]["short"],
                thumb=_jpeg_url(_orient(mm.rgb, orient), 132, 72), size=f"{mm.rgb.shape[1]}x{mm.rgb.shape[0]}",
                bytes=len(lad[r].data), margins=_fl(m_lad[r]), fired=[bool(v) for v in fired_lad[r]],
                p=dict(A1=_f(fz["A1"][r]), A2=_f(fz["A2"][r]), A3=_f(fz["A3"][r])),
                trust=_fl(fz["trust"][r]), trust_sum=_f(fz["trust"][r].sum()), verdict=labels_lad[r] if r else lab,
                looks_like=RUNG_IDS[int(np.argmax(post[r]))]))

        m_life = margin(S_life)
        truth = None
        if rec is not None:
            truth = dict(id=rec["id"], y=rec["y"], family=rec["family"], label=corpus.FAMILY_LABEL[rec["family"]],
                         note=rec["note"], split=rec["split"], rung=RUNG_IDS[sample.get("rung", 0)],
                         correct=(lab == "fake") == bool(rec["y"]) if lab != "inconclusive" else None)
            if rec["has_mask"] and (H, W) == (corpus.SIZE, corpus.SIZE):
                gm = cv2.imread(str(CORPUS_DIR / "masks" / f"{rec['id']}.png"), 0)
                if gm is not None:
                    truth["mask"] = _full_map(gm.astype(np.float32) / 255.0, (0, 0, H, W), (H, W), 0.0, 1.0, orient)
                    if fm is not None:
                        g = _canvas(gm.astype(np.float32) / 255.0, (0, 0, H, W), (H, W), 0.0, 1.0) > 0.5
                        pr = fm >= max(0.25, 0.5 * float(fm.max()))
                        truth["iou"] = _f((g & pr).sum() / max(1, (g | pr).sum()))

        ex = cont["exif"]
        result = dict(
            file=dict(name=name, format=cont["format"], width=W, height=H, bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest()[:16],
                      region=dict(x=x0, y=y0, w=w, h=h, full=(h == H - H % 8 and w == W - W % 8)),
                      declared_q=cont["declared_q"], std_tables=cont["std_tables"], chroma=cont["chroma_sub"],
                      progressive=cont["progressive"], markers=cont["markers"], exif=ex,
                      text_keys=cont["text_keys"], has_xmp=cont["has_xmp"], has_c2pa=cont["has_c2pa"],
                      device=device["id"] if device else None, device_note=dev_note, pce=_f(out0["pce"], 0)),
            preview=_jpeg_url(_orient(m0.rgb, orient), 960, 84),
            region_box=_region_box(out0["region"], (H, W), orient),
            user_label=user_label,
            cues=cue_out, fused_map=fused_map,
            damage=dict(posterior=_fl(post[0]), rung=RUNG_IDS[rung_hat], expected=_f(float(post[0] @ np.arange(R)), 2),
                        envelope=_f(env, 2), features={n: _f(v) for n, v in zip(cues.DAMAGE_NAMES, F_lad[0])}),
            ladder=ladder_out,
            lifecycle=dict(events=[dict(id=e["id"], label=e["label"]) for e in EVENTS], depth=LIFECYCLE_DEPTH,
                           n_states=len(paths), order_effect=order_effect, by_depth=depth_summary,
                           states=[dict(path=list(p), margins=_fl(m_life[i]), fired=[bool(v) for v in fired_life[i]])
                                   for i, p in enumerate(paths)]),
            relationship=dict(comove=[_fl(row) for row in comove], reference=[_fl(row) for row in ref_rho],
                              reference_rung=RUNG_IDS[rung_hat], edges=edges, theta=_f(self.frozen.theta)),
            fusion=dict(A1=_f(p0), A2=_f(fz["A2"][0]), A3=_f(fz["A3"][0]), spread=_f(fz["spread"][0]),
                        trust_sum=_f(trust_sum), bias_note="A1 is the reported arm; A2 and A3 are the ablation arms"),
            erasure=erasure,
            verdict=dict(label=lab, title=title, p=_f(p0), confidence=_f(conf),
                         confidence_word="high" if conf >= 0.7 else "moderate" if conf >= 0.4 else "low",
                         reasons=reasons, decisive=decisive, holds_through=RUNG_IDS[holds],
                         outside_envelope=bool(env > 1.5)),
            truth=truth,
            timing=dict(seconds=round(time.time() - t0, 2), states=1 + len(jobs)),
            config=config_hash(),
        )
        yield dict(type="result", data=result)
