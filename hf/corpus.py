"""01 Data plane: the controlled core set.

The project's core set is 4+ team phones with flat-field frames, consented
manipulations and a true R0. The demo reproduces that protocol in simulation so
that every ground truth is known exactly:

  * scenes   : never-compressed photographs (Kodak suite), cropped to 512x512
  * devices  : four simulated cameras, each with its own sensor fingerprint (PRNU),
               its own JPEG quantisation tables and its own EXIF identity
  * K        : each device's fingerprint is *estimated* from flat-field frames shot
               through that device, exactly as it would be for a real phone
  * fakes    : splicing, copy-move, inpainting, a synthesised region, full synthesis,
               each saved the way an editor or a generator would save it

One record per sample: id, modality, label, family, device_id, split, R0 hash.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from . import codec
from .config import BASE_DIR, CORPUS_DIR, REF_DIR, SEED
from .cues import luma, noise_residual

SIZE = 512
DEVICES = [
    dict(id="D1", make="HyperCam", model="HC-A1", serial="HCA1-000417", q=90),
    dict(id="D2", make="HyperCam", model="HC-B2", serial="HCB2-001903", q=86),
    dict(id="D3", make="Nordlys", model="N7 Pro", serial="N7P-58210", q=93),
    dict(id="D4", make="Kestrel", model="K-5", serial="K5-220764", q=88),
]
FAMILIES = ["splice", "copymove", "inpaint", "synth_region", "full_synth", "erase"]
FAMILY_LABEL = dict(real="Authentic camera original", real_web="Authentic photo, web copy",
                    splice="Spliced region from another photo",
                    copymove="Copy-move within the frame", inpaint="Object removed by inpainting",
                    synth_region="Synthesised region (face-swap style)",
                    full_synth="Fully synthetic image",
                    erase="Object erased by AI (LaMa)")
EDITORS = ["Adobe Photoshop 25.3 (Windows)", "GIMP 2.10.36", "Adobe Photoshop Lightroom 7.1",
           "Affinity Photo 2.4", "Pixelmator Pro 3.5"]


# ------------------------------------------------------------------------ devices
def _device_tables(dev: dict):
    rng = np.random.default_rng([SEED, 7, int(dev["id"][1:])])
    lum, chrom = codec.std_tables(dev["q"])
    jit = lambda t: np.clip(np.round(t * rng.uniform(0.86, 1.18, 64)), 1, 255).astype(int)  # noqa: E731
    return jit(lum), jit(chrom)


def _true_prnu(dev: dict) -> np.ndarray:
    rng = np.random.default_rng([SEED, 3, int(dev["id"][1:])])
    w = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    k = 0.55 * w + 0.45 * cv2.GaussianBlur(w, (0, 0), 0.9) / 0.32
    return (k / k.std() * 0.009).astype(np.float32)


def _expose(scene: np.ndarray, dev: dict, rng) -> np.ndarray:
    """Sensor model: exposure, multiplicative fingerprint (PRNU), additive read noise."""
    x = scene.astype(np.float32) * rng.uniform(0.93, 1.05)
    x = x * (1.0 + dev["K_true"][..., None])
    x += rng.normal(0.0, 0.9, x.shape).astype(np.float32)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def capture(scene: np.ndarray, dev: dict, rng, dt: str) -> bytes:
    """Shoot a scene with a device: sensor model, camera JPEG, camera EXIF."""
    exif = codec.make_exif(make=dev["make"], model=dev["model"], dt_original=dt, dt_modified=dt,
                           serial=dev["serial"])
    return codec.encode_jpeg(_expose(scene, dev, rng), qtables=dev["tables"], sub=2, exif=exif)


def build_devices() -> list[dict]:
    """Create the four devices and estimate each fingerprint K from 14 flat-field frames.

    Flat fields are taken in the device's lossless mode, as they would be with a
    RAW-capable camera app: a JPEG of a flat wall quantises the fingerprint away.
    """
    devs = []
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    for d in DEVICES:
        dev = dict(d)
        dev["tables"] = _device_tables(dev)
        dev["K_true"] = _true_prnu(dev)
        rng = np.random.default_rng([SEED, 5, int(dev["id"][1:])])
        num = np.zeros((SIZE, SIZE), dtype=np.float64)
        den = np.zeros((SIZE, SIZE), dtype=np.float64)
        for _ in range(14):
            level = rng.uniform(105, 205)
            flat = level * (1.0 - 0.10 * ((yy - 0.5) ** 2 + (xx - 0.5) ** 2))
            flat = np.repeat(flat[..., None], 3, axis=2) * rng.uniform(0.96, 1.04, 3)
            Y = luma(_expose(flat, dev, rng))
            num += noise_residual(Y) * Y
            den += Y * Y
        K = (num / den).astype(np.float32)
        K -= K.mean(axis=0, keepdims=True)
        K -= K.mean(axis=1, keepdims=True)
        dev["K"] = K
        devs.append(dev)
    return devs


def save_devices(devs: list[dict]) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF_DIR / "devices.npz", **{f"K_{d['id']}": d["K"] for d in devs},
                        **{f"T_{d['id']}": np.asarray(d["tables"][0]) for d in devs})
    meta = [{k: d[k] for k in ("id", "make", "model", "serial", "q")} for d in devs]
    (REF_DIR / "devices.json").write_text(json.dumps(meta, indent=1))


def load_devices() -> dict[str, dict]:
    meta = json.loads((REF_DIR / "devices.json").read_text())
    z = np.load(REF_DIR / "devices.npz")
    return {m["id"]: dict(m, K=z[f"K_{m['id']}"], luma_table=z[f"T_{m['id']}"]) for m in meta}


# ------------------------------------------------------------------ manipulations
def _blob(rng, lo=0.07, hi=0.17):
    """A soft-edged organic region covering 7-17% of the frame."""
    target = rng.uniform(lo, hi)
    cy, cx = rng.uniform(0.28, 0.72, 2) * SIZE
    a = np.sqrt(target * SIZE * SIZE / np.pi)
    asp = rng.uniform(0.75, 1.35)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    d = ((yy - cy) / (a * asp)) ** 2 + ((xx - cx) / (a / asp)) ** 2
    wob = cv2.GaussianBlur(rng.standard_normal((SIZE, SIZE)).astype(np.float32), (0, 0), 22)
    hard = (d + 0.45 * wob / (wob.std() + 1e-9)) < 1.0
    alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), rng.uniform(1.0, 2.6))
    return alpha[..., None], hard


def _odd_shift(rng, lo=40, hi=190):
    while True:
        ty, tx = (int(v) * int(s) for v, s in zip(rng.integers(lo, hi, 2), rng.choice([-1, 1], 2)))
        if ty % 8 and tx % 8:
            return ty, tx


def _blend(host, patch, alpha):
    return np.clip(np.round(host * (1 - alpha) + patch * alpha), 0, 255).astype(np.uint8)


def m_splice(host, donor, rng):
    alpha, hard = _blob(rng)
    p = donor.astype(np.float32)
    if rng.random() < 0.5:
        s = rng.choice([rng.uniform(0.74, 0.9), rng.uniform(1.12, 1.4)])
        p = cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        p = np.pad(p, ((0, max(0, SIZE - p.shape[0])), (0, max(0, SIZE - p.shape[1])), (0, 0)),
                   mode="reflect")[:SIZE, :SIZE]
    if rng.random() < 0.5:
        p = p[:, ::-1]
    p = np.roll(p, _odd_shift(rng), axis=(0, 1))
    if rng.random() < 0.6:
        a = rng.choice([rng.uniform(0.80, 0.9), rng.uniform(1.1, 1.22)])
        p = p * np.array([a, 1.0, 1.0 / a], dtype=np.float32)
    return _blend(host.astype(np.float32), p, alpha), hard


def m_copymove(host, rng):
    alpha, hard = _blob(rng, 0.06, 0.13)
    p = np.roll(host.astype(np.float32), _odd_shift(rng, 60, 200), axis=(0, 1))
    return _blend(host.astype(np.float32), p, alpha), hard


def m_inpaint(host, rng):
    alpha, hard = _blob(rng, 0.06, 0.12)
    mask = cv2.dilate(hard.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    small = cv2.resize(host, (SIZE // 2, SIZE // 2), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (SIZE // 2, SIZE // 2), interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(small, msmall, 5, cv2.INPAINT_TELEA)
    fill = cv2.resize(fill, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    if rng.random() < 0.6:                   # a careful forger adds grain back
        fill += rng.normal(0, rng.uniform(1.0, 2.2), fill.shape).astype(np.float32)
    return _blend(host.astype(np.float32), fill, alpha), hard


def toy_generator(img, rng):
    """A stand-in for a generative decoder: synthesise at low resolution, then upsample.

    It is not a GAN or a diffusion model. It reproduces the one property C5 reads:
    content that was produced on a coarser grid and interpolated up to full size,
    with the generator's own texture laid down before the upsampling. The upsampling
    factors are non-integer, as in a warped face-swap or an upscaled render: integer
    x2 stages put their periodicity on the JPEG lattice, a known blind spot of C5.
    """
    f = float(rng.choice([1.2, 1.45, 1.7, 1.9, 2.3]))
    h, w = img.shape[:2]
    low = cv2.resize(img, (int(round(w / f)), int(round(h / f))), interpolation=cv2.INTER_AREA)
    low = low.astype(np.float32)
    low = low + 0.6 * (low - cv2.GaussianBlur(low, (0, 0), 2.0))
    low += rng.normal(0, 2.2, low.shape).astype(np.float32)
    interp = cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC
    return np.clip(cv2.resize(low, (w, h), interpolation=interp), 0, 255)


_LAMA = None


def m_erase(host, rng):
    """AI object removal: a region is filled by LaMa, as phone erase tools do."""
    global _LAMA
    if _LAMA is None:
        from .erase.lama import Lama
        _LAMA = Lama(device=__import__("os").environ.get("HF_LAMA_DEVICE"))
    alpha, hard = _blob(rng, 0.06, 0.16)
    s = 1.0 if rng.random() < 0.5 else float(rng.uniform(0.5, 0.8))
    if s < 1.0:
        n = int(SIZE * s)
        sm = cv2.resize(host, (n, n), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(hard.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
        fill = cv2.resize(_LAMA(sm, mm), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    else:
        fill = _LAMA(host, hard)
    return _blend(host.astype(np.float32), fill.astype(np.float32), alpha), hard


def m_synth_region(host, rng):
    alpha, hard = _blob(rng, 0.08, 0.17)
    g = toy_generator(host, rng)
    ys, xs = np.nonzero(hard)
    M = cv2.getRotationMatrix2D((float(xs.mean()), float(ys.mean())), rng.uniform(-2.5, 2.5),
                                rng.uniform(1.0, 1.05))
    g = cv2.warpAffine(g, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _blend(host.astype(np.float32), g, alpha), hard


# ------------------------------------------------------------------- corpus build
def _scenes() -> list[tuple[int, int, np.ndarray]]:
    out = []
    for p in sorted(BASE_DIR.glob("kodim*.png")):
        idx = int(p.stem[-2:])
        img = codec.decode_rgb(p.read_bytes())
        h, w = img.shape[:2]
        if h < SIZE or w < SIZE:
            continue
        for c, off in enumerate((0, max(h, w) - SIZE)):
            crop = img[:SIZE, off:off + SIZE] if w >= h else img[off:off + SIZE, :SIZE]
            out.append((idx, c, np.ascontiguousarray(crop)))
    return out


def _editor_save(rgb, host_exif, dt, rng, devs_by_id=None):
    """Save a manipulated image the way an editor would. Returns (bytes, ext, note)."""
    u = rng.random()
    later = dt[:11] + f"{(int(dt[11:13]) + int(rng.integers(1, 9))) % 24:02d}" + dt[13:]
    if u < 0.55:      # ordinary export: the editor signs its work
        ex = codec.make_exif(make=host_exif["make"], model=host_exif["model"],
                             software=str(rng.choice(EDITORS)), dt_original=dt, dt_modified=later,
                             serial=host_exif["serial"])
        return codec.encode_jpeg(rgb, q=int(rng.integers(90, 98)), sub=0, exif=ex), "jpg", "editor export"
    if u < 0.80:      # lossless export, metadata gone
        return codec.encode_png(rgb), "png", "lossless export, no metadata"
    ex = codec.make_exif(make=host_exif["make"], model=host_exif["model"], dt_original=dt,
                         dt_modified=dt, serial=host_exif["serial"])
    return codec.encode_jpeg(rgb, q=int(rng.integers(90, 96)), sub=2, exif=ex), "jpg", "laundered metadata"


def build_corpus(log=print) -> list[dict]:
    """Generate the controlled core set and write corpus.json. Deterministic in SEED."""
    scenes = _scenes()
    if len(scenes) < 12:
        raise RuntimeError(f"need source scenes in {BASE_DIR}; run: python -m hf.fetch_base")
    devs = build_devices()
    save_devices(devs)
    by_id = {d["id"]: d for d in devs}
    (CORPUS_DIR / "masks").mkdir(parents=True, exist_ok=True)

    kod = sorted({s[0] for s in scenes})
    order = np.random.default_rng([SEED, 1]).permutation(kod)
    n3 = len(kod) // 3
    split_of = {int(k): ("train" if i < n3 else "calib" if i < 2 * n3 else "test")
                for i, k in enumerate(order)}
    records = []

    def emit(sid, blob, ext, y, family, device_id, split, scene_key, note, mask=None):
        (CORPUS_DIR / f"{sid}.{ext}").write_bytes(blob)
        if mask is not None:
            cv2.imwrite(str(CORPUS_DIR / "masks" / f"{sid}.png"), mask.astype(np.uint8) * 255)
        records.append(dict(id=sid, file=f"{sid}.{ext}", modality="img", y=int(y), family=family,
                            device_id=device_id, split=split, scene=scene_key, note=note,
                            has_mask=mask is not None, r0_hash=hashlib.sha256(blob).hexdigest()[:16]))

    for si, (kidx, crop, scene) in enumerate(scenes):
        split = split_of[kidx]
        key = f"k{kidx:02d}{'ab'[crop]}"
        rng = np.random.default_rng([SEED, 11, kidx, crop])
        dt = f"2026:08:{int(rng.integers(2, 28)):02d} {int(rng.integers(7, 15)):02d}:" \
             f"{int(rng.integers(0, 60)):02d}:{int(rng.integers(0, 60)):02d}"
        dev_order = list(rng.permutation(len(devs)))

        # --- authentic: three devices shoot the scene
        for j in range(3):
            d = devs[dev_order[j]]
            blob = capture(scene, d, rng, dt)
            if rng.random() < 0.12:           # shared as a clean lossless copy: no metadata left
                blob, ext, note = codec.encode_png(codec.decode_rgb(blob)), "png", "camera frame, metadata stripped"
            else:
                ext, note = "jpg", "camera original"
            emit(f"{split}_{key}_real{j + 1}", blob, ext, 0, "real", d["id"], split, key, note)

        # --- authentic, off protocol: the same scene as it is found on the web. Resized,
        # re-encoded once, no metadata, no device record. Without these the reference
        # would teach that a missing fingerprint or a resize is evidence of forgery.
        d = devs[dev_order[3 % len(devs)]]
        web = codec.decode_rgb(capture(scene, d, rng, dt))
        if rng.random() < 0.7:
            n = int(round(SIZE * rng.uniform(0.62, 0.9)))
            web = cv2.resize(web, (n, n), interpolation=cv2.INTER_AREA)
        emit(f"{split}_{key}_web", codec.encode_jpeg(web, q=int(rng.integers(80, 96)), sub=2), "jpg", 0,
             "real_web", None, split, key, "web copy: resized, re-encoded, no metadata")

        # --- manipulated: each starts from its own fresh capture
        same_split = [s for s in scenes if split_of[s[0]] == split and s[0] != kidx]
        for fam in FAMILIES:
            d = devs[int(rng.integers(0, len(devs)))]
            hexif = dict(make=d["make"], model=d["model"], serial=d["serial"])
            if fam == "full_synth":
                g = toy_generator(scene.astype(np.float32) * rng.uniform(0.95, 1.05), rng)
                rgb = np.clip(np.round(g), 0, 255).astype(np.uint8)
                u = rng.random()
                if u < 0.6:
                    text = {"parameters": f"photograph, natural light, 35mm\nSteps: 30, Sampler: DPM++ 2M, "
                                          f"CFG scale: 6.5, Seed: {int(rng.integers(1e9))}, "
                                          f"Model: toy-generator-v0"}
                    blob, ext, note, dev_id = codec.encode_png(rgb, text=text), "png", "generator output with parameters", None
                elif u < 0.8:
                    blob, ext, note, dev_id = codec.encode_jpeg(rgb, q=95, sub=0), "jpg", "generator output, no metadata", None
                else:                          # passed off as a camera frame: forged EXIF, no fingerprint
                    ex = codec.make_exif(make=d["make"], model=d["model"], dt_original=dt, dt_modified=dt,
                                         serial=d["serial"])
                    blob, ext, note, dev_id = codec.encode_jpeg(rgb, q=92, sub=2, exif=ex), "jpg", "forged camera EXIF", d["id"]
                emit(f"{split}_{key}_{fam}", blob, ext, 1, fam, dev_id, split, key, note)
                continue
            host = codec.decode_rgb(capture(scene, d, rng, dt))
            if fam == "splice":
                dk, dc, dscene = same_split[int(rng.integers(0, len(same_split)))]
                others = [x for x in devs if x["id"] != d["id"]]
                donor = codec.decode_rgb(capture(dscene, others[int(rng.integers(0, len(others)))], rng, dt))
                rgb, mask = m_splice(host, donor, rng)
            elif fam == "copymove":
                rgb, mask = m_copymove(host, rng)
            elif fam == "inpaint":
                rgb, mask = m_inpaint(host, rng)
            elif fam == "erase":
                rgb, mask = m_erase(host, rng)
            else:
                rgb, mask = m_synth_region(host, rng)
            blob, ext, note = _editor_save(rgb, hexif, dt, rng)
            emit(f"{split}_{key}_{fam}", blob, ext, 1, fam, d["id"], split, key, note, mask=mask)
        if (si + 1) % 12 == 0:
            log(f"  corpus: {si + 1}/{len(scenes)} scenes")
    (CORPUS_DIR / "corpus.json").write_text(json.dumps(records, indent=1))
    return records


def load_corpus() -> list[dict]:
    return json.loads((CORPUS_DIR / "corpus.json").read_text())
