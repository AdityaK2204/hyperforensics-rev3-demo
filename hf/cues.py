"""03 Cue battery: eight hand-built cues in six origin classes, a learned erasure cue (C9),
plus the damage features.

Every cue returns one scalar score, higher meaning "more evidence of manipulation
or synthesis". The localising cues (C1-C4, C6) build a block map, turn it into
content-conditional normal scores, and measure how far the most anomalous
contiguous region sits from the rest of the frame. C5 tests for periodic peaks in
a prediction-residual spectrum. C8 reads the container. C7 is video only and is
always masked here.

    extract(media) -> scores[K], maps, details, damage features
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn
from scipy.special import gammaincc, ndtri
from scipy.stats import rankdata

from . import codec
from .config import CUE_IDS, CUE_PARAMS as P, K

ZIGZAG = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0),
          (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]
DAMAGE_NAMES = ["declared_q", "lattice_coarseness", "prior_lattice", "grid_sparsity",
                "offgrid_sparsity", "hf_energy", "resampling_peak", "has_metadata",
                "log_bytes_per_px", "chroma_subsampled", "is_jpeg"]
TWO_PI = 2.0 * np.pi
NULL_STAT = 2.5       # expected value of the region statistic when a map carries no structure


# ------------------------------------------------------------------ small helpers
def luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb.astype(np.float32)
    return 0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]


def analysis_region(h: int, w: int) -> tuple[int, int, int, int]:
    """Central crop on the file's own 8x8 grid: (y0, x0, height, width)."""
    cap = P["analysis_cap"]
    h8, w8 = h - h % 8, w - w % 8
    ch, cw = min(h8, cap), min(w8, cap)
    y0 = ((h8 - ch) // 2) // 8 * 8
    x0 = ((w8 - cw) // 2) // 8 * 8
    return y0, x0, ch, cw


def blockify(a: np.ndarray, b: int = 8) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h // b, b, w // b, b).swapaxes(1, 2)


def block_mean(a: np.ndarray, b: int = 8) -> np.ndarray:
    h, w = a.shape[0] - a.shape[0] % b, a.shape[1] - a.shape[1] % b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def normal_scores(m, valid, covs=(), bins=(), min_n=24):
    """Content-conditional normal scores.

    Blocks are grouped by quantile bins of the content covariates (texture,
    brightness...). Inside each group a block's value is replaced by its rank,
    mapped through the normal quantile function. The result is N(0,1) by
    construction whatever the image content, so no cue can score highly just
    because a frame contains both sky and foliage.
    """
    z = np.zeros(m.shape, dtype=np.float32)
    ok = valid.copy()
    bins = list(bins)
    nv = int(valid.sum())
    while bins and int(np.prod(bins)) * min_n > nv and max(bins) > 1:
        bins[int(np.argmax(bins))] -= 1
    idx = np.zeros(m.shape, dtype=np.int32)
    mult = 1
    if nv >= min_n:
        for c, nb in zip(covs, bins):
            if nb <= 1:
                continue
            qs = np.quantile(c[valid], np.linspace(0, 1, nb + 1)[1:-1])
            idx += np.searchsorted(qs, c).astype(np.int32) * mult
            mult *= nb
    rank_mode = P["z_mode"] == "rank"
    clip = float(P["z_clip"])
    for b in range(mult):
        sel = valid & (idx == b)
        n = int(sel.sum())
        if n < min_n:
            ok &= ~sel
            continue
        v = m[sel]
        if rank_mode:
            z[sel] = ndtri((rankdata(v) - 0.5) / n)
        else:
            med = np.median(v)
            q10, q90 = np.quantile(v, [0.1, 0.9])
            scale = max(1.4826 * float(np.median(np.abs(v - med))), 0.25 * float(q90 - q10) / 2.563, 1e-6)
            z[sel] = np.clip((v - med) / scale, -clip, clip)
    return z, ok


def region_stat(z, valid, side="both", scales=(5, 9)):
    """How far the most anomalous contiguous region sits from the rest of the frame.

    Normal scores are summed over small windows (isolated blocks cancel, regions
    add up), the window field is re-standardised against its own median and MAD (so
    a frame whose content is naturally patchy is held to its own standard), and the
    statistic is the mean of the most extreme 2% of windows. About 2.5 on a map
    with no structure.
    """
    empty = np.zeros(z.shape, dtype=np.float32)
    if int(valid.sum()) < 40:
        return 0.0, empty
    best, best_map = 0.0, empty
    v = valid.astype(np.float32)
    for k in scales:
        if min(z.shape) < k + 2:
            continue
        s = cv2.boxFilter(z * v, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        n = cv2.boxFilter(v, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        good = valid & (n >= 0.5 * k * k)
        if int(good.sum()) < 30:
            continue
        zs = s / np.sqrt(np.maximum(n, 1.0))
        q25, med, q75 = (float(v) for v in np.quantile(zs[good], [0.25, 0.5, 0.75]))
        # background spread from the narrower half of the distribution: a tampered
        # region pushes one side out, and must not be allowed to set its own yardstick
        spread = min(med - q25, q75 - med) / 0.6745 if P["self_norm"] else 1.0
        t = (zs - med) / max(spread, 0.5)
        t = np.abs(t) if side == "both" else (-t if side == "low" else t)
        vals = t[good]
        nn = max(4, int(P["tail_frac"] * vals.size))
        top = float(np.partition(vals, -nn)[-nn:].mean())
        if top > best:
            best, best_map = top, np.where(good, np.maximum(t, 0), 0).astype(np.float32)
    return best, best_map


# ------------------------------------------------------------------ shared context
class Frame:
    """Pixels of the analysed region plus the block statistics several cues share."""

    def __init__(self, rgb_full: np.ndarray):
        self.full_h, self.full_w = rgb_full.shape[:2]
        self.y0, self.x0, self.h, self.w = analysis_region(self.full_h, self.full_w)
        self.rgb = np.ascontiguousarray(rgb_full[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w])
        self.Y = luma(self.rgb)
        blocks = blockify(self.Y)
        self.inten = blocks.mean(axis=(2, 3))
        self.act = np.log(blocks.std(axis=(2, 3)) + 0.5)
        self.valid = (self.inten > 6) & (self.inten < 249)
        self.C = dctn(blocks - 128.0, axes=(2, 3), norm="ortho").astype(np.float32)
        # the same statistics on a grid shifted by (4,4): the off-grid reference
        self.C_off = dctn(blockify(self.Y[4:-4, 4:-4]) - 128.0, axes=(2, 3), norm="ortho").astype(np.float32)
        self.grid_on = self.grid_off = 0.0
        self.lattice_coarseness = self.prior_lattice = self.resampling_peak = 0.0
        self.pce = None


def _sparsity(C: np.ndarray) -> np.ndarray:
    """Soft count of AC coefficients that are (near) zero in each 8x8 block."""
    s = np.exp(-(C * C) / (2 * 0.5 ** 2))
    return s.sum(axis=(2, 3)) - s[..., 0, 0]


# ----------------------------------------------------------------------------- C1
def c1_error_level(f: Frame):
    """Error level analysis: re-save at known qualities and look at the residual.

    The residual at one quality depends mostly on texture; the *shape* of the
    residual across qualities depends on compression history. A region with a
    different history from its surroundings shows up in that shape.
    """
    x = f.rgb.astype(np.float32)
    qs = P["ela_qualities"]
    E, live = {}, []
    for q in qs:
        rec = codec.decode_rgb(codec.encode_jpeg(f.rgb, q=q, sub=0)).astype(np.float32)
        err = block_mean(np.abs(x - rec).mean(axis=2))
        E[q] = np.log(err + 0.1)
        if float(np.median(err)) > 0.35:      # a re-save at the file's own table changes nothing:
            live.append(q)                    # that quality says nothing about history
    if len(live) < 2:
        return NULL_STAT, np.zeros(f.valid.shape, dtype=np.float32), ["re-save residual is flat at every quality"]
    ref = live[0]
    feats = [(f"residual at Q{q} relative to Q{ref}", E[q] - E[ref]) for q in live[1:]]
    feats.append((f"residual at Q{live[-2]}", E[live[-2]]))
    best = (-1.0, None, "")
    for name, m in feats:
        z, ok = normal_scores(m, f.valid, (f.act, f.inten), (8, 3))
        s, zmap = region_stat(z, ok, "both")
        if s > best[0]:
            best = (s, zmap, name)
    return best[0], best[1], [f"strongest regional inconsistency: {best[2]}"]


# ----------------------------------------------------------------------------- C2
def c2_block_dct(f: Frame):
    """Block-DCT statistics: how much sparser the DCT is on the file's 8x8 grid than off it.

    JPEG zeroes most high-frequency coefficients on its own grid and nowhere else, so
    a region whose grid is missing or shifted stands out from the rest of the frame.
    """
    on = _sparsity(f.C)
    off = _sparsity(f.C_off)                              # (H8-1, W8-1), centred on block corners
    off = np.pad(off, ((1, 1), (1, 1)), mode="edge")
    off = 0.25 * (off[:-1, :-1] + off[1:, :-1] + off[:-1, 1:] + off[1:, 1:])
    # two content models. Off-grid sparsity says how sparse a block would be with no
    # grid at all, so what is left over is the grid's own contribution; texture and
    # brightness cover the smooth regions where both sparsities saturate.
    z, ok = normal_scores(on, f.valid, (off, f.inten), (12, 2))
    s, zmap = region_stat(z, ok, "both")
    z2, ok2 = normal_scores(on - off, f.valid, (f.act, f.inten), (8, 3))
    s2, zmap2 = region_stat(z2, ok2, "both")
    if s2 > s:
        s, zmap = s2, zmap2
    if f.valid.any():
        f.grid_on, f.grid_off = float(on[f.valid].mean()) / 63.0, float(off[f.valid].mean()) / 63.0
    return s, zmap, [f"DCT sparsity on the 8x8 grid {f.grid_on:.2f}, off the grid {f.grid_off:.2f}"]


# ----------------------------------------------------------------------------- C3
def c3_quant_history(f: Frame, declared_table):
    """Quantisation history: find a prior quantisation lattice and see who obeys it.

    For each low-frequency DCT mode, the coarsest step q whose lattice the
    coefficients still sit on is recovered from the data. Modes whose lattice is not
    explained by the file's own (last) table are evidence of an earlier compression;
    blocks that ignore that lattice have a different history from the rest.
    """
    M, qmax, fmin = P["c3_modes"], P["c3_qmax"], P["c3_fmin"]
    shape = f.valid.shape
    nb = shape[0] * shape[1]
    qs = np.arange(2, qmax + 1, dtype=np.float32)
    lat_q, lat_f = np.ones(M, dtype=np.float32), np.zeros(M, dtype=np.float32)
    coefs = []
    for m, (u, v) in enumerate(ZIGZAG[:M]):
        c = f.C[:, :, u, v].ravel()
        coefs.append(c)
        ratio = c[None, :] / qs[:, None]
        sel = np.abs(ratio) > 0.5
        n = sel.sum(axis=1)
        fq = (np.cos(TWO_PI * ratio) * sel).sum(axis=1) / np.maximum(n, 1)
        ok = (fq >= fmin) & (n >= max(40, 0.01 * nb))
        if ok.any():
            j = int(np.nonzero(ok)[0].max())
            lat_q[m], lat_f[m] = qs[j], fq[j]
    info = []
    for m, (u, v) in enumerate(ZIGZAG[:M]):
        if lat_q[m] < 2:
            continue
        dq = None if declared_table is None else int(declared_table[u * 8 + v])
        if dq is None or dq % int(lat_q[m]) != 0:
            info.append(m)
    f.lattice_coarseness = float(np.log2(lat_q[:10]).mean())
    f.prior_lattice = float(lat_f[info].sum() / M) if info else 0.0
    empty = np.zeros(shape, dtype=np.float32)
    if not info:
        note = "no quantisation lattice beyond the file's own table" if declared_table is not None \
            else "no quantisation lattice found in the pixels"
        return NULL_STAT, empty, [note]
    num = np.zeros(nb, dtype=np.float32)
    den = np.zeros(nb, dtype=np.float32)
    cnt = np.zeros(nb, dtype=np.float32)
    for m in info:
        q, c = lat_q[m], coefs[m]
        sel = np.abs(c) > 0.5 * q
        num += lat_f[m] * np.cos(TWO_PI * c / q) * sel
        den += lat_f[m] * sel
        cnt += sel
    kappa = (num / np.maximum(den, 1e-6)).reshape(shape)
    cnt = cnt.reshape(shape)
    valid = f.valid & (cnt >= 2)
    steps = ", ".join(str(int(lat_q[m])) for m in info[:6])
    if valid.mean() < 0.06:
        return NULL_STAT, empty, ["prior lattice found, but too few textured blocks to map it"]
    z, ok = normal_scores(kappa, valid, (cnt,), (4,))
    s, zmap = region_stat(z, ok, "low")
    return s, zmap, [f"prior quantisation lattice on {len(info)} of {M} modes (steps {steps})"]


# ----------------------------------------------------------------------------- C4
def noise_residual(Y: np.ndarray) -> np.ndarray:
    """W = x - F(x): high-pass residual with locally adaptive (Wiener-style) suppression of edges."""
    hp = Y - cv2.GaussianBlur(Y, (0, 0), 1.1)
    v = cv2.blur(hp * hp, (5, 5))
    s0 = P["prnu_sigma0"]
    W = hp * np.minimum(1.0, (s0 * s0) / (v + 1e-3))
    W -= W.mean(axis=0, keepdims=True)
    W -= W.mean(axis=1, keepdims=True)
    # remove everything that repeats every 8 pixels: JPEG block structure is shared by
    # every camera and every re-save, and would otherwise pass for a fingerprint match
    h8, w8 = W.shape[0] // 8 * 8, W.shape[1] // 8 * 8
    if h8 >= 16 and w8 >= 16:
        pat = W[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).mean(axis=(0, 2))
        W -= np.tile(pat, (W.shape[0] // 8 + 1, W.shape[1] // 8 + 1))[:W.shape[0], :W.shape[1]]
    return W


def _fit(a: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.float32)
    hh, ww = min(h, a.shape[0]), min(w, a.shape[1])
    out[:hh, :ww] = a[:hh, :ww]
    return out


def _pce(W: np.ndarray, X: np.ndarray, r: int = 12):
    """Peak-to-correlation energy with a small shift search. Returns (pce, dy, dx)."""
    xc = np.fft.irfft2(np.fft.rfft2(W) * np.conj(np.fft.rfft2(X)), s=W.shape)
    win = np.concatenate([np.concatenate([xc[-r:, -r:], xc[-r:, :r + 1]], axis=1),
                          np.concatenate([xc[:r + 1, -r:], xc[:r + 1, :r + 1]], axis=1)], axis=0)
    py, px = np.unravel_index(int(np.argmax(win)), win.shape)
    dy, dx = int(py - r), int(px - r)
    peak = float(win[py, px])
    mask = np.ones(xc.shape, dtype=bool)
    cols = [(dx + xx) % xc.shape[1] for xx in range(-5, 6)]
    for yy in range(-5, 6):
        mask[(dy + yy) % xc.shape[0], cols] = False
    energy = float(np.mean(xc[mask] ** 2))
    return (peak * peak) / max(energy, 1e-12) * (1.0 if peak > 0 else -1.0), dy, dx


def prnu_align(f: Frame, device: dict, W: np.ndarray | None = None):
    """Best alignment of a frame against a device fingerprint: (pce, dy, dx, K_aligned, W)."""
    Kd = device["K"]
    H, Wd = f.full_h, f.full_w
    hyps = [Kd]
    if (H, Wd) != Kd.shape:
        sx, sy = Wd / Kd.shape[1], H / Kd.shape[0]
        if abs(sx - sy) < 0.03 * sx and 0.3 < sx < 1.6:
            hyps.append(cv2.resize(Kd, (Wd, H), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR))
    if W is None:
        W = noise_residual(f.Y)
    best = None
    for Kh in hyps:
        Kr = _fit(Kh[f.y0:, f.x0:], f.h, f.w)
        pce, dy, dx = _pce(W, Kr * f.Y)
        if best is None or pce > best[0]:
            best = (pce, dy, dx, Kr)
    return best + (W,)


def c4_prnu(f: Frame, device: dict):
    """Sensor noise: does the frame carry the claimed device's fingerprint, everywhere?

    PCE(W, K) answers whether the fingerprint is there at all; the block-wise
    correlation map answers whether some region is missing it. Two geometric
    hypotheses are tried when the frame is not sensor-sized: cropped, or rescaled.
    """
    pce, dy, dx, Kr, W = prnu_align(f, device)
    X = (np.roll(Kr, (dy, dx), axis=(0, 1)) if (dy, dx) != (0, 0) else Kr) * f.Y
    b = P["prnu_block"]
    rho = block_mean(W * X, b) / (np.sqrt(block_mean(W * W, b) * block_mean(X * X, b)) + 1e-9)
    inten = block_mean(f.Y, b)
    act = np.log(np.sqrt(np.maximum(block_mean(f.Y * f.Y, b) - inten ** 2, 0)) + 0.5)
    valid = (inten > 20) & (inten < 245)
    z, ok = normal_scores(rho, valid, (inten, act), (4, 4))
    local, zmap = region_stat(z, ok, "low", scales=(3, 7))
    match = pce >= P["pce_match"]
    absence = 8.0 * (1.0 - float(np.clip(np.log1p(max(pce, 0.0)) / np.log1p(P["pce_match"]), 0, 1)))
    f.pce = pce
    note = [f"PCE against the {device['id']} fingerprint: {pce:,.0f} "
            f"({'match' if match else 'no match'}; match threshold {P['pce_match']:.0f})"]
    if (dy, dx) != (0, 0) and match:
        note.append(f"fingerprint re-aligned by ({dy}, {dx}) px")
    if absence > local:
        note.append("fingerprint absent or too weak across the whole frame")
        zmap = np.full(zmap.shape, absence / 2.0, dtype=np.float32)
    return max(local, absence), zmap, note


# ----------------------------------------------------------------------------- C5
_LP_KERNEL = np.array([[-0.25, 0.5, -0.25], [0.5, 0.0, 0.5], [-0.25, 0.5, -0.25]], dtype=np.float32)
_C5_CACHE: dict = {}


def _c5_masks(n: int):
    if n not in _C5_CACHE:
        fy = np.fft.fftfreq(n)[:, None] * n
        fx = np.fft.rfftfreq(n)[None, :] * n
        rad = np.sqrt(fy ** 2 + fx ** 2)
        step = n // 8
        on_lat = (np.abs(((fy + step / 2) % step) - step / 2) <= 1) & \
                 (np.abs(((fx + step / 2) % step) - step / 2) <= 1)
        allow = (rad >= n / 14) & ~on_lat
        hann = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
        _C5_CACHE[n] = (allow, hann, int(allow.sum()))
    return _C5_CACHE[n]


def _peak_significance(S: np.ndarray, m: float, allow: np.ndarray, nbins: int, ntests: int) -> float:
    """-log10 of the chance that pure noise shows a spectral peak this prominent.

    S is an average of m periodograms. Under noise each bin, divided by its local
    background, is Gamma(m, 1/m); the family-wise p-value over all bins and all
    tests is the score, so local and global tests share one scale.
    """
    bg = cv2.blur(S, (9, 9), borderType=cv2.BORDER_REFLECT) + 1e-12
    x = float((S / bg)[allow].max())
    p = float(gammaincc(m, m * x)) * nbins * ntests
    return float(np.clip(-np.log10(max(p, 1e-40)), 0.0, 40.0))


def c5_resampling(f: Frame):
    """Generator resampling: periodic peaks in the spectrum of a linear-prediction residual.

    Interpolated pixels are linear combinations of their neighbours, so the residual
    of a fixed linear predictor becomes periodic wherever content was upsampled or
    warped. The test runs on every 2x2 group of windows (a synthesised region) and
    on the whole frame (a synthesised image). Peaks on the 8-pixel JPEG lattice are
    ignored: those belong to C1-C3.
    """
    n, st = P["c5_window"], P["c5_stride"]
    if f.h < 2 * n or f.w < 2 * n:
        n, st = n // 2, st // 2
        if f.h < 2 * n or f.w < 2 * n:
            return 0.0, np.zeros((1, 1), dtype=np.float32), ["frame too small for spectral analysis"]
    e = np.abs(f.Y - cv2.filter2D(f.Y, -1, _LP_KERNEL, borderType=cv2.BORDER_REFLECT))
    allow, hann, nbins = _c5_masks(n)
    ys = list(range(0, f.h - n + 1, st))
    xs = list(range(0, f.w - n + 1, st))
    wins = np.stack([e[y:y + n, x:x + n] for y in ys for x in xs])
    wins = (wins - wins.mean(axis=(1, 2), keepdims=True)) * hann[None]
    S = (np.abs(np.fft.rfft2(wins)) ** 2).reshape(len(ys), len(xs), n, n // 2 + 1)
    S = S / (S.mean(axis=(2, 3), keepdims=True) + 1e-12)
    gy, gx = len(ys) - 1, len(xs) - 1
    grid = np.zeros((max(gy, 1), max(gx, 1)), dtype=np.float32)
    ntests = max(1, gy * gx)
    for i in range(gy):
        for j in range(gx):
            grid[i, j] = _peak_significance(S[i:i + 2, j:j + 2].mean(axis=(0, 1)), 4.0, allow, nbins, ntests)
    m_all = float(min(16, len(ys) * len(xs)))
    glob = _peak_significance(S.mean(axis=(0, 1)), m_all, allow, nbins, 1)
    f.resampling_peak = glob
    local = float(grid.max())
    where = "across the whole frame" if glob >= local else "in one region"
    return max(local, glob), grid, [f"off-lattice periodicity {where}: significance 10^-{max(local, glob):.1f}"]


# ----------------------------------------------------------------------------- C6
def c6_illumination(f: Frame):
    """Illumination consistency: the colour of the light, read off near-neutral surfaces.

    Grey, white and lightly tinted surfaces take on the colour of whatever lights
    them. Their chromaticity is mapped across the frame; a region lit by a different
    source disagrees with the rest. Compression barely touches it, and it is weak
    and scene-dependent, as scene-physics cues are expected to be.
    """
    b = P["c6_block"]
    lin = np.power(f.rgb.astype(np.float32) / 255.0, 2.2) + 1e-3
    lr = np.log(lin[..., 0] / lin[..., 1])
    lb = np.log(lin[..., 2] / lin[..., 1])
    inten = lin.mean(axis=2)
    cand = ((np.abs(lr) < 0.25) & (np.abs(lb) < 0.25) & (inten > 0.02) & (inten < 0.85)).astype(np.float32)
    n = block_mean(cand, b)
    if min(n.shape) < 5:
        return 0.0, np.zeros((1, 1), dtype=np.float32), ["frame too small for an illuminant map"]
    mr = block_mean(lr * cand, b) / np.maximum(n, 1e-6)
    mb = block_mean(lb * cand, b) / np.maximum(n, 1e-6)
    valid = n > 0.2
    if valid.mean() < 0.15:
        return 0.0, np.zeros(n.shape, dtype=np.float32), ["too few near-neutral surfaces to read the light from"]
    zr, ok = normal_scores(mr, valid)
    zb, _ = normal_scores(mb, valid)
    sr, mapr = region_stat(zr, ok, "both", scales=(3, 5))
    sb, mapb = region_stat(zb, ok, "both", scales=(3, 5))
    s, zmap = (sr, mapr) if sr >= sb else (sb, mapb)
    return s, zmap, [f"illuminant colour read from {int(valid.sum())} of {valid.size} surface patches"]


# ----------------------------------------------------------------------------- C9
_c9_lock = __import__("threading").Lock()


def c9_learned(rgb: np.ndarray):
    """Learned erasure detector on the whole frame at its 1024 px working scale."""
    from .erase import net as erasenet
    with _c9_lock:
        r = erasenet.run(rgb)
    return r["logit"], r["map"], [f"erasure network: P(region filled in) = {r['score']:.2f}, "
                                  f"strongest area {float(r['map'].max()):.2f}"]


def c9_available() -> bool:
    from .erase import net as erasenet
    return erasenet.available()


# ------------------------------------------------------------------------- extract
def extract(media, device: dict | None = None, want_maps: bool = False, learned: bool = False) -> dict:
    """Run the battery on one state of a file.

    media   : ladder.Media (pixels + container bytes of that state)
    device  : registry entry {'id', 'K'} for the claimed camera, or None (C4 masked)
    """
    f = Frame(media.rgb)
    cont = codec.read_container(media.data)
    scores = np.full(K, np.nan, dtype=np.float32)
    maps: dict = {}
    details: dict = {}
    masked: dict = {"C7": "video only: masked for images"}
    c9_map = None

    def put(i, res):
        scores[i] = res[0]
        details[CUE_IDS[i]] = res[2]
        if want_maps and res[1] is not None:
            maps[CUE_IDS[i]] = res[1]

    put(0, c1_error_level(f))
    put(1, c2_block_dct(f))
    put(2, c3_quant_history(f, cont["luma_table"]))
    if device is not None:
        put(3, c4_prnu(f, device))
    else:
        masked["C4"] = "no device fingerprint K registered for this file: masked"
    put(4, c5_resampling(f))
    put(5, c6_illumination(f))
    s8, why8 = codec.container_score(cont)
    scores[7] = s8
    details["C8"] = why8
    if learned and K > 8:
        s9, c9_map, why9 = c9_learned(media.rgb)
        scores[8] = s9
        details["C9"] = why9
        if want_maps:
            maps["C9"] = c9_map
    elif K > 8:
        masked["C9"] = "erasure network not run on this state"

    hp = f.Y - cv2.GaussianBlur(f.Y, (0, 0), 1.1)
    has_meta = bool(cont["exif"] or cont["text_keys"] or cont["has_xmp"] or cont["has_c2pa"])
    damage = np.array([
        cont["declared_q"] if cont["declared_q"] is not None else 100.0,
        f.lattice_coarseness, f.prior_lattice, f.grid_on, f.grid_off,
        float(np.log(np.median(np.abs(hp)) + 0.01)),
        f.resampling_peak, 1.0 if has_meta else 0.0,
        float(np.log(len(media.data) / max(1, f.full_h * f.full_w))),
        1.0 if cont["chroma_sub"] in ("4:2:0", "4:2:2") else 0.0,
        1.0 if cont["format"] in ("JPEG", "MPO") else 0.0], dtype=np.float32)
    return dict(scores=scores, maps=maps, details=details, masked=masked, damage=damage,
                container=cont, region=(f.y0, f.x0, f.h, f.w), pce=f.pce)
