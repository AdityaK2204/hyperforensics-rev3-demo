"""02 Degradation engine: the nine-rung ladder and the lifecycle permutation engine.

A `Media` is one state of a file: its pixels and the exact container bytes of
that state. Every operation is deterministic, so any sample reaches any rung, or
any lifecycle path, from its bytes and the config registry alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import codec
from .config import EVENTS, LADDER, RUNG_IDS


@dataclass
class Media:
    rgb: np.ndarray               # H x W x 3 uint8, pixels of this state
    data: bytes                   # container bytes of this state
    exif: bytes | None = None     # metadata that a re-save would carry along
    trace: tuple = ()             # operations applied so far

    @classmethod
    def from_bytes(cls, data: bytes) -> "Media":
        rgb, info = codec.decode(data)
        return cls(rgb=rgb, data=data, exif=info.get("exif"))


def apply_ops(m: Media, ops: list) -> Media:
    if not ops:
        return m
    rgb, exif, data = m.rgb, m.exif, m.data
    dirty = False                 # pixels or metadata changed since the last encode
    for name, p in ops:
        if name == "jpeg":
            data = codec.encode_jpeg(rgb, q=p["q"], sub=p.get("sub", 2), exif=exif,
                                     progressive=p.get("progressive", False))
            rgb = codec.decode_rgb(data)
            dirty = False
        elif name == "resize":
            h, w = rgb.shape[:2]
            s = float(p["scale"])
            nw, nh = max(48, int(round(w * s))), max(48, int(round(h * s)))
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
            dirty = True
        elif name == "shift":      # drop rows/cols at the top-left: the 8x8 grid no longer lines up
            rgb = rgb[p["dy"]:, p["dx"]:]
            dirty = True
        elif name == "crop":
            h, w = rgb.shape[:2]
            rgb = rgb[p["t"]:h - p["b"], p["l"]:w - p["r"]]
            dirty = True
        elif name == "strip":
            exif = None
            dirty = True
        else:
            raise ValueError(f"unknown ladder op: {name}")
    rgb = np.ascontiguousarray(rgb)
    if dirty:
        data = codec.encode_png(rgb, exif=exif)
    return Media(rgb=rgb, data=data, exif=exif, trace=m.trace + tuple(o[0] for o in ops))


def rung(m: Media, r: int | str) -> Media:
    """Take a file to rung r of the ladder (r is an index 0-8 or an id 'R0'-'R8')."""
    i = RUNG_IDS.index(r) if isinstance(r, str) else int(r)
    return apply_ops(m, LADDER[i]["ops"])


def ladder_states(m: Media) -> list[Media]:
    return [rung(m, i) for i in range(len(LADDER))]


def lifecycle_states(m: Media, depth: int) -> dict[tuple, Media]:
    """Every ordered arrangement of lifecycle events up to `depth`, sharing prefixes.

    With four events and depth three this is 4 + 12 + 24 = 40 distinct states: each
    is one plausible history of the file after it left the person who made it.
    """
    out: dict[tuple, Media] = {(): m}
    frontier: list[tuple] = [()]
    for _ in range(depth):
        nxt = []
        for path in frontier:
            for ev in EVENTS:
                if ev["id"] in path:
                    continue
                p2 = path + (ev["id"],)
                out[p2] = apply_ops(out[path], ev["ops"])
                nxt.append(p2)
        frontier = nxt
    del out[()]
    return out
