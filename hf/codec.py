"""Container I/O: decode, encode, quantisation tables, and the metadata reader behind C8."""
from __future__ import annotations

import io
import re
import struct

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

Image.MAX_IMAGE_PIXELS = 60_000_000

# IJG reference tables, natural (row-major) order.
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.int32)
STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99], dtype=np.int32)


def std_tables(q: int) -> tuple[np.ndarray, np.ndarray]:
    q = int(min(100, max(1, q)))
    scale = 5000 // q if q < 50 else 200 - 2 * q
    f = lambda base: np.clip((base * scale + 50) // 100, 1, 255)  # noqa: E731
    return f(STD_LUMA), f(STD_CHROMA)


_STD_BANK = np.stack([std_tables(q)[0] for q in range(1, 101)])


def quality_of(luma_table) -> tuple[int, bool]:
    """Nearest IJG quality for a luminance table, and whether the match is exact."""
    t = np.asarray(luma_table, dtype=np.int32).reshape(-1)[:64]
    d = np.abs(_STD_BANK - t[None, :]).mean(axis=1)
    j = int(np.argmin(d))
    return j + 1, bool(d[j] == 0)


# ------------------------------------------------------------------------ decode
def _to_rgb(im: Image.Image) -> np.ndarray:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if im.mode in ("I;16", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        a = a / max(1.0, float(a.max())) * 255.0
        im = Image.fromarray(a.astype(np.uint8))
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def decode(data: bytes) -> tuple[np.ndarray, dict]:
    im = Image.open(io.BytesIO(data))
    im.load()
    info = dict(format=(im.format or "?").upper(), width=im.width, height=im.height,
                exif=im.info.get("exif"), luma_table=None, chroma_sub=None,
                progressive=bool(im.info.get("progressive") or im.info.get("progression")))
    q = getattr(im, "quantization", None)
    if q:
        info["luma_table"] = np.asarray(list(q[min(q.keys())]), dtype=np.int32)
    layer = getattr(im, "layer", None)
    if layer and len(layer) >= 3:
        info["chroma_sub"] = f"{layer[0][1]}x{layer[0][2]}"
    return _to_rgb(im), info


def decode_rgb(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data))
    im.load()
    return _to_rgb(im)


# ------------------------------------------------------------------------ encode
def encode_jpeg(rgb, q=None, sub=2, qtables=None, exif=None, progressive=False, xmp: bytes | None = None) -> bytes:
    kw: dict = dict(format="JPEG", subsampling=sub)
    if qtables is not None:
        kw["qtables"] = [[int(v) for v in t] for t in qtables]
    else:
        kw["quality"] = int(q)
    if exif:
        kw["exif"] = exif
    if progressive:
        kw["progressive"] = True
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, **kw)
    return _inject_app1(b.getvalue(), xmp) if xmp else b.getvalue()


def encode_png(rgb, exif=None, text: dict | None = None) -> bytes:
    kw: dict = dict(format="PNG", compress_level=3)
    if exif:
        kw["exif"] = exif
    if text:
        pi = PngInfo()
        for k, v in text.items():
            pi.add_text(k, v)
        kw["pnginfo"] = pi
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, **kw)
    return b.getvalue()


def make_exif(make=None, model=None, software=None, dt_original=None, dt_modified=None,
              serial=None) -> bytes:
    ex = Image.Exif()
    if make:
        ex[271] = make
    if model:
        ex[272] = model
    if software:
        ex[305] = software
    if dt_modified:
        ex[306] = dt_modified
    sub = {}
    if dt_original:
        sub[36867] = dt_original
    if serial:
        sub[42033] = serial
    if sub:
        ex[0x8769] = sub
    return ex.tobytes()


# ------------------------------------------------------------ container reader (C8)
_GENERATOR = re.compile(
    r"stable.?diffusion|sdxl|midjourney|dall[\s\-·.]?e\b|adobe firefly|firefly image|"
    r"\bflux\.?1|black forest labs|comfyui|automatic1111|invokeai|novelai|leonardo\.ai|ideogram|"
    r"gpt-image|openai|google imagen|imagen ?[2-4]\b|made with google ai|gemini|grok|"
    r"trainedalgorithmicmedia|ai[\s\-_]generated|generative ai|toy-generator|"
    r"cfg scale|negative prompt|sampler:|steps: ?\d", re.I)
_EDITOR = re.compile(
    r"photoshop|gimp|lightroom|affinity|pixelmator|snapseed|picsart|canva\b|paint\.net|facetune|"
    r"faceapp|luminar|capture one|darktable|krita|pixlr|photopea|inshot|remini|meitu|corel|"
    r"paintshop|acdsee|photoscape|fotor|befunky", re.I)
# Object removal and generative edits that phone galleries and editors declare in metadata.
_AI_EDIT = re.compile(
    r"clean ?up|magic eraser|magic editor|object eraser|generative (?:fill|erase|edit|expand)|"
    r"ai eraser|ai erase|edited with (?:google )?ai|photo unblur|best take|reimagine|"
    r"remove tool|content-aware fill|sketch to image", re.I)
# IPTC Digital Source Type vocabulary (cv.iptc.org/newscodes/digitalsourcetype)
_DST = {
    "trainedalgorithmicmedia": ("generator", 4.0, "created by a generative AI model"),
    "compositewithtrainedalgorithmicmedia": ("ai_edit", 4.0, "edited with generative AI"),
    "algorithmicmedia": ("generator", 3.0, "algorithmically created"),
    "compositesynthetic": ("ai_edit", 3.0, "composite including synthetic elements"),
    "composite": ("edit", 2.5, "composite of several sources"),
    "minorhumanedits": ("edit", 0.5, "minor human edits"),
    "digitalcapture": ("capture", -1.0, "captured by a camera"),
    "computationalcapture": ("capture", -1.0, "computational camera capture"),
}
_XMP_FIELDS = re.compile(
    r"(DigitalSourceType|Credit|CreatorTool|softwareAgent|Software|description|Source)"
    r"\s*(?:=\s*\"([^\"]{1,300})\"|>\s*(?:<rdf:(?:Alt|Seq|Bag)>\s*<rdf:li[^>]*>)?([^<]{1,300})<)", re.I)


def _xmp_values(xmp: str) -> dict:
    """Field values of interest from an XMP packet, without namespace URIs or tag names."""
    out: dict = {}
    for m in _XMP_FIELDS.finditer(xmp):
        val = (m.group(2) or m.group(3) or "").strip()
        if val:
            out.setdefault(m.group(1), val)
    return out


def _inject_app1(jpeg: bytes, payload: bytes) -> bytes:
    """Insert an APP1 segment right after SOI (and APP0, if present)."""
    i = 2
    if jpeg[2:4] == b"\xff\xe0":
        i = 4 + struct.unpack(">H", jpeg[4:6])[0]
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg[:i] + seg + jpeg[i:]


def make_xmp(credit: str | None = None, source_type: str | None = None, creator_tool: str | None = None) -> bytes:
    attrs = []
    if credit:
        attrs.append(f'photoshop:Credit="{credit}"')
    if source_type:
        attrs.append(f'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/{source_type}"')
    if creator_tool:
        attrs.append(f'xmp:CreatorTool="{creator_tool}"')
    pkt = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
           'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
           'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
           + " ".join(attrs) + "/></rdf:RDF></x:xmpmeta>")
    return b"http://ns.adobe.com/xap/1.0/\x00" + pkt.encode()


def _jpeg_segments(data: bytes) -> list[tuple[str, bytes]]:
    out, i, n = [], 2, len(data)
    while i + 4 <= n and data[i] == 0xFF:
        m = data[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xDA or m == 0xD9:
            break
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        payload = data[i + 4:i + 2 + length]
        if 0xE0 <= m <= 0xEF:
            name = f"APP{m - 0xE0}"
        else:
            name = {0xFE: "COM", 0xDB: "DQT", 0xC0: "SOF0", 0xC2: "SOF2", 0xC4: "DHT",
                    0xDD: "DRI"}.get(m, f"x{m:02X}")
        out.append((name, payload))
        i += 2 + length
    return out


def read_container(data: bytes) -> dict:
    """Everything C8 and the damage hub read from the file bytes, without touching pixels."""
    c = dict(format="?", bytes=len(data), markers=[], exif={}, text_keys=[], has_xmp=False,
             has_icc=False, has_c2pa=False, has_adobe=False, declared_q=None, std_tables=None,
             chroma_sub=None, progressive=False, luma_table=None, generator=None, editor=None,
             ai_edit=None, source_type=None, xmp={})
    hay: list[str] = []          # metadata *values* only: never namespace URIs or tag names
    xmp_raw: list[str] = []
    try:
        im = Image.open(io.BytesIO(data))
        c["format"] = (im.format or "?").upper()
        c["width"], c["height"] = im.size
    except Exception:  # noqa: BLE001
        return c
    q = getattr(im, "quantization", None)
    if q:
        t = np.asarray(list(q[min(q.keys())]), dtype=np.int32)
        c["luma_table"] = t
        c["declared_q"], c["std_tables"] = quality_of(t)
    layer = getattr(im, "layer", None)
    if layer and len(layer) >= 3:
        c["chroma_sub"] = {"2x2": "4:2:0", "2x1": "4:2:2", "1x1": "4:4:4"}.get(
            f"{layer[0][1]}x{layer[0][2]}", f"{layer[0][1]}x{layer[0][2]}")
    c["progressive"] = bool(im.info.get("progressive") or im.info.get("progression"))
    c["has_icc"] = bool(im.info.get("icc_profile"))

    try:
        ex = im.getexif()
        names = {271: "Make", 272: "Model", 305: "Software", 306: "DateTime", 315: "Artist",
                 270: "ImageDescription"}
        for tag, nm in names.items():
            if tag in ex and str(ex[tag]).strip():
                c["exif"][nm] = str(ex[tag]).strip()[:120]
        sub = ex.get_ifd(0x8769)
        for tag, nm in {36867: "DateTimeOriginal", 42033: "BodySerialNumber", 37510: "UserComment",
                        42036: "LensModel"}.items():
            if tag in sub:
                v = sub[tag]
                v = v.decode("latin1", "ignore") if isinstance(v, bytes) else str(v)
                c["exif"][nm] = v.strip("\x00 ").strip()[:160]
    except Exception:  # noqa: BLE001
        pass
    hay += list(c["exif"].values())

    if c["format"] in ("JPEG", "MPO"):          # MPO: iPhone and multi-picture JPEGs
        for name, payload in _jpeg_segments(data):
            ident = payload[:40]
            label = name
            if name == "APP0" and ident.startswith(b"JFIF"):
                label = "APP0 JFIF"
            elif name == "APP1" and ident.startswith(b"Exif"):
                label = "APP1 Exif"
            elif name == "APP1" and b"ns.adobe.com/xap" in ident:
                label, c["has_xmp"] = "APP1 XMP", True
                xmp_raw.append(payload[:60000].decode("utf-8", "ignore"))
            elif name == "APP2" and ident.startswith(b"MPF"):
                label = "APP2 MPF"
            elif name == "APP2" and ident.startswith(b"ICC_PROFILE"):
                label = "APP2 ICC"
            elif name == "APP11":
                label = "APP11 JUMBF"
                if b"c2pa" in payload[:4000] or b"jumb" in payload[:64]:
                    c["has_c2pa"] = True
                hay.append(payload[:60000].decode("latin1", "ignore"))
            elif name == "APP13":
                label, c["has_adobe"] = "APP13 Photoshop", True
            elif name == "APP14" and ident.startswith(b"Adobe"):
                label, c["has_adobe"] = "APP14 Adobe", True
            elif name == "COM":
                hay.append(payload[:4000].decode("latin1", "ignore"))
            if name.startswith("APP") or name == "COM":
                c["markers"].append(label)
    else:
        for k, v in (getattr(im, "text", None) or {}).items():
            c["text_keys"].append(str(k))
            hay.append(f"{k}: {str(v)[:20000]}")
        if b"c2pa" in data[:200000]:
            c["has_c2pa"] = True
        xmp = im.info.get("XML:com.adobe.xmp") or im.info.get("xmp")
        if xmp:
            c["has_xmp"] = True
            xmp_raw.append(xmp.decode("utf-8", "ignore") if isinstance(xmp, bytes) else str(xmp)[:60000])

    for x in xmp_raw:
        for k, v in _xmp_values(x).items():
            c["xmp"].setdefault(k, v[:200])
    hay += list(c["xmp"].values())
    dst = c["xmp"].get("DigitalSourceType", "")
    if dst:
        c["source_type"] = dst.rstrip("/").rsplit("/", 1)[-1].lower()

    blob = "\n".join(hay)
    g = _GENERATOR.search(blob)
    e = _EDITOR.search(blob)
    a = _AI_EDIT.search(blob)
    c["generator"] = g.group(0) if g else None
    c["editor"] = e.group(0) if e else None
    c["ai_edit"] = a.group(0) if a else None
    return c


def container_score(c: dict) -> tuple[float, list[str]]:
    """C8: evidence weights over what the container says about how the file was made.

    Positive means the container points at editing or synthesis; negative means it
    looks like an untouched camera original; zero means the container says nothing
    (which is what every file looks like after a platform round-trip).
    """
    s, why = 0.0, []
    ex = c["exif"]
    dst = _DST.get(c.get("source_type") or "")
    flagged = False
    if dst and dst[1] > 0:
        s += dst[1]
        flagged = True
        why.append(f"IPTC digital source type declares the image {dst[2]}")
    if c.get("ai_edit"):
        credit = c["xmp"].get("Credit") or c["xmp"].get("CreatorTool") or ex.get("Software") or c["ai_edit"]
        if not flagged:
            s += 4.0
        flagged = True
        why.append(f"AI object-removal or generative edit declared: '{credit}'")
    if c["generator"] and not flagged:
        s += 4.0
        flagged = True
        why.append(f"generator marker in metadata: '{c['generator']}'")
    if c["editor"]:
        s += 2.5 if not flagged else 0.0
        why.append(f"editing software tag: '{c['editor']}'")
    if c["has_adobe"] and not c["editor"] and not flagged:
        s += 0.8
        why.append("Adobe/Photoshop application segment present")
    dto, dtm = ex.get("DateTimeOriginal"), ex.get("DateTime")
    if dto and dtm and dto != dtm:
        s += 1.0
        why.append("modify date differs from capture date")
    clean = not (flagged or c["generator"] or c["editor"])
    if ex.get("Make") and ex.get("Model") and dto and clean:
        s -= 1.5
        why.append(f"camera provenance: {ex['Make']} {ex['Model']}, capture date present")
    elif dst and dst[1] < 0 and clean:
        s -= 1.0
        why.append(f"IPTC digital source type declares the image {dst[2]}")
    if not ex and not c["text_keys"] and not c["has_xmp"] and not c["has_c2pa"]:
        why.append("no metadata in the container: nothing to read")
    elif not why:
        why.append("metadata present, with no editing or generator marker and no camera identity")
    return s, why
