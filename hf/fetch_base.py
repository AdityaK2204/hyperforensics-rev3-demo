"""Fetch the lossless source scenes for the controlled core set.

The Kodak Lossless True Color Image Suite: 24 never-compressed photographs
(768x512 PNG), the standard choice when a true R0 with no compression history is
needed. About 15 MB in total. Run once:  python -m hf.fetch_base
"""
from __future__ import annotations

import sys
import urllib.request

from .config import BASE_DIR

URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def fetch(n: int = 24) -> int:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    have = 0
    for i in range(1, n + 1):
        dst = BASE_DIR / f"kodim{i:02d}.png"
        if dst.exists() and dst.stat().st_size > 100_000:
            have += 1
            continue
        req = urllib.request.Request(URL.format(i), headers={"User-Agent": "Mozilla/5.0"})
        try:
            blob = urllib.request.urlopen(req, timeout=60).read()
        except Exception as exc:  # noqa: BLE001
            print(f"  kodim{i:02d}: failed ({exc})", flush=True)
            continue
        if not blob.startswith(PNG_MAGIC):
            print(f"  kodim{i:02d}: not a PNG, skipped", flush=True)
            continue
        dst.write_bytes(blob)
        have += 1
        print(f"  kodim{i:02d}: {len(blob) // 1024} KB", flush=True)
    return have


if __name__ == "__main__":
    got = fetch()
    print(f"{got}/24 source scenes in {BASE_DIR}")
    sys.exit(0 if got >= 12 else 1)
