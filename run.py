"""HyperForensics Rev 3 demo launcher.

    python run.py                 start the demo at http://127.0.0.1:8765
    python run.py --open          ...and open it in the browser
    python run.py --rebuild       regenerate the corpus and every measurement first
    python run.py --port 9000

The first run builds the reference (source scenes, controlled core set, score
tensor, measurements, fusion arms). That takes several minutes once; afterwards
the server starts in a few seconds.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    ap = argparse.ArgumentParser(description="HyperForensics Rev 3 demo")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rebuild", action="store_true", help="rebuild corpus and reference from scratch")
    ap.add_argument("--open", action="store_true", help="open the browser when the server is up")
    ap.add_argument("--demo-dir", help="folder behind the 'Demo images' button (default: demo_images/)")
    args = ap.parse_args()
    if args.demo_dir:
        import os
        os.environ["HF_DEMO_DIR"] = os.path.abspath(args.demo_dir)

    from hf import build, fetch_base
    from hf.config import BASE_DIR

    if len(list(BASE_DIR.glob("kodim*.png"))) < 12:
        print("Fetching the lossless source scenes (Kodak suite, about 15 MB)...")
        if fetch_base.fetch() < 12:
            raise SystemExit("Could not fetch the source scenes. Check the connection and run again.")
    if args.rebuild or not build.ready():
        print("Building the reference. This runs once and takes several minutes.")
        build.build(force=args.rebuild)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"\n  HyperForensics Rev 3 demo  ->  {url}\n")
    if args.open:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("hf.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
