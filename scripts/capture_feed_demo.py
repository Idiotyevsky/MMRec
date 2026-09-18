#!/usr/bin/env python
"""Capture the playable-feed demo as an animated GIF.

    bash scripts/start_demo.sh &
    python scripts/prepare_media_demo.py
    python scripts/capture_feed_demo.py

Drives the running ``/watch`` page with Playwright:

    open feed -> video plays -> open "Why this video?" -> rank movement
    -> close -> Next -> second video plays

Frames are captured with change detection and timestamped durations (the same
machinery as ``capture_demo.py``), so the GIF plays back at the speed the demo
actually ran and every frame carries information.

This is the **product** story.  ``assets/shortrec_demo.gif`` is the **system**
story; the two are deliberately not the same.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.capture_demo import save_gif  # noqa: E402


def demo_user() -> int:
    p = ROOT / "artifacts" / "demo_media_manifest.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text(encoding="utf-8"))["user_id"])
        except Exception:
            pass
    return 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:5173")
    ap.add_argument("--out", default="assets/feed_playback_demo.gif")
    ap.add_argument("--user-id", type=int, default=None)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--max-mb", type=float, default=8.0)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("playwright is not installed: pip install playwright && "
                         "python -m playwright install chromium")

    user_id = args.user_id if args.user_id is not None else demo_user()
    print(f"demo user: {user_id}")

    captures: list[tuple[Image.Image, float]] = []
    t0 = time.time()
    last_sig: list[np.ndarray | None] = [None]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=1)
        page.goto(f"{args.base_url.rstrip('/')}/watch?user={user_id}",
                  wait_until="networkidle", timeout=180_000)
        # let the first video actually start painting frames
        page.wait_for_timeout(4000)
        state = page.evaluate("""() => { const v=document.querySelector('video');
            return v ? {readyState:v.readyState, w:v.videoWidth} : null; }""")
        print(f"video state: {state}")
        if not state or state.get("readyState", 0) < 2:
            raise SystemExit("the video never became playable; is demo media prepared?")

        def frame(min_delta: float = 0.6) -> None:
            """Capture only when the page (including the video) has changed."""
            buf = io.BytesIO(page.screenshot())
            img = Image.open(buf).convert("RGB")
            sig = np.asarray(img.resize((96, 60)), dtype=np.int16)
            if last_sig[0] is not None and float(np.abs(sig - last_sig[0]).mean()) < min_delta:
                return
            last_sig[0] = sig
            captures.append((img, time.time() - t0))

        # 1. the feed playing
        for _ in range(12):
            frame()
            page.wait_for_timeout(320)

        # 2. open the recommendation trace
        page.get_by_role("button", name="Why this video?").click()
        page.wait_for_timeout(700)
        for _ in range(10):
            frame()
            page.wait_for_timeout(320)

        # 3. close and swipe to the next recommendation
        page.get_by_role("button", name="Hide trace").click()
        page.wait_for_timeout(400)
        page.get_by_role("button", name="Next ↓").click()
        page.wait_for_timeout(1200)
        for _ in range(12):
            frame()
            page.wait_for_timeout(320)

        # 4. the second item's trace, to show a different recall source / movement
        page.get_by_role("button", name="Why this video?").click()
        page.wait_for_timeout(700)
        for _ in range(8):
            frame()
            page.wait_for_timeout(320)

        browser.close()

    out = ROOT / args.out
    path = save_gif(captures, out, args.max_mb, max_seconds=args.max_seconds)
    print(f"    {path.relative_to(ROOT)}  ({path.stat().st_size/1024**2:.2f} MB, "
          f"{len(captures)} frames captured)")


if __name__ == "__main__":
    main()
