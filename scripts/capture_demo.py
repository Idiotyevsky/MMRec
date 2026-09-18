#!/usr/bin/env python
"""Capture the ShortRec demo: static screenshots and an animated GIF.

    bash scripts/start_demo.sh &
    python scripts/capture_demo.py                 # screenshots + GIF
    python scripts/capture_demo.py --no-gif        # screenshots only

Everything is captured from the **running application** driven by Playwright.
The GIF follows one real request:

    select the demo user -> Run Recommendation -> staged pipeline reveal
    -> compare with SASRec -> open the Inspector trace

The user comes from ``artifacts/demo_user.json`` (written by
``scripts/find_demo_user.py``), so the case shown is chosen by explicit criteria
rather than by hand.

GIF assembly uses Pillow; if the result is larger than ``--max-mb`` it is
re-encoded at a smaller width and lower frame rate.  Nothing is retouched.
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


def load_demo_user() -> int:
    path = ROOT / "artifacts" / "demo_user.json"
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["user_id"])
        except Exception:
            pass
    return 7


def save_gif(captures: list[tuple[Image.Image, float]], out: Path, max_mb: float,
             max_seconds: float = 18.0) -> Path:
    """Assemble timestamped frames into a looping GIF.

    Each frame carries the wall-clock time at which it was captured, so the GIF
    plays back at the speed the demo actually ran.  A fixed frame duration would
    silently speed the animation up (or slow it down) by whatever ratio the
    capture loop happens to have.
    """
    if not captures:
        raise SystemExit("no frames captured")
    frames = [f for f, _ in captures]
    stamps = [t for _, t in captures]
    # per-frame duration = real elapsed time to the next capture
    durations = []
    for i in range(len(stamps)):
        nxt = stamps[i + 1] if i + 1 < len(stamps) else stamps[i] + (stamps[-1] - stamps[-2] if len(stamps) > 1 else 0.4)
        durations.append(max(int((nxt - stamps[i]) * 1000), 60))

    def encode(width: int, every: int, colours: int) -> tuple[Path, float]:
        idx = list(range(0, len(frames), every))
        scaled = []
        for i in idx:
            f = frames[i]
            h = int(f.height * width / f.width)
            scaled.append(f.resize((width, h), Image.LANCZOS).convert(
                "P", palette=Image.ADAPTIVE, colors=colours))
        durs = [durations[i] for i in idx]
        target = out if width == frames[0].width else out.with_name(out.stem + f"_{width}.gif")
        scaled[0].save(
            target, save_all=True, append_images=scaled[1:],
            duration=durs, loop=0, optimize=True, disposal=2,
        )
        return target, target.stat().st_size / 1024**2

    total = sum(durations) / 1000
    if max_seconds and total > max_seconds:
        # preserve the relative pacing but fit the target length; a GIF that
        # takes 25 s to say what can be said in 16 s loses the viewer
        scale = max_seconds / total
        durations = [max(int(d * scale), 60) for d in durations]
        print(f"    gif timeline scaled {total:.1f}s -> {sum(durations)/1000:.1f}s")
    total = sum(durations) / 1000
    print(f"    gif timeline: {len(frames)} frames, {total:.1f} s")
    for width, every, colours in (
        (frames[0].width, 1, 128),
        (1200, 2, 96),
        (1000, 2, 64),
        (860, 3, 48),
    ):
        path, mb = encode(width, every, colours)
        print(f"    gif {width}px every={every} colours={colours}: {mb:.2f} MB")
        if mb <= max_mb:
            return path
    print(f"    warning: smallest encoding still {mb:.2f} MB (limit {max_mb})")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:5173")
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--user-id", type=int, default=None, help="default: artifacts/demo_user.json")
    ap.add_argument("--max-mb", type=float, default=8.0)
    ap.add_argument("--max-seconds", type=float, default=17.0,
                    help="scale the GIF timeline down to at most this many seconds")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-screenshots", action="store_true")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("playwright is not installed: pip install playwright && "
                         "python -m playwright install chromium")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    user_id = args.user_id if args.user_id is not None else load_demo_user()
    print(f"demo user: {user_id}")

    captures: list[tuple[Image.Image, float]] = []
    t_start = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=1)
        base = args.base_url.rstrip("/")

        def shot(name: str) -> None:
            page.screenshot(path=str(out_dir / f"{name}.png"), full_page=False)
            print(f"    wrote {name}.png")

        _last_sig: list[float | None] = [None]

        def frame(min_delta: float = 0.08) -> None:
            """Capture a frame, skipping it if the page has not visibly changed.

            Without this the GIF is mostly duplicate stills: the capture loop
            runs faster than the UI updates, and Pillow then collapses the
            identical frames into long holds.  Keeping only distinct states makes
            every GIF frame carry information.
            """
            buf = io.BytesIO(page.screenshot())
            img = Image.open(buf).convert("RGB")
            sig = np.asarray(img.resize((96, 60)), dtype=np.int16)
            if _last_sig[0] is not None:
                delta = float(np.abs(sig - _last_sig[0]).mean())
                if delta < min_delta:
                    return
            _last_sig[0] = sig
            captures.append((img, time.time() - t_start))

        # ---------------- Live Demo ----------------
        print("==> /")
        page.goto(base + "/", wait_until="networkidle", timeout=180_000)
        page.wait_for_timeout(1500)

        # pick the demo user
        inp = page.get_by_label("user id")
        inp.fill(str(user_id))
        page.get_by_role("button", name="Go", exact=True).click()
        page.wait_for_timeout(1200)

        # GIF pacing: Pillow collapses consecutive identical frames, so capture
        # at the cadence the page actually changes (the reveal ticks every
        # ~525 ms) and scroll smoothly, otherwise the GIF degenerates into a
        # handful of stills.
        if not args.no_gif:
            for _ in range(2):
                frame()
                page.wait_for_timeout(320)

        page.get_by_role("button", name="Run Recommendation").click()
        # the staged reveal is ~4.2 s; capture one frame per stage
        if not args.no_gif:
            for _ in range(10):
                frame()
                page.wait_for_timeout(430)
        else:
            page.wait_for_timeout(6000)

        def scroll_capture(total: int, steps: int, pause: int = 300) -> None:
            per = max(total // max(steps, 1), 1)
            for _ in range(steps):
                page.mouse.wheel(0, per)
                page.wait_for_timeout(pause)
                if not args.no_gif:
                    frame()

        scroll_capture(1100, 3)          # feed cards
        scroll_capture(1200, 3)          # comparison panel
        if not args.no_gif:
            frame()
            page.wait_for_timeout(400)
            frame()
        if not args.no_screenshots:
            page.mouse.wheel(0, -3000)
            page.wait_for_timeout(700)
            shot("live_demo")

        # ---------------- Inspector ----------------
        print("==> /inspect")
        page.goto(base + "/inspect", wait_until="networkidle", timeout=180_000)
        page.wait_for_timeout(1500)
        # same user as the demo, so the trace shown is the case that was selected
        page.get_by_label("user id").fill(str(user_id))
        page.get_by_role("button", name="Go", exact=True).click()
        page.wait_for_timeout(3000)
        if not args.no_gif:
            frame()
        # open the trace of the item the multimodal ranker moved the most
        moved = page.locator(".badge.cold").first
        if moved.count() > 0:
            try:
                moved.click(timeout=3000)
                page.wait_for_timeout(1300)
                if not args.no_gif:
                    frame()
                    page.wait_for_timeout(400)
                    frame()
            except Exception:
                pass
        if not args.no_screenshots:
            shot("inspector_demo")

        # ---------------- Cold Start ----------------
        print("==> /cold")
        page.goto(base + "/cold", wait_until="networkidle", timeout=180_000)
        page.wait_for_timeout(2500)
        if not args.no_screenshots:
            shot("cold_start_demo")

        # ---------------- System ----------------
        print("==> /system")
        page.goto(base + "/system", wait_until="networkidle", timeout=180_000)
        page.wait_for_timeout(2500)
        if not args.no_screenshots:
            shot("system_architecture")
            # the charts and the trade-off table are the informative part
            page.mouse.wheel(0, 1450)
            page.wait_for_timeout(1200)
            shot("system_demo")

        browser.close()

    if not args.no_gif:
        print("==> assembling gif")
        path = save_gif(captures, out_dir / "shortrec_demo.gif", args.max_mb,
                        max_seconds=args.max_seconds)
        print(f"    {path.relative_to(ROOT)}  ({path.stat().st_size/1024**2:.2f} MB, "
              f"{len(captures)} frames captured)")
    print("done")


if __name__ == "__main__":
    main()
