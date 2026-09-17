#!/usr/bin/env python
"""Capture demo screenshots from a running ShortRec frontend.

    bash scripts/start_demo.sh &
    python scripts/capture_screenshots.py --base-url http://127.0.0.1:5173

Writes ``assets/{feed,inspector,cold_start,system}_demo.png``.  These are real
screenshots of the running application; nothing is mocked or retouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    ("feed_demo", "/", "Recommended feed"),
    ("inspector_demo", "/inspect", "Recommendation Inspector"),
    ("cold_start_demo", "/cold", "Cold Start Explorer"),
    ("system_demo", "/system", "System"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:5173")
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--wait-ms", type=int, default=4000)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("playwright is not installed: pip install playwright && "
                         "python -m playwright install chromium")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=2)
        for name, route, expect in PAGES:
            if args.only and name not in args.only:
                continue
            url = args.base_url.rstrip("/") + route
            print(f"==> {url}")
            page.goto(url, wait_until="networkidle", timeout=120_000)
            page.wait_for_timeout(args.wait_ms)
            # make sure the page actually rendered content, not just a shell
            body = page.inner_text("body")
            if expect.split()[0].lower() not in body.lower() and "shortrec" not in body.lower():
                print(f"    warning: expected text {expect!r} not found on the page")
            path = out_dir / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            written.append(path)
            print(f"    wrote {path.relative_to(ROOT)}")
        browser.close()

    print(f"\ncaptured {len(written)} screenshot(s)")
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
