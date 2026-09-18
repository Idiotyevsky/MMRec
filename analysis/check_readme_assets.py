#!/usr/bin/env python
"""Verify that everything the README points at actually exists.

    python analysis/check_readme_assets.py

Checks:

* every relative image path resolves on disk;
* every relative ``docs/`` link resolves;
* every in-page anchor (``#foo``) matches a heading or an explicit
  ``<a id="foo">`` target;
* HTML tags used by the README are balanced.

External links (shields.io, GitHub Actions) are not fetched: the checker is
offline and deterministic, so it can run in CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slugify(heading: str) -> str:
    """GitHub's heading-to-anchor rule, close enough for our own headings."""
    text = heading.strip().lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    problems: list[str] = []

    # ---- images ----
    images = set(re.findall(r'src="([^"]+\.(?:png|gif|svg|jpg|jpeg|webp))"', text))
    images |= set(re.findall(r"!\[[^\]]*\]\(([^)]+\.(?:png|gif|svg|jpg|jpeg|webp))\)", text))
    for img in sorted(images):
        if img.startswith(("http://", "https://")):
            continue
        if not (ROOT / img).exists():
            problems.append(f"missing image: {img}")

    # ---- docs / relative markdown links ----
    links = set(re.findall(r"\]\(([^)#][^)]*\.md)\)", text))
    for link in sorted(links):
        if link.startswith(("http://", "https://")):
            continue
        if not (ROOT / link).exists():
            problems.append(f"missing link target: {link}")

    # ---- in-page anchors ----
    anchors = set(re.findall(r'<a id="([^"]+)"', text))
    anchors |= {slugify(m) for m in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M)}
    for target in sorted(set(re.findall(r"\]\(#([\w-]+)\)", text))):
        if target not in anchors:
            problems.append(f"broken anchor: #{target}")

    # ---- tag balance ----
    for tag in ("details", "summary", "table", "tr", "td", "p"):
        opens = len(re.findall(rf"<{tag}[ >]", text))
        closes = len(re.findall(rf"</{tag}>", text))
        if opens != closes:
            problems.append(f"unbalanced <{tag}>: {opens} open, {closes} close")

    print(f"images: {len(images)} | links: {len(links)} | anchors: {len(anchors)}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("README assets, links, anchors and tags: OK")


if __name__ == "__main__":
    main()
