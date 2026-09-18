#!/usr/bin/env python
"""Inject the generated result tables into README.md.

    python analysis/aggregate_results.py
    python analysis/update_readme.py

Replaces everything between matched markers::

    <!-- TABLE:OVERALL -->
    ...generated...
    <!-- /TABLE:OVERALL -->

so the README can never drift from `results/tables/*.csv`.  Tables with no
finished runs are written as `TBD` rather than omitted silently.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the README is out of date")
    args = ap.parse_args()

    gen = subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "make_readme_tables.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if gen.returncode != 0:
        raise SystemExit(gen.stderr)
    # `make_readme_tables` emits `<!-- KIND:KEY -->` for the portfolio blocks and
    # a bare `<!-- KEY -->` for the result tables; a bare key defaults to TABLE,
    # which keeps every original marker working unchanged.
    blocks: dict[str, str] = {}
    for kind, key, body in re.findall(
        r"<!-- (?:([A-Za-z]+):)?(\w+) -->\n(.*?)(?=\n<!-- |\Z)", gen.stdout, flags=re.S
    ):
        blocks[f"{kind or 'TABLE'}:{key}"] = body

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    original = text

    injected = []
    for full, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- {full} -->\n).*?(\n<!-- /{full} -->)", flags=re.S
        )
        if not pattern.search(text):
            print(f"  marker {full} not found in README -- skipped")
            continue
        text = pattern.sub(lambda m: m.group(1) + body.strip() + m.group(2), text)
        injected.append(full)

    if args.check:
        if text != original:
            print("README is out of date; run python analysis/update_readme.py")
            raise SystemExit(1)
        print(f"README tables are up to date ({len(injected)} checked)")
        return

    for full in injected:
        print(f"  injected {full}")

    readme.write_text(text, encoding="utf-8")
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
