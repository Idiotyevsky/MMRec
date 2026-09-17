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
    blocks = dict(re.findall(r"<!-- (\w+) -->\n(.*?)(?=\n<!-- |\Z)", gen.stdout, flags=re.S))

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    original = text

    injected = []
    for key, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- TABLE:{key} -->\n).*?(\n<!-- /TABLE:{key} -->)", flags=re.S
        )
        if not pattern.search(text):
            print(f"  marker TABLE:{key} not found in README -- skipped")
            continue
        text = pattern.sub(lambda m: m.group(1) + body.strip() + m.group(2), text)
        injected.append(key)

    if args.check:
        if text != original:
            print("README is out of date; run python analysis/update_readme.py")
            raise SystemExit(1)
        print(f"README tables are up to date ({len(injected)} checked)")
        return

    for key in injected:
        print(f"  injected TABLE:{key}")

    readme.write_text(text, encoding="utf-8")
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
