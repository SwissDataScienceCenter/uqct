"""Rename non-canonical boundary outputs to the legacy 'boundary_diffusion:...'
filename pattern. Idempotent: skips files that already match the canonical name.

Source pattern:   <dataset>_<intensity>_<uuid>_<start>-<end>_boundary.h5
Target pattern:   boundary_diffusion:<dataset>:<intensity>:True:<start>-<end>:0:<mtime>.h5

Run on the cluster (where the cal h5s live) or locally if needed.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

from uqct.utils import get_results_dir

SRC = re.compile(
    r"^(?P<ds>lung|composite|lamino)_(?P<intensity>[\d.eE+-]+)_"
    r"[\w-]+_(?P<start>\d+)-(?P<end>\d+)_boundary\.h5$"
)


def main() -> int:
    boundary_dir = get_results_dir() / "boundary_sampling"
    if not boundary_dir.is_dir():
        print(f"no such dir: {boundary_dir}", file=sys.stderr)
        return 1
    renamed = 0
    skipped = 0
    for f in sorted(boundary_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name.startswith("boundary_diffusion:"):
            skipped += 1
            continue
        m = SRC.match(f.name)
        if m is None:
            print(f"  ?? skip (no match): {f.name}")
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S.%f")
        new_name = (
            f"boundary_diffusion:{m['ds']}:{m['intensity']}:True:"
            f"{m['start']}-{m['end']}:0:{mtime}.h5"
        )
        target = f.with_name(new_name)
        if target.exists():
            print(f"  !! target already exists, skipping: {target.name}")
            continue
        print(f"  {f.name} -> {target.name}")
        os.rename(f, target)
        renamed += 1
    print(f"\nrenamed {renamed}, already-canonical {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
