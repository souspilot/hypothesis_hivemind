#!/usr/bin/env python3
"""
Removes model entries containing invalid samples (blank, "ERROR: ..."
strings, or garbled/degenerate output -- see hypothesis_engine.is_valid
for the full definition) from existing hypothesis-generation output
files, so the next run of the main script regenerates ONLY those
model/paper combinations -- not the whole file.

Relies on the existing skip logic in generate_hypotheses.py /
generate_new_hypotheses.py: valid existing samples are preserved and only
the shortfall gets regenerated. Deleting a model_id key entirely (rather
than leaving a partial, contaminated list) makes that logic correctly see
it as "needs topping up", so a re-run fills in just what's missing --
papers/models that already came back clean are left alone.

Usage:
    python clean_blank_samples.py results/new_hypotheses
    python clean_blank_samples.py results2/new_hypotheses
    python clean_blank_samples.py results/underlying_hypotheses
    python clean_blank_samples.py results2/underlying_hypotheses
"""

import json
import sys
from pathlib import Path

from hypothesis_engine import is_valid


def clean_file(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    removed = {}

    for model_id in list(data.keys()):
        samples = data[model_id]
        bad_count = sum(1 for s in samples if not is_valid(s))
        if bad_count > 0:
            removed[model_id] = bad_count
            del data[model_id]

    if removed:
        path.write_text(json.dumps(data, indent=2))

    return removed


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    files = sorted(results_dir.glob("*.json"))
    print(f"Scanning {len(files)} files in {results_dir}/\n")

    total_files_affected = 0
    for path in files:
        removed = clean_file(path)
        if removed:
            total_files_affected += 1
            print(f"{path.name}: cleared {removed}")

    print(f"\nDone. {total_files_affected}/{len(files)} files had invalid entries removed.")
    print("Re-run the main generation script to regenerate just those model/paper combos.")


if __name__ == "__main__":
    main()