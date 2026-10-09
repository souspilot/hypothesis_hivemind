#!/usr/bin/env python3
"""Remove invalid responses from stored hypothesis outputs.

Preserve valid responses; remove a model key only if none remain.
The next generation run fills the missing samples. This script edits data.

Usage:
  python clean_blank_samples.py results/new_hypotheses
  python clean_blank_samples.py results2/underlying_hypotheses"""

import json
import sys
from pathlib import Path

from hypothesis_engine import is_valid


def clean_file(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    removed = {}

    for model_id in list(data.keys()):
        samples = data[model_id]
        valid = [s for s in samples if is_valid(s)]
        bad_count = len(samples) - len(valid)

        if bad_count == 0:
            continue

        removed[model_id] = bad_count
        if valid:
            # Preserve valid responses for the next generation run.
            data[model_id] = valid
        else:
            # Remove model entries with no valid responses.
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
