#!/usr/bin/env python3
"""
Removes model entries containing blank ("") samples OR error samples
(text containing "ERROR", matching the "ERROR: {exc}" strings written by
sample_model's except block) from existing hypothesis-generation output
files, so the next run of the main script regenerates ONLY those
model/paper combinations -- not the whole file.

Relies on the existing skip logic in generate_hypotheses.py /
generate_new_hypotheses.py: `if len(result.get(model_id, [])) >= N_SAMPLES:
skip`. Deleting a model_id key entirely (rather than leaving a partial,
contaminated list) makes that check correctly see it as "not done yet",
so a re-run regenerates a full fresh set of N_SAMPLES for just that model
-- papers/models that already came back clean are left alone.

Usage:
    python clean_blank_samples.py results/new_hypotheses
    python clean_blank_samples.py results2/new_hypotheses
    python clean_blank_samples.py results/underlying_hypotheses
"""

import json
import sys
from pathlib import Path


def is_bad_sample(sample: str) -> bool:
    """A sample is bad if it's blank, or if it's one of the literal
    "ERROR: ..." strings sample_model writes when a call raises an
    exception. "ERROR" is matched case-sensitively (all caps only) so
    this doesn't accidentally match a genuine hypothesis that happens to
    discuss error rates, error bars, etc. in lowercase or mixed case."""
    return sample == "" or "ERROR" in sample


def clean_file(path: Path) -> dict[str, dict[str, int]]:
    data = json.loads(path.read_text())
    removed = {}

    for model_id in list(data.keys()):
        samples = data[model_id]
        blank_count = sum(1 for s in samples if s == "")
        error_count = sum(1 for s in samples if s != "" and "ERROR" in s)

        if blank_count > 0 or error_count > 0:
            removed[model_id] = {"blank": blank_count, "error": error_count}
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

    print(f"\nDone. {total_files_affected}/{len(files)} files had blank/error entries removed.")
    print("Re-run the main generation script to regenerate just those model/paper combos.")


if __name__ == "__main__":
    main()