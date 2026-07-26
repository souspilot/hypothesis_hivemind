#!/usr/bin/env python3
"""
Strips blank ("") and error ("ERROR: {exc}") samples out of existing
hypothesis-generation output files, in place, leaving the good samples
where they are.

This relies on the TOP-UP resume logic in generate_hypotheses.py /
generate_new_hypotheses.py: for each model_id, it counts how many valid
samples remain and generates only the shortfall (N_SAMPLES - len(valid)),
appending rather than overwriting. So this script's only job is to remove
the bad entries -- it does NOT need to delete the whole model_id key to
force a full regeneration; that would throw away good samples along with
the bad ones and cost extra API calls for no reason.

"ERROR:" is matched as an exact prefix (not "ERROR" anywhere in the
string), matching exactly what sample_model writes on a failed call --
this avoids ever mistaking a genuine hypothesis that happens to mention
error rates, error bars, etc. for a failed sample.

Note: papers listed in plos_skip.txt are never sent to any model in the
first place (see generate_hypotheses.py / generate_new_hypotheses.py), so
their output files are untouched by both the generator and this cleaner.

Usage:
    python clean_blank_samples.py results/new_hypotheses
    python clean_blank_samples.py results2/new_hypotheses
    python clean_blank_samples.py results/underlying_hypotheses
"""

import json
import sys
from pathlib import Path


def is_bad_sample(sample: str) -> bool:
    text = str(sample)
    return text == "" or text.startswith("ERROR:")


def clean_file(path: Path) -> dict[str, dict[str, int]]:
    data = json.loads(path.read_text())
    changed: dict[str, dict[str, int]] = {}

    for model_id, samples in data.items():
        kept = [s for s in samples if not is_bad_sample(s)]
        removed = len(samples) - len(kept)
        if removed:
            blank_count = sum(1 for s in samples if s == "")
            error_count = removed - blank_count
            changed[model_id] = {"blank": blank_count, "error": error_count, "kept": len(kept)}
            data[model_id] = kept  # strip bad entries; keep the good ones in place

    if changed:
        path.write_text(json.dumps(data, indent=2))

    return changed


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    files = sorted(results_dir.glob("*.json"))
    print(f"Scanning {len(files)} files in {results_dir}/\n")

    total_files_affected = 0
    total_removed = 0
    for path in files:
        changed = clean_file(path)
        if changed:
            total_files_affected += 1
            removed_here = sum(v["blank"] + v["error"] for v in changed.values())
            total_removed += removed_here
            print(f"{path.name}: stripped {removed_here} bad samples across {len(changed)} model(s) -- {changed}")

    print(f"\nDone. {total_files_affected}/{len(files)} files touched, {total_removed} bad samples stripped.")
    print("Re-run the main generation script -- it will top up only the missing samples per model.")


if __name__ == "__main__":
    main()