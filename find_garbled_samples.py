#!/usr/bin/env python3
"""
Reports samples containing embedded newlines, for manual review.

This does NOT delete or modify anything -- it's purely a report. A sample
containing a newline isn't necessarily garbled (the model may have
ignored the "single sentence" instruction in an otherwise coherent way,
or used a newline as legitimate formatting) -- it's just a candidate
worth a human look, not an automatic verdict.

Usage:
    python find_garbled_samples.py results/new_hypotheses
    python find_garbled_samples.py results2/new_hypotheses --min-newlines 1
    python find_garbled_samples.py results/underlying_hypotheses --full-text
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--min-newlines", type=int, default=2,
        help="Flag samples with at least this many embedded newlines (default: 2, "
             "matching what the two confirmed-garbled examples had: 4 and 7).",
    )
    parser.add_argument(
        "--full-text", action="store_true",
        help="Print the full sample text instead of a truncated preview.",
    )
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("*.json"))
    print(f"Scanning {len(files)} files in {args.results_dir}/ "
          f"(flagging >= {args.min_newlines} embedded newlines)\n")

    total_flagged = 0
    for path in files:
        data = json.loads(path.read_text())
        for model_id, samples in data.items():
            for i, sample in enumerate(samples):
                if sample is None:
                    continue
                text = str(sample)
                n_newlines = text.count("\n")
                if n_newlines >= args.min_newlines:
                    total_flagged += 1
                    preview = text if args.full_text else text.replace("\n", "\\n")[:150]
                    print(f"--- {path.stem} / {model_id} / slot {i + 1} ({n_newlines} newlines) ---")
                    print(preview)
                    print()

    print(f"Done. {total_flagged} sample(s) flagged for review.")
    print("Nothing was deleted. To remove specific ones, edit the JSON files directly,")
    print("or tell me which patterns to treat as garbled and I can build a targeted cleanup.")


if __name__ == "__main__":
    main()