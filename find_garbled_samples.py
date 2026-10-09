#!/usr/bin/env python3
"""Report response newline counts for manual inspection without editing data.

Headers and formatting can introduce a few newlines in valid responses.
Sort samples by newline count to highlight outliers.

Usage:
  python find_garbled_samples.py results/new_hypotheses
  python find_garbled_samples.py results2/new_hypotheses --min-newlines 20"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--min-newlines", type=int, default=1,
        help="Only list samples with at least this many embedded newlines "
             "(default: 1 -- shows everything; the distribution summary "
             "printed first is usually more useful than this cutoff for "
             "deciding what's actually worth a look).",
    )
    parser.add_argument(
        "--full-text", action="store_true",
        help="Print the full sample text instead of a truncated preview.",
    )
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("*.json"))
    print(f"Scanning {len(files)} files in {args.results_dir}/\n")

    # Collect newline counts and sample locations for the report.
    flagged: list[tuple[int, str, str, int, str]] = []  # (n_newlines, paper, model, slot, text)
    newline_counts: Counter = Counter()

    for path in files:
        data = json.loads(path.read_text())
        for model_id, samples in data.items():
            for i, sample in enumerate(samples):
                if sample is None:
                    continue
                text = str(sample)
                n = text.count("\n")
                if n >= 1:
                    newline_counts[n] += 1
                    flagged.append((n, path.stem, model_id, i + 1, text))

    # Newline-count distribution.
    if newline_counts:
        print("Newline-count distribution across all samples with >=1 newline:")
        for n in sorted(newline_counts):
            print(f"  {n:>4} newline(s): {newline_counts[n]} sample(s)")
        counts_sorted = sorted(newline_counts)
        gap_note = ""
        if len(counts_sorted) > 1:
            biggest_gap = max(
                (counts_sorted[i + 1] - counts_sorted[i], counts_sorted[i])
                for i in range(len(counts_sorted) - 1)
            )
            if biggest_gap[0] >= 10:
                gap_note = (
                    f"\n  -> Biggest jump: {biggest_gap[1]} to "
                    f"{counts_sorted[counts_sorted.index(biggest_gap[1]) + 1]} newlines. "
                    f"Inspect samples above this gap for garbled output."
                )
        print(gap_note)
        print()

    # List samples in descending newline count.
    flagged.sort(key=lambda t: t[0], reverse=True)
    shown = 0
    for n, paper, model_id, slot, text in flagged:
        if n < args.min_newlines:
            continue
        shown += 1
        preview = text if args.full_text else text.replace("\n", "\\n")[:150]
        print(f"--- {paper} / {model_id} / slot {slot} ({n} newlines) ---")
        print(preview)
        print()

    print(f"Done. {shown} sample(s) shown (of {len(flagged)} with >=1 newline; "
          f"--min-newlines={args.min_newlines}).")
    print("Nothing was deleted. To remove specific ones, edit the JSON files directly,")
    print("or tell me which patterns to treat as garbled and I can build a targeted cleanup.")


if __name__ == "__main__":
    main()
