"""
Generate underlying hypotheses from experiment summaries.

Input:  data{suffix}/experiments_summary/<paper_id>.json
Output: results{suffix}/underlying_hypotheses/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }

Two data sources exist side by side -- data/ (native-format papers) and
data2/ (XML-converted papers) -- each with its own mirrored results{,2}/
tree. Use --source to pick one, or omit it to process both in one run:

  python generate_hypotheses.py             # both data/ and data2/
  python generate_hypotheses.py --source 1  # just data/  -> results/
  python generate_hypotheses.py --source 2  # just data2/ -> results2/
"""

import argparse
import logging
from pathlib import Path

from model_utils import build_all_models
from hypothesis_engine import run_batch, load_skip_ids

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 10
MAX_WORKERS = 24

# Maps --source values to the directory suffix: data/ has no suffix,
# data2/ has "2". Keeping this as an explicit dict (rather than just using
# the CLI value directly as the suffix) means the CLI-facing names don't
# have to be awkward things like an empty-string argument.
SOURCE_SUFFIXES = {"1": "", "2": "2"}

# Paper IDs (one per line) to skip entirely -- never sent to any model,
# never written to the output file. See plos_skip.txt for the current list
# and why: certain papers' text reliably trips a provider content filter
# for the Claude models, and retrying doesn't change that outcome.
SKIP_LIST_PATH = Path("plos_skip.txt")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. Given a description of the "
    "experiments and methods from a research paper, infer the underlying "
    "hypothesis being tested — the core scientific claim the experiments were "
    "designed to validate. A hypothesis is a specific, testable, and falsifiable "
    "prediction about the relationship between variables. "
    "Output ONLY the hypothesis as a single declarative sentence. "
    "Do not include preamble, explanation, or any other text."
)

USER_INSTRUCTION = (
    "Generate a single testable hypothesis based on the experiment description above. "
    "Express it as one declarative sentence (e.g. 'If X, then Y because Z')."
)

# ---------------------------------------------------------------------------
# Input handling specific to this script's schema
# ---------------------------------------------------------------------------

def extract_text(data: dict) -> str:
    return data["experiments_summary"]


def should_skip_content(data: dict) -> str | None:
    """Summary files that failed upstream (during summarization) carry an
    "error" key instead of "experiments_summary" -- nothing usable to send
    a model, so skip rather than crash on the missing key."""
    if "error" in data:
        return f"upstream error: {data['error']}"
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def paths_for_suffix(suffix: str) -> tuple[Path, Path]:
    return Path(f"data{suffix}/experiments_summary"), Path(f"results{suffix}/underlying_hypotheses")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=sorted(SOURCE_SUFFIXES), default=None,
        help="1 = data/ (native format), 2 = data2/ (XML-converted). Default: both.",
    )
    args = parser.parse_args()
    suffixes = [SOURCE_SUFFIXES[args.source]] if args.source else list(SOURCE_SUFFIXES.values())

    models = build_all_models()
    skip_ids = load_skip_ids(SKIP_LIST_PATH)

    for suffix in suffixes:
        summary_dir, output_dir = paths_for_suffix(suffix)
        log.info("=== Source: data%s/ → results%s/ ===", suffix, suffix)
        run_batch(
            models=models,
            input_dir=summary_dir,
            output_dir=output_dir,
            extract_text=extract_text,
            system_prompt=SYSTEM_PROMPT,
            user_instruction=USER_INSTRUCTION,
            skip_ids=skip_ids,
            should_skip_content=should_skip_content,
            n_samples=N_SAMPLES,
            max_workers=MAX_WORKERS,
        )


if __name__ == "__main__":
    main()