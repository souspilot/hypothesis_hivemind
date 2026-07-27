"""
Generate novel hypotheses from full paper context.

Input:  data{suffix}/train/<paper_id>.json
Output: results{suffix}/new_hypotheses/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }

Two data sources exist side by side -- data/ (native-format papers) and
data2/ (XML-converted papers) -- each with its own mirrored results{,2}/
tree. Use --source to pick one, or omit it to process both in one run:

  python generate_new_hypotheses.py             # both data/ and data2/
  python generate_new_hypotheses.py --source 1  # just data/  -> results/
  python generate_new_hypotheses.py --source 2  # just data2/ -> results2/
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
# data2/ has "2".
SOURCE_SUFFIXES = {"1": "", "2": "2"}

# Paper IDs (one per line) to skip entirely -- shared with
# generate_hypotheses.py. Certain papers' text reliably trips a provider
# content filter (seen in practice: anthropic/claude-sonnet-4.6 returning
# "blocked by the provider's content filter" on every single sample,
# never succeeding), and retrying doesn't change that outcome -- it just
# burns API calls on a guaranteed failure every run. See plos_skip.txt.
SKIP_LIST_PATH = Path("plos_skip.txt")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert research scientist. Given the context of a research paper, "
    "your task is to generate a single novel hypothesis that logically extends "
    "beyond the paper's existing findings — not a restatement of them. "
    "The hypothesis must be: (1) grounded in a gap or open question identified "
    "in the paper, (2) specific and testable, (3) falsifiable. "
    "Output ONLY the hypothesis as a single declarative sentence with no preamble or explanation."
)

USER_INSTRUCTION = (
    "Based on the research context above, generate one novel hypothesis "
    "that extends beyond what this paper has already established."
)

# ---------------------------------------------------------------------------
# Input handling specific to this script's schema
# ---------------------------------------------------------------------------

def extract_text(data: dict) -> str:
    title = data.get("title", "")

    # Three schema shapes have shown up across this project's data sources,
    # so we check each in order rather than assuming just one:
    #
    # 1. Current schema (data{suffix}/train, from xml_to_json.py): a plain
    #    STRING at the top level, under "abstract_text".
    # 2. Old S2ORC-style schema: a plain STRING at the top level, under
    #    "abstract" directly -- already exactly what we want, no
    #    reconstruction needed.
    # 3. Same old schema, fallback path: a LIST of paragraph-dicts nested
    #    inside pdf_parse.abstract (redundant with #2 in the same files,
    #    but kept as a fallback in case a file only has this form).
    abstract = data.get("abstract_text", "")

    if not abstract:
        raw_abstract = data.get("abstract", "")
        if isinstance(raw_abstract, str):
            abstract = raw_abstract
        elif isinstance(raw_abstract, list):
            abstract = " ".join(b.get("text", "") for b in raw_abstract)

    if not abstract:
        abstract = " ".join(
            b.get("text", "") for b in data.get("pdf_parse", {}).get("abstract", [])
        )

    body = "\n\n".join(b["text"] for b in data.get("pdf_parse", {}).get("body_text", []))
    return f"Title: {title}\n\nAbstract: {abstract}\n\n{body}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def paths_for_suffix(suffix: str) -> tuple[Path, Path]:
    return Path(f"data{suffix}/train"), Path(f"results{suffix}/new_hypotheses")


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
        train_dir, output_dir = paths_for_suffix(suffix)
        log.info("=== Source: data%s/ → results%s/ ===", suffix, suffix)
        run_batch(
            models=models,
            input_dir=train_dir,
            output_dir=output_dir,
            extract_text=extract_text,
            system_prompt=SYSTEM_PROMPT,
            user_instruction=USER_INSTRUCTION,
            skip_ids=skip_ids,
            n_samples=N_SAMPLES,
            max_workers=MAX_WORKERS,
        )


if __name__ == "__main__":
    main()