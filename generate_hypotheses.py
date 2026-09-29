"""
Generate underlying hypotheses from experiment summaries.

Input:  <dataset summaries_dir>/<paper_id>.json
Output: <dataset results_dir(recover)>/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }

Datasets are defined in config.py; --dataset picks one or more (default: all):

  python generate_hypotheses.py                    # every dataset
  python generate_hypotheses.py --dataset plos     # just PLOS Biology
"""

import argparse
import logging

from config import DATASETS, N_SAMPLES, TASKS, parse_dataset_args
from hypothesis_engine import run_batch, setup_logging
from model_utils import build_all_models

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_WORKERS = 24

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

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*", choices=list(DATASETS), help="default: all")
    args = parser.parse_args()

    models = build_all_models()
    task = TASKS["recover"]
    for dataset in parse_dataset_args(args.dataset):
        log.info("=== %s: %s ===", dataset.label, task.label)
        run_batch(
            models=models,
            input_dir=dataset.summaries_dir,
            output_dir=dataset.results_dir(task),
            extract_text=extract_text,
            system_prompt=SYSTEM_PROMPT,
            user_instruction=USER_INSTRUCTION,
            skip_ids=dataset.skip_ids(),
            should_skip_content=should_skip_content,
            n_samples=N_SAMPLES,
            max_workers=MAX_WORKERS,
        )


if __name__ == "__main__":
    main()