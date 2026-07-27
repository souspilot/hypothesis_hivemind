"""
Generate underlying hypotheses from experiment summaries.

Input:  data2/experiments_summary/<paper_id>.json
Output: results2/underlying_hypotheses/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }
"""

from pathlib import Path

from model_utils import build_all_models
from hypothesis_engine import run_batch, load_skip_ids

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 10
MAX_WORKERS = 24

SUMMARY_DIR = Path("data2/experiments_summary")
OUTPUT_DIR  = Path("results2/underlying_hypotheses")

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

def main() -> None:
    models = build_all_models()
    skip_ids = load_skip_ids(SKIP_LIST_PATH)

    run_batch(
        models=models,
        input_dir=SUMMARY_DIR,
        output_dir=OUTPUT_DIR,
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