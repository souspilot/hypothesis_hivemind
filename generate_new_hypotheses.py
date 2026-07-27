"""
Generate novel hypotheses from full paper context.

Input:  data/train/<paper_id>.json
Output: results/new_hypotheses/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }
"""

from pathlib import Path

from model_utils import build_all_models
from hypothesis_engine import run_batch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 10
MAX_WORKERS = 24

TRAIN_DIR  = Path("data/train")
OUTPUT_DIR = Path("results/new_hypotheses")

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
    # 1. Current schema (data/train, from xml_to_json.py): a plain STRING
    #    at the top level, under "abstract_text".
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

def main() -> None:
    models = build_all_models()

    run_batch(
        models=models,
        input_dir=TRAIN_DIR,
        output_dir=OUTPUT_DIR,
        extract_text=extract_text,
        system_prompt=SYSTEM_PROMPT,
        user_instruction=USER_INSTRUCTION,
        n_samples=N_SAMPLES,
        max_workers=MAX_WORKERS,
    )


if __name__ == "__main__":
    main()