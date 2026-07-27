"""
Generate novel hypotheses from full paper context.

Input:  data/train/<paper_id>.json
Output: results/new_hypotheses/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }
"""

import json
import logging
import time
from pathlib import Path

from model_utils import BaseModel, build_all_models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 10
SLEEP_BETWEEN_CALLS = 0.01

TRAIN_DIR  = Path("data/processed")
OUTPUT_DIR = Path("results/new_hypotheses")

# Paper IDs (one per line) to skip entirely -- never sent to any model,
# never written to the output file. See plos_skip.txt for the current list
# and why: certain papers' text reliably trips a provider content filter
# for the Claude models, and retrying doesn't change that outcome.
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
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid(sample: str) -> bool:
    """A sample is invalid if it's blank (empty-string generation) or if
    it's one of the literal 'ERROR: ...' strings written when a call
    raises an exception. Neither is a usable hypothesis."""
    text = str(sample)
    return text != "" and not text.startswith("ERROR:")


def _load_skip_ids(path: Path) -> set[str]:
    """Paper IDs to skip entirely, one per line (blank lines ignored).
    Matches paper_path.stem, e.g. 'journal.pbio.3003761' -- not the
    filename with its .json extension."""
    if not path.exists():
        return set()
    ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if ids:
        log.info("Loaded %d paper ID(s) to skip from %s", len(ids), path)
    return ids


SKIP_IDS = _load_skip_ids(SKIP_LIST_PATH)

# ---------------------------------------------------------------------------
# Paper text extraction
# ---------------------------------------------------------------------------

def extract_paper_text(data: dict) -> str:
    title = data.get("title", "")

    # Current schema (data/train, produced by xml_to_json.py) stores the
    # abstract as a plain STRING at the top level, under "abstract_text" --
    # not nested inside pdf_parse at all. Reading pdf_parse.get("abstract")
    # silently returns nothing for every file in this format (no error,
    # no warning -- the abstract just vanishes from the prompt).
    abstract = data.get("abstract_text", "")

    if not abstract:
        # Fallback for the older PDF-derived S2ORC-style format, in case
        # any files in that shape end up in this folder too: there,
        # abstract was a top-level list of paragraph dicts (still not
        # nested inside pdf_parse).
        abstract = " ".join(
            b.get("text", "") for b in data.get("abstract", [])
        )

    body = "\n\n".join(b["text"] for b in data.get("pdf_parse", {}).get("body_text", []))
    return f"Title: {title}\n\nAbstract: {abstract}\n\n{body}"

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_new_hypothesis(model: BaseModel, paper_text: str) -> str:
    return model.generate(SYSTEM_PROMPT, paper_text, USER_INSTRUCTION)


def sample_model(model: BaseModel, model_id: str, paper_text: str, n_needed: int) -> list[str]:
    """Generate exactly n_needed new samples (the shortfall, not a full fresh batch)."""
    samples = []
    for i in range(n_needed):
        try:
            text = get_new_hypothesis(model, paper_text)
            samples.append(text)
            log.info("    [%d/%d] %s", i + 1, n_needed, text[:90])
        except Exception as exc:
            log.error("    [%d/%d] ERROR: %s", i + 1, n_needed, exc)
            samples.append(f"ERROR: {exc}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return samples

# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_paper(models: dict[str, BaseModel], paper_path: Path) -> dict | None:
    paper_id = paper_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    if paper_id in SKIP_IDS:
        log.info("Skipping %s — listed in %s", paper_id, SKIP_LIST_PATH)
        return None

    with open(paper_path) as f:
        paper_text = extract_paper_text(json.load(f))

    result: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)

    for model_id, model in models.items():
        existing = result.get(model_id, [])
        valid = [s for s in existing if _is_valid(s)]
        n_needed = N_SAMPLES - len(valid)

        if n_needed <= 0:
            log.info("  [skip] %s already has %d valid samples", model_id, len(valid))
            result[model_id] = valid[:N_SAMPLES]
            continue

        log.info("  %s has %d valid samples, topping up %d more...", model_id, len(valid), n_needed)
        new_samples = sample_model(model, model_id, paper_text, n_needed)
        result[model_id] = valid + new_samples

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    models = build_all_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_files = sorted(TRAIN_DIR.glob("*.json"))
    log.info(
        "Found %d papers | %d models | %d samples → up to ~%d API calls",
        len(paper_files), len(models), N_SAMPLES,
        len(paper_files) * len(models) * N_SAMPLES,
    )

    success = skipped = errors = 0
    for i, paper_path in enumerate(paper_files, 1):
        log.info("[%d/%d] %s", i, len(paper_files), paper_path.stem)
        try:
            result = process_paper(models, paper_path)
            success += 1 if result else 0
            skipped += 0 if result else 1
        except Exception as exc:
            log.error("Failed %s: %s", paper_path.stem, exc)
            errors += 1

    log.info("Done — %d processed, %d skipped, %d errors", success, skipped, errors)


if __name__ == "__main__":
    main()