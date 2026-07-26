"""
Generate underlying hypotheses from experiment summaries.

Input:  data/experiments_summary/<paper_id>.json
Output: results/underlying_hypotheses/<paper_id>.json
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
    Matches summary_path.stem, e.g. 'journal.pbio.3003761' -- not the
    filename with its .json extension."""
    if not path.exists():
        return set()
    ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if ids:
        log.info("Loaded %d paper ID(s) to skip from %s", len(ids), path)
    return ids


SKIP_IDS = _load_skip_ids(SKIP_LIST_PATH)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_hypothesis(model: BaseModel, experiments_summary: str) -> str:
    return model.generate(SYSTEM_PROMPT, experiments_summary, USER_INSTRUCTION)


def sample_model(model: BaseModel, model_id: str, experiments_summary: str, n_needed: int) -> list[str]:
    """Generate exactly n_needed new samples (the shortfall, not a full fresh batch)."""
    samples = []
    for i in range(n_needed):
        try:
            text = get_hypothesis(model, experiments_summary)
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

def process_summary(models: dict[str, BaseModel], summary_path: Path) -> dict | None:
    paper_id = summary_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    if paper_id in SKIP_IDS:
        log.info("Skipping %s — listed in %s", paper_id, SKIP_LIST_PATH)
        return None

    with open(summary_path) as f:
        summary_data = json.load(f)

    if "error" in summary_data:
        log.warning("Skipping %s — upstream error: %s", paper_id, summary_data["error"])
        return None

    experiments_summary = summary_data["experiments_summary"]

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
        new_samples = sample_model(model, model_id, experiments_summary, n_needed)
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

    summary_files = sorted(SUMMARY_DIR.glob("*.json"))
    log.info(
        "Found %d summaries | %d models | %d samples → up to ~%d API calls",
        len(summary_files), len(models), N_SAMPLES,
        len(summary_files) * len(models) * N_SAMPLES,
    )

    success = skipped = errors = 0
    for i, summary_path in enumerate(summary_files, 1):
        log.info("[%d/%d] %s", i, len(summary_files), summary_path.stem)
        try:
            result = process_summary(models, summary_path)
            success += 1 if result else 0
            skipped += 0 if result else 1
        except Exception as exc:
            log.error("Failed %s: %s", summary_path.stem, exc)
            errors += 1

    log.info("Done — %d processed, %d skipped, %d errors", success, skipped, errors)


if __name__ == "__main__":
    main()