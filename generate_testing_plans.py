# DEPRECATED: NOT DOING THIS EXPERIMENT

"""
Generate a testing plan for each model's own hypotheses.

For each paper and each model, pairs each hypothesis response (from
results/new_hypotheses/) with a testing plan produced by the same model.
N_SAMPLES hypothesis responses → N_SAMPLES testing plans per model.

Input:  results/new_hypotheses/<paper_id>.json
Output: results/testing_plans/<paper_id>.json
        { "<model_id>": ["plan_for_sample_1", ..., "plan_for_sample_N"], ... }
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
SLEEP_BETWEEN_CALLS = 0.2

HYPOTHESES_DIR = Path("results/new_hypotheses")
OUTPUT_DIR     = Path("results/testing_plans")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert research scientist. Given a hypothesis, produce a concise "
    "but complete testing plan to validate it. "
    "Your plan MUST cover these five components in order:\n"
    "1. Experimental design — how the experiment is set up\n"
    "2. Controls — baselines or control conditions to compare against\n"
    "3. Methodology — step-by-step procedure and data collection approach\n"
    "4. Metrics — quantitative measures used to evaluate the hypothesis\n"
    "5. Expected outcome — what result would confirm or refute the hypothesis\n"
    "Be specific and actionable. Output ONLY the five-component plan with no preamble."
)

USER_INSTRUCTION = "Write a testing plan for the following hypothesis above."

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
# Inference
# ---------------------------------------------------------------------------

def get_testing_plan(model: BaseModel, hypothesis: str) -> str:
    return model.generate(SYSTEM_PROMPT, hypothesis, USER_INSTRUCTION)


def sample_model(model: BaseModel, model_id: str, hypothesis_samples: list[str]) -> list[str]:
    """Generate one testing plan per hypothesis sample from this model."""
    plans = []
    for i, hypothesis in enumerate(hypothesis_samples):
        try:
            plan = get_testing_plan(model, hypothesis)
            plans.append(plan)
            log.info("    [%d/%d] %s", i + 1, len(hypothesis_samples), plan[:90])
        except Exception as exc:
            log.error("    [%d/%d] ERROR: %s", i + 1, len(hypothesis_samples), exc)
            plans.append(f"ERROR: {exc}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return plans

# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_paper(models: dict[str, BaseModel], hypotheses_path: Path) -> dict | None:
    paper_id = hypotheses_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    with open(hypotheses_path) as f:
        hypotheses_data: dict[str, list[str]] = json.load(f)

    # Resume: load existing so we skip already-completed models
    result: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)

    for model_id, model in models.items():
        existing = result.get(model_id, [])
        hypothesis_samples = hypotheses_data.get(model_id, [])

        if not hypothesis_samples:
            log.warning("  [skip] %s — no hypotheses found", model_id)
            continue

        if len(existing) >= len(hypothesis_samples):
            log.info("  [skip] %s already has %d plans", model_id, len(existing))
            continue

        # Only generate plans for samples not yet processed
        remaining = hypothesis_samples[len(existing):]
        log.info("  %s (%d plans to generate)...", model_id, len(remaining))
        new_plans = sample_model(model, model_id, remaining)
        result[model_id] = existing + new_plans

        # Checkpoint after every model
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    models = build_all_models()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hypothesis_files = sorted(HYPOTHESES_DIR.glob("*.json"))
    log.info(
        "Found %d hypothesis files | %d models | %d samples → ~%d API calls",
        len(hypothesis_files), len(models), N_SAMPLES,
        len(hypothesis_files) * len(models) * N_SAMPLES,
    )

    success = skipped = errors = 0
    for i, hyp_path in enumerate(hypothesis_files, 1):
        log.info("[%d/%d] %s", i, len(hypothesis_files), hyp_path.stem)
        try:
            result = process_paper(models, hyp_path)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            log.error("Failed %s: %s", hyp_path.stem, exc)
            errors += 1

    log.info("Done — %d processed, %d skipped, %d errors", success, skipped, errors)


if __name__ == "__main__":
    main()
