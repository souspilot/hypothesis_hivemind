"""
Generate hypotheses from experiment summaries using multiple LLMs.

Uses LangChain's init_chat_model for a unified interface across all providers.

Output format per paper (results/underlying_hypotheses/<paper_id>.json):
  { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 10
SLEEP_BETWEEN_CALLS = 0.2  # seconds

# Model IDs in "provider:model" format understood by init_chat_model
MODELS = [
    # Claude
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-sonnet-4-5",
    "anthropic:claude-haiku-4-5",
    # GPT-5 family
    "openai:gpt-5-mini",
    "openai:gpt-5",
]

SUMMARY_DIR = Path("data/experiments_summary")
OUTPUT_DIR = Path("results/underlying_hypotheses")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a scientific reasoning assistant. Given a description of the "
        "experiments and methods from a research paper, infer the underlying "
        "hypothesis being tested — the core scientific claim the experiments were "
        "designed to validate. A hypothesis is a specific, testable, and falsifiable "
        "prediction about the relationship between variables. "
        "Output ONLY the hypothesis as a single declarative sentence. "
        "Do not include preamble, explanation, or any other text.",
    ),
    (
        "human",
        "Generate a single testable hypothesis based on the following experiment "
        "description. Express it as one declarative sentence "
        "(e.g. 'If X, then Y because Z'):\n\n{experiments_summary}",
    ),
])

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

def get_hypothesis(model, experiments_summary: str) -> str:
    chain = PROMPT | model
    response = chain.invoke({"experiments_summary": experiments_summary})
    return response.content.strip()


def sample_model(model, model_id: str, experiments_summary: str) -> list[str]:
    samples = []
    for i in range(N_SAMPLES):
        try:
            text = get_hypothesis(model, experiments_summary)
            samples.append(text)
            log.info("    [%d/%d] %s", i + 1, N_SAMPLES, text[:90])
        except Exception as exc:
            log.error("    [%d/%d] ERROR: %s", i + 1, N_SAMPLES, exc)
            samples.append(f"ERROR: {exc}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return samples

# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_summary(models: dict, summary_path: Path) -> dict | None:
    paper_id = summary_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    with open(summary_path) as f:
        summary_data = json.load(f)

    if "error" in summary_data:
        log.warning("Skipping %s — upstream error: %s", paper_id, summary_data["error"])
        return None

    experiments_summary = summary_data["experiments_summary"]

    # Resume: load existing so we skip already-completed models
    result: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)

    for model_id, model in models.items():
        existing = result.get(model_id, [])
        if len(existing) >= N_SAMPLES:
            log.info("  [skip] %s already has %d samples", model_id, len(existing))
            continue

        log.info("  %s (%d samples)...", model_id, N_SAMPLES)
        result[model_id] = sample_model(model, model_id, experiments_summary)

        # Checkpoint after every model in case of interruption
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    # Build all models once — init_chat_model picks the right provider automatically
    models = {
        model_id: init_chat_model(model_id, temperature=1)
        for model_id in MODELS
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(SUMMARY_DIR.glob("*.json"))
    log.info(
        "Found %d summaries | %d models | %d samples → ~%d API calls",
        len(summary_files), len(models), N_SAMPLES,
        len(summary_files) * len(models) * N_SAMPLES,
    )

    success = skipped = errors = 0
    for i, summary_path in enumerate(summary_files, 1):
        log.info("[%d/%d] %s", i, len(summary_files), summary_path.stem)
        try:
            result = process_summary(models, summary_path)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            log.error("Failed %s: %s", summary_path.stem, exc)
            errors += 1

    log.info("Done — %d processed, %d skipped, %d errors", success, skipped, errors)


if __name__ == "__main__":
    main()
