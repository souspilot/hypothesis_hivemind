"""
Generate novel hypotheses from full paper context.

Input:  data/train/<paper_id>.json
Output: results/new_hypotheses/<paper_id>.json
        { "<model_id>": ["hypothesis_1", ..., "hypothesis_N"], ... }
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from model_utils import BaseModel, build_all_models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 10

# How many (model, sample) calls run at once, across the whole paper. Every
# call is an independent network request -- there's no reason to wait for
# one to finish before starting the next, the way 20 people can each be on
# their own phone call simultaneously instead of queuing for one phone.
# 12 models x 10 samples = 120 possible calls per paper; MAX_WORKERS bounds
# how many of those are in flight at once. If you start seeing frequent
# "ERROR: ... 429 ..." (rate limit) entries in the output, lower this; if
# you see none and want more speed, raise it.
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
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper text extraction
# ---------------------------------------------------------------------------

def extract_paper_text(data: dict) -> str:
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
# Inference
# ---------------------------------------------------------------------------

def get_new_hypothesis(model: BaseModel, paper_text: str) -> str:
    return model.generate(SYSTEM_PROMPT, paper_text, USER_INSTRUCTION)

# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_paper(models: dict[str, BaseModel], paper_path: Path) -> dict:
    paper_id = paper_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    with open(paper_path) as f:
        paper_text = extract_paper_text(json.load(f))

    result: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)

    # result_lock guards all reads/writes to `result` and the output file
    # from here on, since multiple worker threads will be filling in
    # different slots (and sometimes different models) at the same time.
    result_lock = threading.Lock()

    def save():
        with result_lock:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

    # Figure out which (model_id, sample_index) slots still need work.
    # Any model that needs regenerating gets a fresh N_SAMPLES-length list
    # of placeholders -- same "fully regenerate this model's set" behavior
    # as before, just filled in concurrently now instead of one at a time.
    tasks: list[tuple[str, BaseModel, int]] = []
    for model_id, model in models.items():
        existing = result.get(model_id, [])
        if len(existing) >= N_SAMPLES:
            log.info("  [skip] %s already has %d samples", model_id, N_SAMPLES)
            continue
        result[model_id] = [None] * N_SAMPLES
        for i in range(N_SAMPLES):
            tasks.append((model_id, model, i))

    if not tasks:
        return result

    log.info("  Dispatching %d calls across up to %d workers...", len(tasks), MAX_WORKERS)

    def run_one(model_id: str, model: BaseModel, i: int) -> str:
        try:
            text = get_new_hypothesis(model, paper_text)
            log.info("  %s [%d/%d] %s", model_id, i + 1, N_SAMPLES, text[:90])
        except Exception as exc:
            text = f"ERROR: {exc}"
            log.error("  %s [%d/%d] ERROR: %s", model_id, i + 1, N_SAMPLES, exc)

        with result_lock:
            result[model_id][i] = text

        return model_id

    completed_models: set[str] = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_one, model_id, model, i): model_id
            for model_id, model, i in tasks
        }
        for future in as_completed(futures):
            model_id = futures[future]
            future.result()  # re-raises if run_one itself blew up unexpectedly

            # Save once a model's full set of N_SAMPLES is filled in, not
            # after every single call -- keeps the same per-model
            # resumability as before without writing to disk 120 times.
            with result_lock:
                model_done = all(x is not None for x in result[model_id])
            if model_done and model_id not in completed_models:
                completed_models.add(model_id)
                save()

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    models = build_all_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_files = sorted(TRAIN_DIR.glob("*.json"))
    log.info(
        "Found %d papers | %d models | %d samples → ~%d API calls",
        len(paper_files), len(models), N_SAMPLES,
        len(paper_files) * len(models) * N_SAMPLES,
    )

    success = errors = 0
    for i, paper_path in enumerate(paper_files, 1):
        log.info("[%d/%d] %s", i, len(paper_files), paper_path.stem)
        try:
            process_paper(models, paper_path)
            success += 1
        except Exception as exc:
            log.error("Failed %s: %s", paper_path.stem, exc)
            errors += 1

    log.info("Done — %d processed, %d errors", success, errors)


if __name__ == "__main__":
    main()