"""
Shared engine for the hypothesis-generation scripts.

Both generate_hypotheses.py (experiment summaries -> underlying hypotheses)
and generate_new_hypotheses.py (full paper text -> novel hypotheses) do the
same underlying work: for each input file, for each model, make sure there
are N_SAMPLES *valid* generations, dispatching all needed calls concurrently
and writing incrementally as each model's set completes. The only things
that actually differ between the two are the input directory, the text
extracted from each file, the prompt, and (for the summaries script) a
paper-ID skip list and an upstream-error check.

This module holds the one shared implementation; each script supplies just
those differences. Fixing a bug here (as happened separately, several times,
in each script's own copy of this logic) now only needs doing once.
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from model_utils import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def is_valid(sample) -> bool:
    """A sample is invalid if it's blank, one of the literal 'ERROR: ...'
    strings written when a call raises an exception, a lingering None
    placeholder, or garbled output (>= 100 embedded newlines).

    The 100-newline threshold is backed by real data, not a guess: across
    a full audit of this project's output, legitimate samples topped out
    at 4 newlines (models adding an unrequested "**Hypothesis:** ... this
    is grounded in..." structure despite being told not to -- verbose,
    but still one coherent hypothesis). Confirmed-garbled samples (mostly
    moonshotai/kimi-k2.7-code hitting its reasoning-token budget mid-
    generation and degenerating into repeated fragments like "important."
    or "to to to to...") started at 241 newlines. 100 sits in the middle
    of that gap with wide margin on both sides -- not a fine-grained
    judgment call, a threshold picked to land cleanly between two
    observed, well-separated clusters."""
    if sample is None:
        return False
    text = str(sample)
    if text == "" or text.startswith("ERROR:"):
        return False
    if text.count("\n") >= 100:
        return False
    return True


def load_skip_ids(path: Path) -> set[str]:
    """IDs to skip entirely, one per line (blank lines ignored). Matches
    the input file's stem, e.g. 'journal.pbio.3003761' -- not the filename
    with its .json extension. Returns an empty set if the file doesn't
    exist, so this is safe to call unconditionally even for scripts that
    don't use a skip list."""
    if not path.exists():
        return set()
    ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if ids:
        log.info("Loaded %d ID(s) to skip from %s", len(ids), path)
    return ids


def process_file(
    models: dict[str, BaseModel],
    file_path: Path,
    output_dir: Path,
    extract_text: Callable[[dict], str],
    system_prompt: str,
    user_instruction: str,
    skip_ids: frozenset[str] = frozenset(),
    should_skip_content: Optional[Callable[[dict], Optional[str]]] = None,
    n_samples: int = 10,
    max_workers: int = 24,
) -> dict | None:
    """
    Process one input file, topping up every model to n_samples valid
    generations. Returns the result dict, or None if the file was skipped.

    extract_text(data) -> str            : pulls the text to send the model
    should_skip_content(data) -> str|None: return a skip reason to bail
                                            before calling any model (e.g. an
                                            upstream error already recorded
                                            in this file), or None to proceed
    """
    file_id = file_path.stem

    if file_id in skip_ids:
        log.info("Skipping %s — in skip list", file_id)
        return None

    output_path = output_dir / f"{file_id}.json"

    with open(file_path) as f:
        data = json.load(f)

    if should_skip_content is not None:
        reason = should_skip_content(data)
        if reason:
            log.warning("Skipping %s — %s", file_id, reason)
            return None

    text = extract_text(data)

    result: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)

    # result_lock guards all reads/writes to `result` and the output file,
    # since multiple worker threads fill in different slots (and sometimes
    # different models) at the same time.
    result_lock = threading.Lock()

    def save():
        with result_lock:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

    # For each model, keep whatever EXISTING samples are still valid and
    # only queue up work for the shortfall -- not a full fresh N_SAMPLES
    # regeneration. This means an "ERROR: rate limited" from a prior run
    # gets silently retried on the next run without needing an external
    # cleanup pass first.
    tasks: list[tuple[str, BaseModel, int]] = []
    for model_id, model in models.items():
        existing = result.get(model_id, [])
        valid = [s for s in existing if is_valid(s)]
        n_needed = n_samples - len(valid)

        if n_needed <= 0:
            log.info("  [skip] %s already has %d valid samples", model_id, len(valid))
            result[model_id] = valid[:n_samples]
            continue

        log.info("  %s has %d valid samples, topping up %d more...", model_id, len(valid), n_needed)
        result[model_id] = valid + [None] * n_needed
        start_index = len(valid)
        for offset in range(n_needed):
            tasks.append((model_id, model, start_index + offset))

    if not tasks:
        return result

    log.info("  Dispatching %d calls across up to %d workers...", len(tasks), max_workers)

    def run_one(model_id: str, model: BaseModel, i: int) -> str:
        try:
            reply = model.generate(system_prompt, text, user_instruction)
            log.info("  %s [slot %d] %s", model_id, i + 1, reply[:90])
        except Exception as exc:
            reply = f"ERROR: {exc}"
            log.error("  %s [slot %d] ERROR: %s", model_id, i + 1, exc)

        with result_lock:
            result[model_id][i] = reply

        return model_id

    completed_models: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, model_id, model, i): model_id
            for model_id, model, i in tasks
        }
        for future in as_completed(futures):
            model_id = futures[future]
            future.result()  # re-raises if run_one itself blew up unexpectedly

            with result_lock:
                model_done = all(x is not None for x in result[model_id])
            if model_done and model_id not in completed_models:
                completed_models.add(model_id)
                save()

    return result


def run_batch(
    models: dict[str, BaseModel],
    input_dir: Path,
    output_dir: Path,
    extract_text: Callable[[dict], str],
    system_prompt: str,
    user_instruction: str,
    skip_ids: frozenset[str] = frozenset(),
    should_skip_content: Optional[Callable[[dict], Optional[str]]] = None,
    n_samples: int = 10,
    max_workers: int = 24,
) -> None:
    """Drives process_file() over every *.json file in input_dir, with
    top-level progress logging and per-file error isolation (one bad file
    doesn't stop the batch)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    log.info(
        "Found %d files | %d models | %d samples → up to ~%d API calls",
        len(files), len(models), n_samples,
        len(files) * len(models) * n_samples,
    )

    success = skipped = errors = 0
    for i, file_path in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), file_path.stem)
        try:
            result = process_file(
                models, file_path, output_dir, extract_text,
                system_prompt, user_instruction,
                skip_ids=skip_ids, should_skip_content=should_skip_content,
                n_samples=n_samples, max_workers=max_workers,
            )
            success += 1 if result else 0
            skipped += 0 if result else 1
        except Exception as exc:
            log.error("Failed %s: %s", file_path.stem, exc)
            errors += 1

    log.info("Done — %d processed, %d skipped, %d errors", success, skipped, errors)