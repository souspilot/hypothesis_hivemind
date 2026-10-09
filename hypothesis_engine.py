"""Shared hypothesis-generation engine.

Preserve valid stored responses and generate missing samples concurrently.
The calling scripts supply input extraction, prompts, and paper exclusions."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from config import stored_key

if TYPE_CHECKING:  # analysis code imports is_valid without the API SDKs installed
    from model_utils import BaseModel

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """Called from each script's main(), not at import, so that importing
    is_valid (e.g. from analysis.py) doesn't turn on INFO logging globally."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def is_valid(sample) -> bool:
    """Reject empty strings, None, ERROR-prefixed responses, and garbled output.

    The garbling threshold is 100 newlines. In the original output audit,
    legitimate samples had at most four newlines; confirmed garbled samples
    had at least 241."""
    if sample is None:
        return False
    text = str(sample)
    if text == "" or text.startswith("ERROR:"):
        return False
    if text.count("\n") >= 100:
        return False
    return True


def load_skip_ids(path: Path) -> set[str]:
    """Read paper IDs from a file, ignoring blank lines. Missing files yield no IDs."""
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
    should_skip_content(data) -> str|None: return a reason to skip
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

    # Workers share the result dictionary and output file.
    result_lock = threading.Lock()

    def save():
        with result_lock:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

    # Preserve valid responses and retry only the missing samples.
    tasks: list[tuple[str, BaseModel, int]] = []
    for slug, model in models.items():
        # Preserve legacy model keys from the original direct-API runs.
        model_id = stored_key(result, slug)
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
