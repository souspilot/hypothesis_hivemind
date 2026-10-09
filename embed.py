"""Embed valid hypotheses and cache vectors by paper and model.

Reuse entries only when sample counts and content fingerprints match.
Skip-listed papers are excluded and their cached files are removed.

Usage:
  python embed.py
  python embed.py --dataset plos --task novel
  python embed.py --check"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import DATASETS, EMBEDDING_MODEL, OPENROUTER_BASE_URL, TASKS, Dataset, Task, parse_dataset_args
from hypothesis_engine import is_valid
from embedding_cache import fingerprint, fingerprints_path, load_fingerprints

log = logging.getLogger(__name__)
MAX_WORKERS = 16


def plan(dataset: Dataset, task: Task) -> dict[Path, dict[str, list[str]]]:
    """{embeddings_path: {model_key: valid_samples}} for every entry whose
    cache is missing or out of date."""
    skip = dataset.skip_ids()
    results_dir, emb_dir = dataset.results_dir(task), dataset.embeddings_dir(task)
    todo: dict[Path, dict[str, list[str]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        out_path = emb_dir / path.name
        if path.stem in skip:
            continue
        results = json.loads(path.read_text())
        cached = json.loads(out_path.read_text()) if out_path.exists() else {}
        hashes = load_fingerprints(out_path)
        for key, samples in results.items():
            valid = [s for s in samples if is_valid(s)]
            if key not in cached or len(cached[key]) != len(valid) or hashes.get(key) != fingerprint(valid):
                todo.setdefault(out_path, {})[key] = valid
    return todo


def run(dataset: Dataset, task: Task, embedder) -> None:
    for pid in dataset.skip_ids():
        stale = dataset.embeddings_dir(task) / f"{pid}.json"
        if stale.exists():
            stale.unlink()
            log.info("Removed cached embeddings for skip-listed paper %s", pid)
        fingerprints_path(stale).unlink(missing_ok=True)
    todo = plan(dataset, task)
    n = sum(len(v) for v in todo.values())
    log.info("[%s / %s] %d stale or missing entries across %d papers", dataset.key, task.key, n, len(todo))
    if not todo:
        return
    dataset.embeddings_dir(task).mkdir(parents=True, exist_ok=True)

    def embed_paper(out_path: Path, entries: dict[str, list[str]]) -> None:
        cached = json.loads(out_path.read_text()) if out_path.exists() else {}
        hashes = load_fingerprints(out_path)
        for key, samples in entries.items():
            cached[key] = embedder.embed_documents(samples) if samples else []
            hashes[key] = fingerprint(samples)
            log.info("  %s / %s: %d vectors", out_path.stem, key, len(samples))
        out_path.write_text(json.dumps(cached))
        hash_path = fingerprints_path(out_path)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(json.dumps(hashes, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for f in [pool.submit(embed_paper, p, e) for p, e in todo.items()]:
            f.result()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="*", choices=list(DATASETS), help="default: all")
    parser.add_argument("--task", nargs="*", choices=list(TASKS), help="default: all")
    parser.add_argument("--check", action="store_true", help="only report what would be re-embedded")
    args = parser.parse_args()
    tasks = [TASKS[t] for t in args.task] if args.task else list(TASKS.values())
    pairs = [(d, t) for d in parse_dataset_args(args.dataset) for t in tasks]

    if args.check:
        for d, t in pairs:
            todo = plan(d, t)
            print(f"{d.key:7s} {t.key:8s} {sum(len(v) for v in todo.values()):4d} stale entries")
            for path, entries in todo.items():
                for key in entries:
                    print(f"          {path.stem} / {key}")
        return

    from dotenv import load_dotenv
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()
    # Request the configured embedding model through OpenRouter.
    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL,
    )
    for d, t in pairs:
        run(d, t, embedder)


if __name__ == "__main__":
    main()
