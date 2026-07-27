"""
Compute embeddings and produce intra/inter-model similarity heatmaps.

Two data sources exist side by side -- data/ (native-format papers) and
data2/ (XML-converted papers) -- each with its own mirrored results{,2}/
tree, containing both underlying_hypotheses and new_hypotheses. Source and
result type are independent choices; both default to "all" if omitted.

Usage:
  python visualize_similarity.py                                    # everything: both sources x both result types
  python visualize_similarity.py --source 1                         # data/ only, both result types
  python visualize_similarity.py underlying_hypotheses               # both sources, one result type
  python visualize_similarity.py --source 2 new_hypotheses           # one source, one result type

Pipeline per (source, result type) pair:
  1. results{suffix}/<type>/*.json  →  embed with text-embedding-3-small
  2. Save embeddings to embeddings{suffix}/<type>/*.json
  3. Intra-model heatmap  →  plots{suffix}/<type>/intra_model.png
  4. Inter-model heatmap  →  plots{suffix}/<type>/inter_model.png
"""

import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from hypothesis_engine import is_valid, load_skip_ids

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_RESULT_TYPES = ["underlying_hypotheses", "new_hypotheses"]

# Maps --source values to the directory suffix: data/results have no
# suffix, data2/results2 have "2". Source and result type are independent
# dimensions -- every combination of the two is a valid, real directory
# (results/new_hypotheses, results/underlying_hypotheses,
# results2/new_hypotheses, results2/underlying_hypotheses all exist).
SOURCE_SUFFIXES = {"1": "", "2": "2"}

# Routed through OpenRouter's embeddings endpoint (same OPENAI_API-compatible
# shape as model_utils.py's OpenRouterModel, just a different dedicated
# endpoint: /embeddings instead of /chat/completions). This is still
# literally OpenAI's text-embedding-3-small on the backend -- OpenRouter is
# just the single billing/key funnel -- so vectors are numerically identical
# to calling OpenAI directly. Safe to mix with any embeddings/*.json files
# already cached from a prior direct-OpenAI run.
#
# Model slug needs the provider prefix, same convention as the chat models
# in model_utils.py ("anthropic/claude-sonnet-4.6", "openai/gpt-5", etc.) --
# bare "text-embedding-3-small" will 400 on OpenRouter's endpoint.
EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# How many (paper, model) embedding calls run at once, per (source, result
# type) pair. Embedding calls are individually fast, but sequential across
# ~50 papers x ~12 models still adds up to hundreds of round-trips -- same
# reasoning as MAX_WORKERS in generate_new_hypotheses.py.
EMBEDDING_MAX_WORKERS = 16

# Paper IDs to never embed, matching the generation scripts' skip list --
# these files may still exist on disk from before the skip list existed,
# with leftover ERROR: entries mixed into otherwise-valid samples. See
# plos_skip.txt for the current list. Applies to both data sources, since
# it's about problematic paper CONTENT, not which pipeline produced it.
SKIP_LIST_PATH = Path("plos_skip.txt")

SIMILARITY_BINS = np.arange(0.0, 1.01, 0.1)

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
# Paths helper
# ---------------------------------------------------------------------------

def get_paths(result_type: str, suffix: str) -> tuple[Path, Path, Path]:
    results_dir    = Path(f"results{suffix}/{result_type}")
    embeddings_dir = Path(f"embeddings{suffix}/{result_type}")
    heatmap_dir    = Path(f"plots{suffix}/{result_type}")
    return results_dir, embeddings_dir, heatmap_dir

# ---------------------------------------------------------------------------
# Skip list / sample validity
#
# is_valid and load_skip_ids used to be reimplemented separately in this
# file (a third copy of logic that already existed in both generation
# scripts, before they were consolidated into hypothesis_engine.py). Now
# imported from there instead, so there's exactly one definition of "what
# counts as a usable sample" across the whole project.
# ---------------------------------------------------------------------------

SKIP_IDS = load_skip_ids(SKIP_LIST_PATH)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embeddings(model: OpenAIEmbeddings, results_dir: Path, embeddings_dir: Path) -> None:
    """Embed all hypothesis outputs and cache per-paper to embeddings_dir.

    Cache freshness is checked per MODEL, not per file. Previously this
    skipped a paper entirely if its embeddings file already existed --
    which meant a paper embedded back when the model roster had 6 entries
    stayed frozen at those 6 forever, even after generate_new_hypotheses.py
    later added results for 6 more models to the same paper. The newer
    models would then be silently absent from every heatmap with no
    error, since nothing ever re-checked whether the cache still matched
    the current model roster.

    Known limitation: this only detects a model key being entirely ABSENT
    from the cache. It does not detect a model's sample count changing
    (e.g. topped up from 9 valid samples to 10) -- that would need
    comparing cached vector counts against current valid-sample counts,
    which isn't implemented here. In practice this hasn't come up yet,
    but if you start topping up samples for a model that's already been
    through this pipeline once, its embeddings won't reflect the new
    samples until you delete that entry from the cache file by hand.
    """
    pending_paths = []
    for path in sorted(results_dir.glob("*.json")):
        if path.stem in SKIP_IDS:
            log.info("[skip] %s — listed in %s", path.stem, SKIP_LIST_PATH)
            continue
        pending_paths.append(path)

    if not pending_paths:
        log.info("Nothing to embed in %s", results_dir)
        return

    # results_by_paper accumulates each paper's {model_id: [vectors]} as
    # tasks complete -- seeded from whatever's already cached, so existing
    # embeddings are preserved rather than recomputed. paper_lock guards
    # it and the per-paper output-file write, since multiple worker
    # threads finish at arbitrary times and may belong to different
    # papers or the same one.
    results_by_paper: dict[Path, dict[str, list]] = {}
    pending_model_counts: dict[Path, int] = {}
    paper_changed: dict[Path, bool] = {}
    paper_lock = threading.Lock()

    tasks: list[tuple[Path, str, list[str]]] = []
    for path in pending_paths:
        with open(path) as f:
            data: dict[str, list[str]] = json.load(f)

        out_path = embeddings_dir / path.name
        cached: dict[str, list] = {}
        if out_path.exists():
            with open(out_path) as f:
                cached = json.load(f)

        results_by_paper[path] = dict(cached)  # start from what's already cached
        paper_changed[path] = False
        n_models_needing_call = 0

        for model_id, responses in data.items():
            if model_id in cached:
                continue  # already embedded in a prior run -- leave as-is

            valid_responses = [r for r in responses if is_valid(r)]
            n_dropped = len(responses) - len(valid_responses)
            if n_dropped:
                log.warning(
                    "  %s / %s: dropping %d blank/ERROR entr%s before embedding",
                    path.stem, model_id, n_dropped, "y" if n_dropped == 1 else "ies",
                )

            if not valid_responses:
                # No API call needed -- record the empty result directly.
                results_by_paper[path][model_id] = []
                paper_changed[path] = True
                continue

            n_models_needing_call += 1
            tasks.append((path, model_id, valid_responses))

        pending_model_counts[path] = n_models_needing_call
        if n_models_needing_call == 0 and paper_changed[path]:
            # New empty-list model(s) recorded but nothing needed an API
            # call -- save now rather than waiting for a completion event
            # that will never come.
            _save_paper_embeddings(embeddings_dir, path, results_by_paper[path])

    if not tasks:
        log.info("Embeddings already up to date in %s (no new models to embed)", embeddings_dir)
        return

    log.info("  Dispatching %d embedding calls across up to %d workers...", len(tasks), EMBEDDING_MAX_WORKERS)

    def run_one(path: Path, model_id: str, valid_responses: list[str]):
        log.info("  Embedding %s / %s (%d responses)", path.stem, model_id, len(valid_responses))
        vectors = model.embed_documents(valid_responses)
        with paper_lock:
            results_by_paper[path][model_id] = vectors
            pending_model_counts[path] -= 1
            done = pending_model_counts[path] == 0
        if done:
            _save_paper_embeddings(embeddings_dir, path, results_by_paper[path])

    with ThreadPoolExecutor(max_workers=EMBEDDING_MAX_WORKERS) as executor:
        futures = [executor.submit(run_one, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()  # re-raise any embedding-call failure immediately

    log.info("Embeddings ready in %s", embeddings_dir)


def _save_paper_embeddings(embeddings_dir: Path, path: Path, embeddings_map: dict[str, list]) -> None:
    out_path = embeddings_dir / path.name
    with open(out_path, "w") as f:
        json.dump(embeddings_map, f)

# ---------------------------------------------------------------------------
# Intra-model similarity
# ---------------------------------------------------------------------------

def average_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine similarity over all unique pairs in (N, D) embedding matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    idx = np.triu_indices(len(sim_matrix), k=1)
    return float(sim_matrix[idx].mean())


def compute_intra_model_similarities(embeddings_dir: Path) -> dict[str, list[float]]:
    """Return {model_id: [avg_pairwise_sim per paper]}."""
    model_to_sims: dict[str, list[float]] = defaultdict(list)
    for path in sorted(embeddings_dir.glob("*.json")):
        with open(path) as f:
            data: dict[str, list] = json.load(f)
        for model_id, emb_list in data.items():
            if len(emb_list) < 2:
                continue
            sim = average_pairwise_cosine(np.array(emb_list, dtype=np.float32))
            model_to_sims[model_id].append(sim)
    return model_to_sims


def plot_intra_model_heatmap(
    model_to_sims: dict[str, list[float]],
    save_path: Path,
) -> None:
    models = sorted(model_to_sims.keys())
    if not models:
        log.warning("No models with >=2 valid samples in any paper -- skipping intra-model heatmap (%s)", save_path)
        return

    per_model = {}
    for model in models:
        hist, _ = np.histogram(model_to_sims[model], bins=SIMILARITY_BINS)
        per_model[model] = hist / hist.sum() * 100

    # Add Average column
    avg = np.mean(list(per_model.values()), axis=0)
    col_labels = ["Average"] + models
    matrix = np.array([avg] + [per_model[m] for m in models])  # (n_cols, n_bins)

    bin_labels = [
        f"{SIMILARITY_BINS[i]:.1f}–{SIMILARITY_BINS[i+1]:.1f}"
        for i in range(len(SIMILARITY_BINS) - 1)
    ]
    # Reverse so 0.9-1.0 is at the top
    bin_labels = bin_labels[::-1]
    matrix_plot = matrix.T[::-1]  # (n_bins, n_cols)

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(col_labels)), 6))
    sns.heatmap(
        matrix_plot,
        xticklabels=col_labels,
        yticklabels=bin_labels,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "% of papers"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Similarity Score Ranges")
    ax.set_title("Intra-model Repetition Across Papers")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Saved intra-model heatmap → %s", save_path)

# ---------------------------------------------------------------------------
# Inter-model similarity
# ---------------------------------------------------------------------------

def cross_model_avg_similarity(emb_i: np.ndarray, emb_j: np.ndarray) -> float:
    """Mean cosine similarity between every row in emb_i and every row in emb_j."""
    emb_i = emb_i / np.linalg.norm(emb_i, axis=1, keepdims=True)
    emb_j = emb_j / np.linalg.norm(emb_j, axis=1, keepdims=True)
    return float((emb_i @ emb_j.T).mean())


def compute_inter_model_matrix(embeddings_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Return (n_models × n_models similarity matrix, model name list)."""
    all_data: list[dict[str, list]] = []
    for path in sorted(embeddings_dir.glob("*.json")):
        with open(path) as f:
            all_data.append(json.load(f))

    if not all_data:
        raise ValueError(f"No embedding files found in {embeddings_dir} -- run the embedding step first.")

    # Union of model keys across ALL papers, not just the first file --
    # different papers' output JSON can have different model sets (the
    # MODELS list in model_utils.py has changed shape multiple times over
    # this project's life), so using only all_data[0].keys() would
    # silently drop any model missing from whichever paper happened to
    # sort first, with no warning that it was excluded.
    models: list[str] = sorted(set().union(*(d.keys() for d in all_data)))
    model_idx = {m: i for i, m in enumerate(models)}
    n = len(models)

    accum = np.zeros((n, n))
    counts = np.zeros((n, n))

    for paper in all_data:
        for m1 in models:
            for m2 in models:
                if m1 not in paper or m2 not in paper:
                    continue
                e1 = np.array(paper[m1], dtype=np.float32)
                e2 = np.array(paper[m2], dtype=np.float32)
                if len(e1) == 0 or len(e2) == 0:
                    continue
                i, j = model_idx[m1], model_idx[m2]
                accum[i, j] += cross_model_avg_similarity(e1, e2)
                counts[i, j] += 1

    # out=... matters here: without it, cells where counts==0 (model pairs
    # that never co-occurred in the same paper) are left as uninitialized
    # memory rather than a defined value -- numpy warns about exactly this.
    # NaN is the correct "no data" value; seaborn renders NaN cells as
    # blank rather than plotting garbage numbers.
    matrix = np.full((n, n), np.nan)
    np.divide(accum, counts, out=matrix, where=counts > 0)
    return matrix, models


def plot_inter_model_heatmap(
    matrix: np.ndarray,
    models: list[str],
    save_path: Path,
) -> None:
    # Show only lower triangle (symmetric matrix)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(models)), max(8, len(models))))
    sns.heatmap(
        matrix,
        xticklabels=models,
        yticklabels=models,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Avg cosine similarity"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Inter-model Average Cosine Similarity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Saved inter-model heatmap → %s", save_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_embeddings_for(embedding_model: OpenAIEmbeddings, result_type: str, suffix: str) -> tuple[str, str] | None:
    """Embed one (result_type, suffix) pair. Returns (result_type, suffix) on success, None if skipped."""
    results_dir, embeddings_dir, _ = get_paths(result_type, suffix)
    if not results_dir.exists() or not any(results_dir.glob("*.json")):
        log.warning("No results found in %s — skipping", results_dir)
        return None
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== [data%s/%s] Building embeddings ===", suffix, result_type)
    build_embeddings(embedding_model, results_dir, embeddings_dir)
    return (result_type, suffix)


def plot_for(result_type: str, suffix: str) -> None:
    _, embeddings_dir, heatmap_dir = get_paths(result_type, suffix)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== [data%s/%s] Intra-model heatmap ===", suffix, result_type)
    intra_sims = compute_intra_model_similarities(embeddings_dir)
    plot_intra_model_heatmap(intra_sims, heatmap_dir / "intra_model.png")

    log.info("=== [data%s/%s] Inter-model heatmap ===", suffix, result_type)
    matrix, models = compute_inter_model_matrix(embeddings_dir)
    plot_inter_model_heatmap(matrix, models, heatmap_dir / "inter_model.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_types", nargs="*",
        help=f"Which result type(s) to process, from {ALL_RESULT_TYPES}. Default: both.",
    )
    parser.add_argument(
        "--source", choices=sorted(SOURCE_SUFFIXES), default=None,
        help="1 = data/ (native format), 2 = data2/ (XML-converted). Default: both.",
    )
    args = parser.parse_args()

    # Validated manually rather than via argparse's `choices=` on this
    # positional: nargs="*" combined with choices AND a list-valued
    # default crashes argparse's own validation even when zero args are
    # given (it tries to check the whole default list as if it were one
    # choice value). Simpler and more reliable to just check it here.
    result_types = args.result_types if args.result_types else ALL_RESULT_TYPES
    invalid = [rt for rt in result_types if rt not in ALL_RESULT_TYPES]
    if invalid:
        parser.error(f"invalid result type(s) {invalid}; choose from {ALL_RESULT_TYPES}")

    suffixes = [SOURCE_SUFFIXES[args.source]] if args.source else list(SOURCE_SUFFIXES.values())

    pairs = [(rt, suffix) for rt in result_types for suffix in suffixes]

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    # Step 1 — embed all (result_type, source) pairs in parallel (I/O bound API calls)
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, len(pairs))) as executor:
        futures = {
            executor.submit(build_embeddings_for, embedding_model, rt, suffix): (rt, suffix)
            for rt, suffix in pairs
        }
        for future in as_completed(futures):
            rt, suffix = futures[future]
            exc = future.exception()
            if exc:
                log.error("Embedding failed for data%s/%s: %s", suffix, rt, exc)
            elif future.result():
                completed.append(future.result())

    # Step 2 — plot sequentially (fast, avoids matplotlib thread-safety issues)
    for rt, suffix in completed:
        plot_for(rt, suffix)

    log.info("All done.")


if __name__ == "__main__":
    main()