"""
Compute embeddings and produce intra/inter-model similarity heatmaps.

Usage:
  python visualize_similarity.py                        # all result types
  python visualize_similarity.py underlying_hypotheses  # one type
  python visualize_similarity.py new_hypotheses

Pipeline per result type:
  1. results/<type>/*.json  →  embed with text-embedding-3-small
  2. Save embeddings to embeddings/<type>/*.json
  3. Intra-model heatmap  →  plots/<type>/intra_model.png
  4. Inter-model heatmap  →  plots/<type>/inter_model.png
"""

import json
import logging
import sys
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_RESULT_TYPES = ["underlying_hypotheses", "new_hypotheses"]

EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_BINS = np.arange(0.0, 1.01, 0.1)

# ---------------------------------------------------------------------------
# Paths helper
# ---------------------------------------------------------------------------

def get_paths(result_type: str) -> tuple[Path, Path, Path]:
    results_dir    = Path(f"results/{result_type}")
    embeddings_dir = Path(f"embeddings/{result_type}")
    heatmap_dir    = Path(f"plots/{result_type}")
    return results_dir, embeddings_dir, heatmap_dir

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
# Embedding
# ---------------------------------------------------------------------------

def build_embeddings(model: OpenAIEmbeddings, results_dir: Path, embeddings_dir: Path) -> None:
    """Embed all hypothesis outputs and cache per-paper to embeddings_dir."""
    for path in sorted(results_dir.glob("*.json")):
        out_path = embeddings_dir / path.name
        if out_path.exists():
            log.info("[skip] embeddings exist: %s", path.stem)
            continue

        with open(path) as f:
            data: dict[str, list[str]] = json.load(f)

        embeddings_map: dict[str, list] = {}
        for model_id, responses in data.items():
            log.info("  Embedding %s / %s (%d responses)", path.stem, model_id, len(responses))
            embeddings_map[model_id] = model.embed_documents(responses)

        with open(out_path, "w") as f:
            json.dump(embeddings_map, f)

    log.info("Embeddings ready in %s", embeddings_dir)

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

    models = sorted(all_data[0].keys())
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

    matrix = np.divide(accum, counts, where=counts > 0)
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

def build_embeddings_for(embedding_model: OpenAIEmbeddings, result_type: str) -> str | None:
    """Embed one result type. Returns result_type on success, None if skipped."""
    results_dir, embeddings_dir, _ = get_paths(result_type)
    if not results_dir.exists() or not any(results_dir.glob("*.json")):
        log.warning("No results found in %s — skipping", results_dir)
        return None
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== [%s] Building embeddings ===", result_type)
    build_embeddings(embedding_model, results_dir, embeddings_dir)
    return result_type


def plot_for(result_type: str) -> None:
    _, embeddings_dir, heatmap_dir = get_paths(result_type)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== [%s] Intra-model heatmap ===", result_type)
    intra_sims = compute_intra_model_similarities(embeddings_dir)
    plot_intra_model_heatmap(intra_sims, heatmap_dir / "intra_model.png")

    log.info("=== [%s] Inter-model heatmap ===", result_type)
    matrix, models = compute_inter_model_matrix(embeddings_dir)
    plot_inter_model_heatmap(matrix, models, heatmap_dir / "inter_model.png")


def main() -> None:
    load_dotenv()

    result_types = sys.argv[1:] if len(sys.argv) > 1 else ALL_RESULT_TYPES
    result_types = [rt for rt in result_types if rt in ALL_RESULT_TYPES or
                    (log.error("Unknown result type '%s'. Choose from: %s", rt, ALL_RESULT_TYPES) or False)]

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Step 1 — embed all result types in parallel (I/O bound API calls)
    completed = []
    with ThreadPoolExecutor(max_workers=len(result_types)) as executor:
        futures = {executor.submit(build_embeddings_for, embedding_model, rt): rt for rt in result_types}
        for future in as_completed(futures):
            rt = futures[future]
            exc = future.exception()
            if exc:
                log.error("Embedding failed for %s: %s", rt, exc)
            elif future.result():
                completed.append(future.result())

    # Step 2 — plot sequentially (fast, avoids matplotlib thread-safety issues)
    for rt in completed:
        plot_for(rt)

    log.info("All done.")


if __name__ == "__main__":
    main()
