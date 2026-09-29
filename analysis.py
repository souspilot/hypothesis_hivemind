"""
Similarity analysis for the hypothesis-hivemind experiment.

Pure computation: reads cached embeddings (never calls an API) and returns
arrays/dicts that figures.py and make_paper_assets.py turn into figures and
tables.

Definitions (all cosine similarities, all self-pairs excluded):
  intra-model      mean over pairs of distinct samples from the SAME model
                   for the same paper
  model-pair (a,b) mean over all sample pairs (one from a, one from b) for
                   the same paper, a != b
  intra-provider   model-pair similarity averaged over pairs of distinct
                   models from the same provider
  cross-provider   model-pair similarity averaged over pairs of models from
                   different providers
Each quantity is computed per paper, then averaged over papers (so every
paper carries equal weight); CIs bootstrap over papers.
"""

import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from config import MIN_PAPERS, MODELS, MODELS_BY_SLUG, N_SAMPLES, Dataset, Task, canonical_slug
from hypothesis_engine import is_valid

SLUGS = [m.slug for m in MODELS]
PROVIDER_OF = np.array([m.provider for m in MODELS])
SAME_PROVIDER = PROVIDER_OF[:, None] == PROVIDER_OF[None, :]
OFF_DIAG = ~np.eye(len(MODELS), dtype=bool)


@dataclass
class Corpus:
    """All embeddings for one (dataset, task), unit-normalised.

    vectors[p][i] is an (n_samples, dim) array for paper p, model SLUGS[i]
    (possibly with zero rows if the model has no valid samples there).
    """
    dataset: Dataset
    task: Task
    paper_ids: list[str]
    vectors: list[list[np.ndarray]]

    @cached_property
    def pair_matrix(self) -> np.ndarray:
        """(n_papers, n_models, n_models) mean similarity; diagonal is
        intra-model (self-pairs excluded). NaN where a model has too few
        samples on a paper."""
        n_p, n_m = len(self.paper_ids), len(SLUGS)
        out = np.full((n_p, n_m, n_m), np.nan)
        for p, per_model in enumerate(self.vectors):
            counts = np.array([len(v) for v in per_model])
            stacked = np.concatenate([v for v in per_model if len(v)])
            owner = np.repeat(np.arange(n_m), counts)
            sims = stacked @ stacked.T
            onehot = np.zeros((len(owner), n_m))
            onehot[np.arange(len(owner)), owner] = 1
            block_sums = onehot.T @ sims @ onehot
            # Subtract the self-similarity diagonal (each vector with itself
            # = 1.0); the previous version left it in, which inflated every
            # intra-model value to 0.1 + 0.9 * true value at 10 samples.
            n_pairs = np.outer(counts, counts).astype(float)
            np.fill_diagonal(block_sums, np.diag(block_sums) - counts)
            np.fill_diagonal(n_pairs, counts * (counts - 1))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[p] = np.where(n_pairs > 0, block_sums / n_pairs, np.nan)
        return out

    @property
    def intra_model(self) -> np.ndarray:
        """(n_papers, n_models) intra-model similarity."""
        return np.diagonal(self.pair_matrix, axis1=1, axis2=2)

    def model_matrix(self) -> np.ndarray:
        """(n_models, n_models) paper-averaged similarity (Fig. 1)."""
        return np.nanmean(self.pair_matrix, axis=0)

    def group_per_paper(self) -> dict[str, np.ndarray]:
        """Per-paper intra-model / intra-provider / cross-provider means."""
        pm = self.pair_matrix
        return {
            "intra_model": np.nanmean(self.intra_model, axis=1),
            "intra_provider": np.nanmean(pm[:, SAME_PROVIDER & OFF_DIAG], axis=1),
            "cross_provider": np.nanmean(pm[:, ~SAME_PROVIDER], axis=1),
        }

    def paper_level_distributions(self) -> dict[str, np.ndarray]:
        """Similarity at three levels of relatedness (appendix figure):
        same paper & same model, same paper & different models, and
        different papers (any models). Each entry is one value per paper
        (or per pair of papers for the last)."""
        pm = self.pair_matrix
        same_model = np.nanmean(self.intra_model, axis=1)
        cross_model = np.nanmean(pm[:, OFF_DIAG], axis=1)

        centroids = []  # mean vector per paper; mean cos(p,q) = c_p . c_q
        for per_model in self.vectors:
            allv = np.concatenate([v for v in per_model if len(v)])
            centroids.append(allv.mean(axis=0))
        c = np.stack(centroids)
        between = c @ c.T
        iu = np.triu_indices(len(c), k=1)
        return {
            "same_model": same_model,
            "cross_model": cross_model,
            "different_paper": between[iu],
        }

    def paper_matrix(self, p: int) -> np.ndarray:
        """(n_models * N_SAMPLES, dim) embeddings for paper p, grouped by model."""
        return np.concatenate(self.vectors[p])


# ---------------------------------------------------------------------------
# Diversity (Fig. 4): Vendi score and PERMANOVA
#
# Both need full coverage (N_SAMPLES per model), which load_corpus guarantees.
# ---------------------------------------------------------------------------

def vendi_score(x: np.ndarray) -> float:
    """Effective number of distinct items among the rows of x (unit vectors),
    with the cosine kernel: exp(Shannon entropy of the eigenvalues of K/n).
    Friedman & Dieng, "The Vendi Score", TMLR 2023."""
    eig = np.linalg.eigvalsh(x @ x.T / len(x))
    eig = eig[eig > 1e-12]
    return float(np.exp(-(eig * np.log(eig)).sum()))


def permanova_r2(x: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variation among rows of x explained by `groups`:
    between-group / total sum of squares on Euclidean distances (Anderson 2001)."""
    total = ((x - x.mean(axis=0)) ** 2).sum()
    within = sum(((x[groups == g] - x[groups == g].mean(axis=0)) ** 2).sum() for g in np.unique(groups))
    return float(1 - within / total)


def diversity_summary(corpus: "Corpus", n_draws: int = 200, n_unrelated: int = 1000,
                      n_perm: int = 199, seed: int = 0) -> dict:
    """Fig. 4 and its numbers, for one (dataset, task). Every Vendi score is
    over exactly N_SAMPLES hypotheses, so set size cannot drive a difference.

    one[p]:  mean Vendi score of each model's own N_SAMPLES hypotheses for paper p.
    ten[p]:  mean Vendi score of N_SAMPLES hypotheses for paper p, one each from
             N_SAMPLES randomly chosen models (n_draws draws).
    unrelated: Vendi scores of N_SAMPLES hypotheses about N_SAMPLES different
             papers (random model each); a scale reference only.
    r2_model / r2_provider: PERMANOVA R^2 of model / provider identity per paper;
             p_max is the largest per-paper permutation p-value.
    """
    rng = np.random.default_rng(seed)
    n_m, m = len(SLUGS), N_SAMPLES
    models = np.repeat(np.arange(n_m), m)
    providers = np.repeat(PROVIDER_OF, m)

    one, ten, r2_model, r2_provider, p_values = [], [], [], [], []
    for p, per_model in enumerate(corpus.vectors):
        one.append(np.mean([vendi_score(v) for v in per_model]))
        ten.append(np.mean([
            vendi_score(np.stack([per_model[mi][rng.integers(m)]
                                  for mi in rng.choice(n_m, m, replace=False)]))
            for _ in range(n_draws)]))
        x = corpus.paper_matrix(p)
        r2 = permanova_r2(x, models)
        null = [permanova_r2(x, rng.permutation(models)) for _ in range(n_perm)]
        p_values.append((1 + sum(v >= r2 for v in null)) / (n_perm + 1))
        r2_model.append(r2)
        r2_provider.append(permanova_r2(x, providers))

    unrelated = np.array([
        vendi_score(np.stack([corpus.vectors[q][rng.integers(n_m)][rng.integers(m)]
                              for q in rng.choice(len(corpus.vectors), m, replace=False)]))
        for _ in range(n_unrelated)])
    one, ten = np.array(one), np.array(ten)
    return {
        "one": one,
        "ten": ten,
        "unrelated": unrelated,
        "one_mean": bootstrap_ci(one, seed=seed),
        "ten_mean": bootstrap_ci(ten, seed=seed),
        "unrelated_mean": float(unrelated.mean()),
        "share_up": float(np.mean(ten > one)),
        "r2_model": bootstrap_ci(np.array(r2_model), seed=seed),
        "r2_provider": bootstrap_ci(np.array(r2_provider), seed=seed),
        "p_max": float(max(p_values)),
        "n_papers": len(corpus.paper_ids),
    }

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class DataIntegrityError(RuntimeError):
    pass


def load_corpus(dataset: Dataset, task: Task) -> tuple[Corpus, dict[str, list[str]]]:
    """Load cached embeddings for one (dataset, task), keeping only papers
    with full coverage: every model has N_SAMPLES valid responses and one
    cached vector per response.

    Returns (corpus, excluded) where excluded maps each dropped paper to the
    reasons. Raises if fewer than MIN_PAPERS papers are fully covered.
    """
    skip = dataset.skip_ids()
    results_dir, emb_dir = dataset.results_dir(task), dataset.embeddings_dir(task)
    candidates = sorted(p.stem for p in results_dir.glob("*.json") if p.stem not in skip)

    paper_ids: list[str] = []
    vectors: list[list[np.ndarray]] = []
    excluded: dict[str, list[str]] = {}
    for pid in candidates:
        results = {canonical_slug(k): v for k, v in _read(results_dir / f"{pid}.json").items()}
        emb_path = emb_dir / f"{pid}.json"
        cached = {canonical_slug(k): v for k, v in _read(emb_path).items()} if emb_path.exists() else {}

        reasons, per_model = [], []
        for slug in SLUGS:
            n_valid = sum(is_valid(s) for s in results.get(slug, []))
            vecs = np.asarray(cached.get(slug, []), dtype=np.float64).reshape(-1, 1536)
            if n_valid < N_SAMPLES:
                reasons.append(f"{slug}: {n_valid}/{N_SAMPLES} valid responses")
            elif len(vecs) != n_valid:
                reasons.append(f"{slug}: {len(vecs)} cached vectors for {n_valid} responses")
            per_model.append(vecs / np.linalg.norm(vecs, axis=1, keepdims=True) if len(vecs) else vecs)
        if reasons:
            excluded[pid] = reasons
        else:
            paper_ids.append(pid)
            vectors.append(per_model)

    if len(paper_ids) < MIN_PAPERS:
        raise DataIntegrityError(
            f"{dataset.label} / {task.key}: only {len(paper_ids)} of {len(candidates)} papers have full "
            f"coverage (need {MIN_PAPERS}). Run `python embed.py --dataset {dataset.key}` first.\n  "
            + "\n  ".join(f"{pid}: {'; '.join(r)}" for pid, r in list(excluded.items())[:20])
        )
    return Corpus(dataset, task, paper_ids, vectors), excluded


def load_outputs(dataset: Dataset, task: Task, paper_ids: list[str]) -> dict[str, dict[str, list[str]]]:
    """{slug: {paper_id: [valid samples]}} from the results files."""
    out: dict[str, dict[str, list[str]]] = {s: {} for s in SLUGS}
    for pid in paper_ids:
        path = dataset.results_dir(task) / f"{pid}.json"
        for key, samples in _read(path).items():
            out[canonical_slug(key)][path.stem] = [s for s in samples if is_valid(s)]
    return out


def load_paper_metadata(dataset: Dataset, used: set[str]) -> list[dict]:
    """[{id, title, url}] for the given papers, sorted by id."""
    rows = []
    for pid in sorted(used):
        meta = _read(dataset.papers_dir / f"{pid}.json")
        rows.append({"id": pid, "title": meta["title"].strip(), "url": dataset.paper_url(pid, meta)})
    return rows


def _read(path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(per_paper: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): 95% percentile bootstrap CI of the mean over papers."""
    x = per_paper[~np.isnan(per_paper)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def length_stats(outputs: dict[str, dict[str, list[str]]]) -> dict[str, tuple[float, float]]:
    """{slug: (mean, std)} character length over all valid samples."""
    stats = {}
    for slug, by_paper in outputs.items():
        lengths = np.array([len(s) for samples in by_paper.values() for s in samples])
        stats[slug] = (float(lengths.mean()), float(lengths.std(ddof=1)))
    return stats


def length_correlations(corpus: Corpus, lengths: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Pearson r across models between mean output length and (a) mean
    intra-model similarity, (b) mean similarity to all other models."""
    mean_len = np.array([lengths[s][0] for s in SLUGS])
    mm = corpus.model_matrix()
    intra = np.diag(mm)
    cross = np.array([mm[i, OFF_DIAG[i]].mean() for i in range(len(SLUGS))])
    return {
        "intra_model": float(np.corrcoef(mean_len, intra)[0, 1]),
        "cross_model": float(np.corrcoef(mean_len, cross)[0, 1]),
    }


def model_name(slug: str) -> str:
    return MODELS_BY_SLUG[slug].name
