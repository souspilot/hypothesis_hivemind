"""
Single source of truth for datasets, tasks, and models.

Every script (generation, embedding, analysis) looks things up here instead
of hard-coding directory suffixes or model-id strings. The on-disk layout is
unchanged -- AI4Mat lives in data/ results/ embeddings/, PLOS Biology in
data2/ results2/ embeddings2/ -- but that mapping now exists in exactly one
place (DATASETS below).
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    key: str            # CLI name, e.g. "ai4mat"
    label: str          # short label for figures/tables
    description: str    # one-line description for the appendix
    suffix: str         # on-disk directory suffix ("" or "2")
    skip_list: Path | None = None  # paper IDs excluded from every stage

    def _dir(self, stem: str) -> Path:
        return ROOT / f"{stem}{self.suffix}"

    @property
    def papers_dir(self) -> Path:
        return self._dir("data") / "train"

    @property
    def summaries_dir(self) -> Path:
        return self._dir("data") / "experiments_summary"

    def results_dir(self, task: "Task") -> Path:
        return self._dir("results") / task.dirname

    def embeddings_dir(self, task: "Task") -> Path:
        return self._dir("embeddings") / task.dirname

    def skip_ids(self) -> frozenset[str]:
        if self.skip_list is None or not self.skip_list.exists():
            return frozenset()
        return frozenset(l.strip() for l in self.skip_list.read_text().splitlines() if l.strip())

    def paper_url(self, paper_id: str, meta: dict) -> str:
        if self.key == "ai4mat":
            return f"https://openreview.net/pdf?id={paper_id}"
        return f"https://doi.org/{meta.get('doi') or '10.1371/' + paper_id}"


DATASETS: dict[str, Dataset] = {
    "ai4mat": Dataset(
        key="ai4mat",
        label="AI4Mat",
        description="Accepted papers, NeurIPS 2025 AI4Mat workshop",
        suffix="",
    ),
    "plos": Dataset(
        key="plos",
        label="PLOS Biology",
        description="PLOS Biology, April--June 2026 issues",
        suffix="2",
        # Papers whose text reliably trips a provider content filter on every
        # sample; excluded from generation and analysis alike.
        skip_list=ROOT / "plos_skip.txt",
    ),
}

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    key: str       # CLI / file-name key
    dirname: str   # directory under results*/ and embeddings*/
    label: str     # label used in figures and tables


TASKS: dict[str, Task] = {
    "recover": Task("recover", "underlying_hypotheses", "Recover underlying hypothesis"),
    "novel": Task("novel", "new_hypotheses", "Generate novel hypothesis"),
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

PROVIDERS = ["Anthropic", "Google", "Moonshot AI", "OpenAI"]

# Provider identity colours, as used in the camera-ready figures.
PROVIDER_COLORS = {
    "Anthropic": "#e07b54",
    "Google": "#4c9f70",
    "Moonshot AI": "#8e6bc7",
    "OpenAI": "#6b9fd4",
}


@dataclass(frozen=True)
class Model:
    slug: str                 # OpenRouter slug; canonical id everywhere
    name: str                 # display name for the paper
    provider: str
    open_weights: bool = False
    # Keys this model was stored under by earlier pipeline versions (the
    # original AI4Mat runs called Anthropic/OpenAI directly, not through
    # OpenRouter). Readers map these back to `slug`.
    legacy_keys: tuple[str, ...] = field(default_factory=tuple)


# Order here is the order models appear in every figure and table.
MODELS: list[Model] = [
    Model("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "Anthropic",
          legacy_keys=("anthropic:claude-haiku-4-5-20251001",)),
    Model("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5", "Anthropic",
          legacy_keys=("anthropic:claude-sonnet-4-5",)),
    Model("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6", "Anthropic",
          legacy_keys=("anthropic:claude-sonnet-4-6",)),
    Model("google/gemma-4-31b-it", "Gemma 4 31B", "Google", open_weights=True),
    Model("google/gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite", "Google"),
    Model("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", "Google"),
    Model("moonshotai/kimi-k2.6", "Kimi K2.6", "Moonshot AI", open_weights=True),
    Model("moonshotai/kimi-k2.7-code", "Kimi K2.7 Code", "Moonshot AI", open_weights=True),
    Model("moonshotai/kimi-k3", "Kimi K3", "Moonshot AI", open_weights=True),
    Model("openai/gpt-5-nano", "GPT-5 Nano", "OpenAI",
          legacy_keys=("openai:gpt-5-nano-2025-08-07",)),
    Model("openai/gpt-5-mini", "GPT-5 Mini", "OpenAI",
          legacy_keys=("openai:gpt-5-mini-2025-08-07",)),
    Model("openai/gpt-5", "GPT-5", "OpenAI",
          legacy_keys=("openai:gpt-5",)),
]

MODELS_BY_SLUG = {m.slug: m for m in MODELS}
_KEY_TO_SLUG = {k: m.slug for m in MODELS for k in (m.slug, *m.legacy_keys)}


def canonical_slug(key: str) -> str:
    """Map any stored model key (current slug or legacy key) to its slug."""
    try:
        return _KEY_TO_SLUG[key]
    except KeyError:
        raise KeyError(f"Unknown model key {key!r}; add it to config.MODELS") from None


def stored_key(result: dict, slug: str) -> str:
    """The key `slug` is already stored under in a results/embeddings dict,
    or `slug` itself if absent. Keeps re-runs from generating a fresh set of
    samples under the new slug when a legacy key already holds them."""
    for key in (slug, *MODELS_BY_SLUG[slug].legacy_keys):
        if key in result:
            return key
    return slug


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

N_SAMPLES = 10
# A (dataset, task) is analysed only on papers where all models have N_SAMPLES
# embedded responses; the build fails if fewer than this many papers qualify.
MIN_PAPERS = 50
EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Generation settings, as actually used, for the appendix table. The AI4Mat
# Anthropic/OpenAI samples predate the switch to OpenRouter (git 78c3df4) and
# were drawn through the providers' own APIs.
GENERATION_SETTINGS = {
    "direct": {
        "Anthropic": "Anthropic API; max\\_tokens 4096; default temperature (1.0)",
        "OpenAI": "OpenAI API; temperature 1.0; default reasoning effort",
    },
    "openrouter": {
        "Anthropic": "OpenRouter (Anthropic protocol); max\\_tokens 4096; default temperature",
        "other": "OpenRouter; max\\_tokens 8192; reasoning effort \\texttt{low}; default temperature",
    },
}


def generation_route(dataset: Dataset, model: Model) -> str:
    """Which settings row in GENERATION_SETTINGS produced (dataset, model)."""
    if dataset.key == "ai4mat" and model.legacy_keys:
        return GENERATION_SETTINGS["direct"][model.provider]
    if model.provider == "Anthropic":
        return GENERATION_SETTINGS["openrouter"]["Anthropic"]
    return GENERATION_SETTINGS["openrouter"]["other"]


def parse_dataset_args(values: list[str] | None) -> list[Dataset]:
    """Shared --dataset handling: none given means all datasets."""
    return [DATASETS[v] for v in values] if values else list(DATASETS.values())
