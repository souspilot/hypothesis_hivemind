"""
Paper figures. Every function takes already-computed arrays (see
analysis.py) and writes <path>.pdf (for LaTeX) and <path>.png (preview).

Sized for the NeurIPS text width (5.5 in) so fonts print at 6-8 pt; include
with \\includegraphics[width=\\linewidth]{...} and do not rescale.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from config import MODELS, PROVIDER_COLORS, PROVIDERS

logging.getLogger("fontTools").setLevel(logging.ERROR)  # harmless font-subsetting chatter

TEXT_WIDTH = 5.5  # inches, NeurIPS \textwidth
INK = "#1f1f1f"
MUTED = "#6b6a66"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
    "savefig.dpi": 300,
})

# Poster/slide reuse: set_scale() enlarges every font, line and marker by the
# same factor. The paper figures use the defaults below (scale 1).
SCALE = 1.0
_SCALED_RC = ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
              "legend.fontsize", "axes.linewidth", "xtick.major.width", "ytick.major.width",
              "xtick.major.size", "ytick.major.size", "xtick.major.pad", "ytick.major.pad")
_BASE_RC = {key: plt.rcParams[key] for key in _SCALED_RC}


def set_scale(scale: float, text_width: float) -> None:
    """Scale all figure typography and strokes by `scale` and set the width
    (inches) figures are drawn at."""
    global SCALE, TEXT_WIDTH
    SCALE, TEXT_WIDTH = scale, text_width
    plt.rcParams.update({key: value * scale for key, value in _BASE_RC.items()})


# Colour scales from the camera-ready figures (17 stops sampled from each
# figure's colourbar, low -> high).
MATRIX_CMAP = LinearSegmentedColormap.from_list("hivemind_matrix", [
    "#fff8ee", "#fde8d5", "#fbd9bc", "#f9c9a3", "#f7ba8c", "#f5aa72", "#ed9762", "#e48457", "#da6e4a",
    "#d05a3e", "#c74733", "#bb372a", "#aa2e27", "#9b2724", "#8b1f20", "#7b171d", "#6b0f1a",
])
HIST_CMAP = LinearSegmentedColormap.from_list("hivemind_hist", [
    "#fffde7", "#fff5d4", "#ffecc0", "#ffe4ad", "#ffdc99", "#ffd486", "#fbc67a", "#f5b672", "#efa56a",
    "#ea9662", "#e4865a", "#da7550", "#c76143", "#b44c36", "#a1382a", "#8e241d", "#7b1010",
])
# Fig. 3 curves: red = same paper, blue = different papers (camera-ready
# colours); the added "same paper, different models" curve is a tint of the red.
LEVEL_COLORS = ["#c0392b", "#d98880", "#3b6fa0"]

NAMES = [m.name for m in MODELS]
PROVIDER_SPANS = []  # (provider, first index, last index) in MODELS order
for _p in PROVIDERS:
    _idx = [i for i, m in enumerate(MODELS) if m.provider == _p]
    PROVIDER_SPANS.append((_p, _idx[0], _idx[-1]))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No CreationDate: rebuilding an unchanged figure yields an identical file.
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _cell_text_color(value, norm) -> str:
    return "white" if norm(value) > 0.55 else INK


def _color_model_labels(labels, first_model: int = 0) -> None:
    """Colour model tick labels by provider; labels before first_model (e.g.
    a pooled "All models" column) keep the default ink."""
    for i, label in enumerate(labels):
        if i >= first_model:
            label.set_color(PROVIDER_COLORS[MODELS[i - first_model].provider])


def _provider_legend(fig, y: float, ncol: int | None = None) -> None:
    handles = [Patch(facecolor=PROVIDER_COLORS[p], edgecolor="none", label=p) for p in PROVIDERS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(PROVIDERS),
               frameon=False, handlelength=1.2, handleheight=0.7, columnspacing=1.6)


def _panel_label(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", pad=4, fontweight="bold")


# Figure 1: model-by-model similarity (lower triangle)

def model_matrices(panels: list[tuple[str, np.ndarray]], path: Path,
                   size: tuple[float, float] | None = None, legend_ncol: int | None = None,
                   diagonal_labels: bool = False) -> None:
    """panels: [(title, n_models x n_models matrix)], drawn side by side on
    one shared colour scale so the panels are directly comparable."""
    n = len(NAMES)
    tri = np.tril(np.ones((n, n), dtype=bool))
    vals = np.concatenate([m[tri] for _, m in panels])
    norm = Normalize(vmin=np.floor(vals.min() * 20) / 20, vmax=np.ceil(vals.max() * 20) / 20)

    fig, axes = plt.subplots(1, len(panels), figsize=size or (TEXT_WIDTH, 3.05),
                             gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for k, (ax, (title, m)) in enumerate(zip(axes, panels)):
        shown = np.where(tri, m, np.nan)
        im = ax.imshow(shown, cmap=MATRIX_CMAP, norm=norm)
        for i in range(n):
            for j in range(i + 1):
                ax.text(j, i, f"{m[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=4.6 * SCALE, color=_cell_text_color(m[i, j], norm))
        # 1.5px surface gaps between provider blocks
        for _, a, _b in PROVIDER_SPANS[1:]:
            ax.axhline(a - 0.5, color="white", lw=1.2 * SCALE)
            ax.axvline(a - 0.5, color="white", lw=1.2 * SCALE)
        if diagonal_labels:
            # Compact layout: name each column beside its diagonal cell instead
            # of along a rotated x axis, which roughly halves the height.
            ax.set_xticks([])
            for i, model in enumerate(MODELS):
                ax.text(i + 0.62, i, model.name, ha="left", va="center",
                        color=PROVIDER_COLORS[model.provider], fontsize=plt.rcParams["ytick.labelsize"])
            ax.set_xlim(-0.5, n + 3.2)
            ax.set_yticks([])
        else:
            ax.set_xticks(range(n), NAMES, rotation=55, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(n), NAMES if k == 0 else [])
        ax.tick_params(length=0, pad=5)
        for s in ax.spines.values():
            s.set_visible(False)
        _color_model_labels(ax.get_xticklabels())
        _color_model_labels(ax.get_yticklabels())
        if len(panels) > 1:
            _panel_label(ax, "ABCD"[k], title)
        elif title:
            ax.set_title(title, loc="left", pad=4 * SCALE, fontweight="bold")

    if diagonal_labels:
        # Vertical bar in the empty upper-right corner of the last panel.
        cax = axes[-1].inset_axes([0.93, 0.5, 0.035, 0.48])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.set_ticks([norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax])
    else:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, aspect=30)
    cbar.set_label("Mean cosine similarity")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2)
    if not diagonal_labels:  # diagonal names already carry provider colour and brand
        _provider_legend(fig, 0.98 if legend_ncol is None else 1.04, legend_ncol)
    _save(fig, path)


# Figure 2: distribution of per-paper intra-model similarity

HIST_EDGES = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
HIST_LABELS = ["< 0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0"]


def intra_model_histograms(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    """panels: [(title, n_papers x n_models intra-model similarity)], stacked.
    Each column shows the % of papers whose intra-model similarity falls in
    each bin; the first column pools all models."""
    cols = ["All models"] + NAMES
    fig, axes = plt.subplots(len(panels), 1, figsize=(TEXT_WIDTH, 2.05 * len(panels) + 0.4),
                             gridspec_kw={"hspace": 0.3})
    axes = np.atleast_1d(axes)
    norm = Normalize(0, 100)
    for k, (ax, (title, intra)) in enumerate(zip(axes, panels)):
        pct = np.empty((len(HIST_LABELS), len(cols)))
        pooled = intra[~np.isnan(intra)]
        pct[:, 0] = np.histogram(pooled, HIST_EDGES)[0] / len(pooled) * 100
        for j in range(intra.shape[1]):
            x = intra[:, j][~np.isnan(intra[:, j])]
            pct[:, j + 1] = np.histogram(x, HIST_EDGES)[0] / len(x) * 100
        pct = pct[::-1]  # highest similarity bin on top
        im = ax.imshow(pct, cmap=HIST_CMAP, norm=norm, aspect="auto")
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=5.2 * SCALE,
                        color=_cell_text_color(pct[i, j], norm))
        ax.axvline(0.5, color="white", lw=2.5 * SCALE)
        for _, a, _b in PROVIDER_SPANS[1:]:
            ax.axvline(a + 0.5, color="white", lw=1.2 * SCALE)
        ax.set_yticks(range(len(HIST_LABELS)), HIST_LABELS[::-1])
        ax.set_ylabel("Intra-model similarity")
        last = k == len(panels) - 1  # panels share the model axis; label it once
        ax.set_xticks(range(len(cols)), cols if last else [], rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(length=0, pad=5)
        for s in ax.spines.values():
            s.set_visible(False)
        _color_model_labels(ax.get_xticklabels(), first_model=1)
        _panel_label(ax, "ABCD"[k], title)

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015, aspect=35)
    cbar.set_label("% of papers")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2)
    _provider_legend(fig, 0.995)
    _save(fig, path)


# Figure 3: same paper vs different papers (embedding sanity check)

LEVELS = [
    ("same_model", "Same paper, same model"),
    ("cross_model", "Same paper, different models"),
    ("different_paper", "Different papers"),
]


def relatedness_densities(grid: list[list[tuple[str, dict[str, np.ndarray]]]], path: Path) -> None:
    """grid[row][col] = (title, {level: values}); rows are datasets, columns
    tasks. Shows that the embedding separates outputs for different papers
    (~0.4) from outputs for the same paper, so high cross-model similarity
    is not an artefact of the embedding model."""
    n_rows, n_cols = len(grid), len(grid[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH, 1.55 * n_rows + 0.35),
                             sharex=True, squeeze=False, gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    xs = np.linspace(0, 1, 400)
    letter = iter("ABCDEFGH")
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            title, dists = grid[r][c]
            peak = 0
            for (key, _), color in zip(LEVELS, LEVEL_COLORS):
                v = dists[key][~np.isnan(dists[key])]
                y = gaussian_kde(v)(xs)
                peak = max(peak, y.max())
                ax.fill_between(xs, y, color=color, alpha=0.18, lw=0)
                ax.plot(xs, y, color=color, lw=1.2 * SCALE)
                mu = v.mean()
                ax.axvline(mu, color=color, lw=0.6 * SCALE, ls=(0, (2, 2)))
                ax.text(mu, y.max() * 1.04, f"{mu:.2f}", ha="center", va="bottom", fontsize=5.8 * SCALE, color=INK)
            ax.set_ylim(0, peak * 1.22)
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if r == n_rows - 1:
                ax.set_xlabel("Mean pairwise cosine similarity")
            if c == 0:
                ax.set_ylabel("Density")
            _panel_label(ax, next(letter), title)

    handles = [plt.Line2D([], [], color=col, lw=1.6 * SCALE, label=lab) for (_, lab), col in zip(LEVELS, LEVEL_COLORS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.06 / n_rows),
               ncol=3, frameon=False, handlelength=1.6, columnspacing=1.6)
    _save(fig, path)


# Figure 4: distinct ideas among 10 hypotheses (Vendi score)

SAME_PAPER_COLOR, OTHER_PAPERS_COLOR = LEVEL_COLORS[0], LEVEL_COLORS[2]
VENDI_LABEL = "Effective no. of distinct hypotheses (of 10)"


def diversity_slopes(grid: list[list[tuple[str, dict]]], path: Path, seed: int = 0) -> None:
    """Paper figure. grid[row][col] = (title, analysis.diversity_summary(...)).
    One thin line per paper from '1 model' to '10 models' (bold: mean), next
    to a cloud of 10-hypothesis sets drawn from 10 different papers."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = len(grid), len(grid[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH, 2.05 * n_rows + 0.2),
                             sharey=True, squeeze=False, gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    x_one, x_ten, x_ref = 0.0, 1.0, 2.15
    letter = iter("ABCDEFGH")
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            title, d = grid[r][c]
            for a, b in zip(d["one"], d["ten"]):
                ax.plot([x_one, x_ten], [a, b], color=SAME_PAPER_COLOR, alpha=0.2, lw=0.5 * SCALE)
            for x, vals in ((x_one, d["one"]), (x_ten, d["ten"])):
                ax.scatter(np.full(len(vals), x), vals, s=4 * SCALE ** 2, color=SAME_PAPER_COLOR, alpha=0.4, lw=0)
            m_one, m_ten, m_ref = d["one_mean"][0], d["ten_mean"][0], d["unrelated_mean"]
            ax.plot([x_one, x_ten], [m_one, m_ten], color=SAME_PAPER_COLOR, lw=2 * SCALE, marker="o",
                    ms=4.5 * SCALE, mfc="white", mew=1.4 * SCALE, zorder=5)
            ref = rng.choice(d["unrelated"], min(150, len(d["unrelated"])), replace=False)
            ax.scatter(x_ref + rng.uniform(-0.17, 0.17, len(ref)), ref, s=4 * SCALE ** 2,
                       color=OTHER_PAPERS_COLOR, alpha=0.3, lw=0)
            ax.plot([x_ref - 0.24, x_ref + 0.24], [m_ref, m_ref], color=OTHER_PAPERS_COLOR, lw=2 * SCALE)
            for x, y, ha, color in ((x_one - 0.09, m_one, "right", SAME_PAPER_COLOR),
                                    (x_ten + 0.09, m_ten, "left", SAME_PAPER_COLOR),
                                    (x_ref + 0.3, m_ref, "left", OTHER_PAPERS_COLOR)):
                ax.text(x, y, f"{y:.1f}", ha=ha, va="center", color=color, fontweight="bold",
                        fontsize=plt.rcParams["font.size"] * 0.95)
            ax.set_xticks([x_one, x_ten, x_ref], ["1 model", "10 models", "10 different\npapers"])
            ax.set_xlim(-0.5, 2.75)
            ax.set_ylim(1, 8.6)
            ax.tick_params(axis="x", length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_ylabel("Effective no. of distinct\nhypotheses (of 10)")
            _panel_label(ax, next(letter), title)
    _save(fig, path)


def diversity_ruler(groups: list[tuple[str, list[tuple[str, dict]]]], path: Path,
                    size: tuple[float, float] | None = None) -> None:
    """Poster figure. groups = [(group heading, [(row label, summary), ...]), ...].
    Each row is a 1-10 scale of distinct ideas among 10 hypotheses: an arrow
    from 1 model to 10 models, and a diamond for 10 different papers (band:
    its interquartile range)."""
    n_rows = sum(len(rows) for _, rows in groups)
    fig, ax = plt.subplots(figsize=size or (TEXT_WIDTH, 0.45 * (n_rows + len(groups)) + 0.5))
    track, y = "#e6e3dc", 0.0
    label_x = 0.75
    for heading, rows in groups:
        ax.text(1, y, heading, ha="left", va="center", fontweight="bold", color=INK)
        y -= 0.85
        for label, d in rows:
            m_one, m_ten, m_ref = d["one_mean"][0], d["ten_mean"][0], d["unrelated_mean"]
            q1, q3 = np.percentile(d["unrelated"], [25, 75])
            ax.plot([1, 10], [y, y], color=track, lw=4.5 * SCALE, solid_capstyle="round", zorder=0)
            ax.plot([q1, q3], [y, y], color=OTHER_PAPERS_COLOR, alpha=0.3, lw=4.5 * SCALE,
                    solid_capstyle="round", zorder=1)
            ax.annotate("", xy=(m_ten, y), xytext=(m_one, y), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=SAME_PAPER_COLOR, lw=1.5 * SCALE,
                                        mutation_scale=7 * SCALE, shrinkA=2.5 * SCALE, shrinkB=3 * SCALE))
            ax.plot(m_one, y, "o", ms=5.5 * SCALE, mfc="white", mec=SAME_PAPER_COLOR, mew=1.4 * SCALE, zorder=3)
            ax.plot(m_ten, y, "o", ms=5.5 * SCALE, color=SAME_PAPER_COLOR, zorder=3)
            ax.plot(m_ref, y, "D", ms=5 * SCALE, color=OTHER_PAPERS_COLOR, zorder=3)
            ax.text(label_x, y, label, ha="right", va="center", color=INK)
            y -= 1
        y -= 0.35
    ax.set_xlim(1, 10)
    ax.set_ylim(y + 0.6, 0.45)
    ax.set_yticks([])
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(VENDI_LABEL)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=5.5 * SCALE, mfc="white", mec=SAME_PAPER_COLOR,
                   mew=1.4 * SCALE, label="1 model"),
        plt.Line2D([], [], ls="none", marker="o", ms=5.5 * SCALE, color=SAME_PAPER_COLOR, label="10 models"),
        plt.Line2D([], [], ls="none", marker="D", ms=5 * SCALE, color=OTHER_PAPERS_COLOR,
                   label="10 different papers"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.6, borderaxespad=0.2)
    _save(fig, path)
