"""
Build every figure, table, and number the paper uses from cached embeddings
(no API calls). Run embed.py first.

Each (dataset, task) is analysed on the papers with full coverage -- all 12
models have N_SAMPLES responses, each embedded. Papers without it are left
out and listed in REPORT.md; the build fails if fewer than MIN_PAPERS remain.

  python make_paper_assets.py                 # -> paper_assets/
  python make_paper_assets.py --out /path/to/overleaf/assets

Outputs
  figures/fig1_model_similarity_{ai4mat,plos}.pdf     main text (AI4Mat) / appendix (PLOS)
  figures/fig2_intra_model_{ai4mat,plos}.pdf          appendix
  figures/fig3_same_vs_different_paper.pdf            appendix
  figures/fig4_diversity.pdf                          distinct ideas (Vendi): 1 model vs 10 models vs 10 papers
  tables/similarity_summary.tex                       intra-model / intra-provider / cross-provider
  tables/output_length.tex                            per-model length + correlation with similarity
  tables/models.tex                                   models and generation settings
  tables/model_matrix_<dataset>_<task>.csv            numbers behind Fig. 1
  tables/per_paper_<dataset>_<task>.csv               per-paper group means
  appendix_datasets.tex                               A.1 / A.2 paper lists
  numbers.tex                                         \\newcommand macros for in-text numbers
  REPORT.md                                           data checks + every number, human readable
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import figures
from analysis import (SLUGS, Corpus, bootstrap_ci, diversity_summary, length_correlations, length_stats,
                      load_corpus, load_outputs, load_paper_metadata)
from config import (DATASETS, EMBEDDING_MODEL, MODELS, N_SAMPLES, PROVIDERS, TASKS, Dataset,
                    generation_route)

GROUPS = [("intra_model", "Intra-model"), ("intra_provider", "Intra-provider"), ("cross_provider", "Cross-provider")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()
    out: Path = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    corpora: dict[tuple[str, str], Corpus] = {}
    excluded: dict[tuple[str, str], dict[str, list[str]]] = {}
    for d in DATASETS.values():
        for t in TASKS.values():
            corpora[d.key, t.key], excluded[d.key, t.key] = load_corpus(d, t)

    report = Report()
    report.data_checks(corpora, excluded)

    # ---- Figures ---------------------------------------------------------
    for d in DATASETS.values():
        figures.model_matrices(
            [(t.label, corpora[d.key, t.key].model_matrix()) for t in TASKS.values()],
            out / "figures" / f"fig1_model_similarity_{d.key}",
        )
        figures.intra_model_histograms(
            [(t.label, corpora[d.key, t.key].intra_model) for t in TASKS.values()],
            out / "figures" / f"fig2_intra_model_{d.key}",
        )
    figures.relatedness_densities(
        [[(f"{d.label}, {t.key} hypothesis", corpora[d.key, t.key].paper_level_distributions())
          for t in TASKS.values()] for d in DATASETS.values()],
        out / "figures" / "fig3_same_vs_different_paper",
    )
    diversity = {key: diversity_summary(c) for key, c in corpora.items()}
    figures.diversity_slopes(
        [[(f"{d.label}, {t.key} hypothesis", diversity[d.key, t.key]) for t in TASKS.values()]
         for d in DATASETS.values()],
        out / "figures" / "fig4_diversity",
    )

    # ---- Numbers ---------------------------------------------------------
    macros: dict[str, str] = {
        "hmNModels": str(len(MODELS)),
        "hmNProviders": str(len(PROVIDERS)),
        "hmNSamples": str(N_SAMPLES),
    }
    summary_rows = []
    for (dk, tk), c in corpora.items():
        prefix = f"hm{_camel(dk)}{_camel(tk)}"
        macros[prefix + "NPapers"] = str(len(c.paper_ids))
        per_paper = c.group_per_paper()
        row = {"dataset": DATASETS[dk].label, "task": TASKS[tk].label, "n_papers": len(c.paper_ids)}
        for g, _ in GROUPS:
            mean, lo, hi = bootstrap_ci(per_paper[g])
            row[g] = (mean, lo, hi)
            macros[prefix + _camel(g)] = f"{mean:.2f}"
        dist = c.paper_level_distributions()
        for level, _ in figures.LEVELS:
            macros[prefix + "Level" + _camel(level)] = f"{np.nanmean(dist[level]):.2f}"
        dv = diversity[dk, tk]
        macros[prefix + "IdeasOneModel"] = f"{dv['one_mean'][0]:.1f}"
        macros[prefix + "IdeasTenModels"] = f"{dv['ten_mean'][0]:.1f}"
        macros[prefix + "IdeasTenPapers"] = f"{dv['unrelated_mean']:.1f}"
        macros[prefix + "RsqModel"] = f"{dv['r2_model'][0] * 100:.0f}\\%"
        macros[prefix + "RsqProvider"] = f"{dv['r2_provider'][0] * 100:.0f}\\%"
        summary_rows.append(row)
        _write_csv(out / "tables" / f"per_paper_{dk}_{tk}.csv",
                   ["paper_id"] + [g for g, _ in GROUPS],
                   [[pid] + [f"{per_paper[g][i]:.4f}" for g, _ in GROUPS] for i, pid in enumerate(c.paper_ids)])
        mm = c.model_matrix()
        _write_csv(out / "tables" / f"model_matrix_{dk}_{tk}.csv", ["model"] + SLUGS,
                   [[s] + [f"{x:.4f}" for x in mm[i]] for i, s in enumerate(SLUGS)])

    lengths, corrs = {}, {}
    for (dk, tk), c in corpora.items():
        lengths[dk, tk] = length_stats(load_outputs(DATASETS[dk], TASKS[tk], c.paper_ids))
        corrs[dk, tk] = length_correlations(c, lengths[dk, tk])
        prefix = f"hm{_camel(dk)}{_camel(tk)}"
        macros[prefix + "LenCorrIntra"] = f"{corrs[dk, tk]['intra_model']:.2f}"
        macros[prefix + "LenCorrCross"] = f"{corrs[dk, tk]['cross_model']:.2f}"

    # Appendix lists every paper analysed in at least one task.
    papers = {d.key: load_paper_metadata(d, {pid for t in TASKS for pid in corpora[d.key, t].paper_ids})
              for d in DATASETS.values()}
    for d in DATASETS.values():
        macros[f"hm{_camel(d.key)}NPapers"] = str(len(papers[d.key]))
        macros[f"hm{_camel(d.key)}NExcluded"] = str(len(d.skip_ids()))

    # ---- Tables ----------------------------------------------------------
    (out / "tables" / "similarity_summary.tex").write_text(similarity_table(summary_rows))
    (out / "tables" / "output_length.tex").write_text(length_table(lengths, corrs))
    (out / "tables" / "models.tex").write_text(models_table())
    (out / "appendix_datasets.tex").write_text(dataset_lists(papers))
    (out / "numbers.tex").write_text(
        "% Generated by make_paper_assets.py -- do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )

    report.numbers(summary_rows, corrs, lengths, corpora)
    report.diversity(diversity)
    (out / "REPORT.md").write_text(report.text())
    print(report.text())
    print(f"\nWrote assets to {out.resolve()}")


# ---------------------------------------------------------------------------
# LaTeX tables (booktabs)
# ---------------------------------------------------------------------------

def similarity_table(rows: list[dict]) -> str:
    lines = [
        "% Generated by make_paper_assets.py",
        "\\begin{tabular}{llrccc}",
        "\\toprule",
        "Dataset & Task & Papers & " + " & ".join(label for _, label in GROUPS) + " \\\\",
        "\\midrule",
    ]
    prev = None
    for r in rows:
        ds = r["dataset"] if r["dataset"] != prev else ""
        if prev is not None and ds:
            lines.append("\\addlinespace")
        prev = r["dataset"]
        cells = [f"{m:.2f} {{\\scriptsize [{lo:.2f}, {hi:.2f}]}}" for m, lo, hi in (r[g] for g, _ in GROUPS)]
        lines.append(f"{ds} & {r['task']} & {r['n_papers']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def length_table(lengths, corrs) -> str:
    keys = [(d, t) for d in DATASETS for t in TASKS]
    lines = [
        "% Generated by make_paper_assets.py. Mean (s.d.) output length in characters.",
        "\\begin{tabular}{l" + "r" * len(keys) + "}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASETS[d].label}}}" for d in DATASETS) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
        "Model & " + " & ".join(t.capitalize() for _, t in keys) + " \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        cells = [f"{lengths[k][m.slug][0]:.0f} ({lengths[k][m.slug][1]:.0f})" for k in keys]
        lines.append(f"{m.name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("$r$(length, intra-model) & " + " & ".join(f"{corrs[k]['intra_model']:.2f}" for k in keys) + " \\\\")
    lines.append("$r$(length, cross-model) & " + " & ".join(f"{corrs[k]['cross_model']:.2f}" for k in keys) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def models_table() -> str:
    routes = []  # distinct settings strings, in first-seen order -> footnote letters
    cells = {}
    for m in MODELS:
        for d in DATASETS.values():
            r = generation_route(d, m)
            if r not in routes:
                routes.append(r)
            cells[m.slug, d.key] = "abcdefgh"[routes.index(r)]
    lines = [
        "% Generated by make_paper_assets.py. Scales itself to the line width.",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllc" + "c" * len(DATASETS) + "}",
        "\\toprule",
        "Model & Provider & Identifier & Open weights & "
        + " & ".join(f"{d.label}" for d in DATASETS.values()) + " \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        check = "\\checkmark" if m.open_weights else ""
        lines.append(
            f"{m.name} & {m.provider} & \\texttt{{{m.slug}}} & {check} & "
            + " & ".join(f"({cells[m.slug, d.key]})" for d in DATASETS.values()) + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "", "\\smallskip",
              "\\begin{minipage}{\\linewidth}\\footnotesize\\raggedright"]
    lines += [f"({'abcdefgh'[i]}) {r}.\\\\" for i, r in enumerate(routes)]
    lines += [
        f"All models: {N_SAMPLES} independent samples per paper and task; no seed; "
        f"outputs embedded with \\texttt{{{EMBEDDING_MODEL.split('/')[-1]}}}.",
        "\\end{minipage}",
        "",
    ]
    return "\n".join(lines)


def dataset_lists(papers: dict[str, list[dict]]) -> str:
    out = ["% Generated by make_paper_assets.py -- requires \\usepackage{url} or hyperref"]
    for i, d in enumerate(DATASETS.values(), 1):
        out.append(f"\\subsection{{Dataset {i}: {d.label}}}")
        out.append(f"\\label{{app:dataset-{d.key}}}")
        n_skip = len(d.skip_ids())
        intro = f"{d.description}; {len(papers[d.key])} papers."
        if n_skip:
            intro += (f" A further {n_skip} papers from the same issues were excluded because a provider "
                      "content filter blocked every generation attempt for at least one model.")
        out.append(intro)
        out.append("\\begin{enumerate}")
        for p in papers[d.key]:
            out.append(f"  \\item {_tex_escape(p['title'])}\\\\\n        \\url{{{p['url']}}}")
        out.append("\\end{enumerate}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.lines: list[str] = ["# Hypothesis hivemind: generated numbers", ""]

    def data_checks(self, corpora, excluded):
        self.lines += ["## Coverage", "",
                       f"Papers are analysed only if all {len(MODELS)} models have {N_SAMPLES} embedded responses.", ""]
        for (dk, tk), c in corpora.items():
            ex = excluded[dk, tk]
            self.lines.append(f"- {DATASETS[dk].label} / {tk}: {len(c.paper_ids)} papers analysed"
                              + (f", {len(ex)} excluded (run `python embed.py --dataset {dk}`)" if ex else ""))
            for pid, reasons in ex.items():
                self.lines.append(f"    - {pid}: {'; '.join(reasons)}")
        self.lines.append("")

    def numbers(self, summary_rows, corrs, lengths, corpora):
        self.lines += ["## Similarity summary (mean over papers, 95% bootstrap CI)", "",
                       "| Dataset | Task | Papers | " + " | ".join(l for _, l in GROUPS) + " |",
                       "|---|---|---|" + "---|" * len(GROUPS)]
        for r in summary_rows:
            self.lines.append(f"| {r['dataset']} | {r['task']} | {r['n_papers']} | "
                              + " | ".join(f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in (r[g] for g, _ in GROUPS)) + " |")
        self.lines += ["", "## Same paper vs different papers (Fig. 3 means)", "",
                       "| Dataset | Task | " + " | ".join(l for _, l in figures.LEVELS) + " |",
                       "|---|---|" + "---|" * len(figures.LEVELS)]
        for (dk, tk), c in corpora.items():
            dist = c.paper_level_distributions()
            self.lines.append(f"| {DATASETS[dk].label} | {tk} | "
                              + " | ".join(f"{np.nanmean(dist[k]):.3f}" for k, _ in figures.LEVELS) + " |")
        self.lines += ["", "## Output length vs similarity (Pearson r across the 12 models)", "",
                       "| Dataset | Task | r(length, intra-model) | r(length, cross-model) |", "|---|---|---|---|"]
        for (dk, tk), c in corrs.items():
            self.lines.append(f"| {DATASETS[dk].label} | {tk} | {c['intra_model']:.2f} | {c['cross_model']:.2f} |")
        self.lines.append("")

    def diversity(self, diversity):
        self.lines += ["## Diversity (Fig. 4)", "",
                       "Distinct ideas = Vendi score (cosine kernel) of 10 hypotheses. 1 model: each model's own 10 "
                       "hypotheses for a paper. 10 models: one hypothesis each from 10 random models, same paper. "
                       "10 papers: 10 hypotheses about 10 different papers (scale reference). R^2 = PERMANOVA share "
                       "of per-paper variation explained by model / provider (mean [95% CI] over papers).", "",
                       "| Dataset | Task | Ideas, 1 model | Ideas, 10 models | Ideas, 10 papers | Papers where 10 > 1 | "
                       "R^2 model | R^2 provider | max per-paper p |",
                       "|---|---|---|---|---|---|---|---|---|"]
        for (dk, tk), d in diversity.items():
            (o, ol, oh), (t, tl, th) = d["one_mean"], d["ten_mean"]
            (rm, rml, rmh), (rp, rpl, rph) = d["r2_model"], d["r2_provider"]
            self.lines.append(f"| {DATASETS[dk].label} | {tk} | {o:.2f} [{ol:.2f}, {oh:.2f}] | "
                              f"{t:.2f} [{tl:.2f}, {th:.2f}] | {d['unrelated_mean']:.2f} | {d['share_up']:.0%} | "
                              f"{rm:.3f} [{rml:.3f}, {rmh:.3f}] | {rp:.3f} [{rpl:.3f}, {rph:.3f}] | {d['p_max']:.3f} |")
        self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------------------

_DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def _camel(s: str) -> str:
    """LaTeX-safe CamelCase: command names may contain letters only, so digits
    are spelled out (ai4mat -> AiFourmat)."""
    s = "".join(_DIGITS.get(ch, ch) for ch in s)
    return "".join(part[:1].upper() + part[1:] for part in s.replace("-", "_").split("_"))


_TEX_CHARS = {
    "\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_",
    "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}",
    "∆": "$\\Delta$", "Δ": "$\\Delta$", "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''",
}


def _tex_escape(s: str) -> str:
    return "".join(_TEX_CHARS.get(ch, ch) for ch in s)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
