# Hypothesis hivemind experiment

Code, model outputs, and figures for the hypothesis hivemind experiment in [*Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery*](https://arxiv.org/abs/2605.08956) (NeurIPS 2026, Position Paper Track).

We compare hypotheses from 12 models across four providers on two tasks: recovering the hypothesis behind a set of experiments, and proposing a novel hypothesis from a full paper. The study uses 50 NeurIPS 2025 AI4Mat papers and 52 PLOS Biology papers from April–June 2026. Each model produces 10 responses per paper and task. Responses are embedded with `text-embedding-3-small` and compared using cosine similarity and the Vendi score.

## Repository contents

| Path | Contents |
|---|---|
| `config.py` | Dataset paths, model identifiers, task definitions, and generation settings |
| `metadata/` | Paper IDs, titles, and source links, without article text |
| `data/experiments_summary/` | AI4Mat experiment summaries |
| `data2/` | PLOS Biology XML, extracted article text, and experiment summaries |
| `results/`, `results2/` | Model responses for AI4Mat and PLOS Biology, respectively |
| `analysis.py` | Similarity, diversity, and statistical calculations |
| `figures.py` | Plotting functions |
| `make_paper_assets.py` | Builds figures, tables, and numerical summaries |
| `paper_assets/` | Saved figures, tables, dataset lists, and analysis report |
| `plos_2026_apr_to_jun.txt` | PLOS article DOI list |
| `plos_skip.txt` | Five PLOS article IDs excluded from generation and analysis |

The task names used on the command line are `recover` and `novel`. Their output directories are `underlying_hypotheses` and `new_hypotheses`. Dataset names are `ai4mat` and `plos`; commands with no dataset or task selection process both where supported.

The saved analysis includes 50 AI4Mat papers and 52 PLOS Biology papers for each task. Additional PLOS files remain in the source and output directories, including skip-listed articles.

## Viewing the results

The figures and tables in `paper_assets/` can be viewed without running the code or making API calls. [The analysis report](paper_assets/REPORT.md) gives coverage counts, similarity estimates, confidence intervals, and diversity measures. [The dataset appendix](paper_assets/appendix_datasets.tex) lists paper titles and source links.

| File in `paper_assets/` | Contents |
|---|---|
| `figures/fig1_main_{ai4mat,plos}.pdf` | Compact similarity heatmaps with model names along the diagonal |
| `figures/fig1_model_similarity_{ai4mat,plos}.pdf` | Similarity heatmaps with full row and column labels |
| `figures/fig2_intra_model_{ai4mat,plos}.pdf` | Intra-model similarity distributions |
| `figures/fig3_same_vs_different_paper.pdf` | Similarity for responses to the same paper and different papers |
| `figures/fig4_diversity.pdf` | Vendi scores for hypotheses sampled from one model, ten models, and ten papers |
| `tables/similarity_summary.tex` | Mean similarities with 95% bootstrap confidence intervals |
| `tables/output_length.tex` | Output lengths and their correlations with similarity |
| `tables/models.tex` | Model identifiers, open-weight status, and generation settings |
| `tables/model_matrix_*.csv` | Values underlying the similarity heatmaps |
| `tables/per_paper_*.csv` | Similarity summaries for individual papers |
| `appendix_datasets.tex` | Paper titles and OpenReview or DOI links |
| `numbers.tex` | LaTeX macros for numerical results |
| `REPORT.md` | Coverage checks and numerical summaries |

Figures are saved as PDF and PNG. The compact `fig1_main_*` files use the same similarity values as the fully labelled heatmaps.

## Reproducing the analysis

The dependency environment is tested with Python 3.14. The pinned NumPy and SciPy versions require Python 3.12 or later.

Create a virtual environment and install the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

For embedding, generation, and article preprocessing, use the full pinned environment, which includes transitive dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. Embeddings are excluded from Git. To check which cached entries are missing, have different sample counts, or lack matching content fingerprints:

```bash
python embed.py --check
```

To rebuild them, set `OPENROUTER_API_KEY` in your environment or in a local `.env` file, then run:

```bash
python embed.py
```

This command makes paid API calls and writes to `embeddings/` and `embeddings2/`. Each cache entry has a SHA-256 fingerprint of the embedding model and ordered response texts. Entries are reused only when the fingerprint and vector count match. Fingerprints are stored in a `.fingerprints/` subdirectory alongside the vector files. Older caches without fingerprints are marked for re-embedding on their next embedding run; this can regenerate all entries even if their counts match. The analysis reader still accepts original study caches without fingerprints, but rejects mismatches when fingerprints are present.

Once embeddings are available, build the assets into a separate directory:

```bash
python make_paper_assets.py --out reproduced_assets
```

The builder makes no API calls. It includes only papers with complete model-output and embedding coverage and requires at least 50 qualifying papers per dataset and task. Excluded papers and reasons are recorded in the generated report.

The builder reads paper titles and source links from `metadata/ai4mat.json` and `metadata/plos.json`. Full article text is not needed to reproduce assets from stored responses and embeddings. For papers absent from those metadata files, the builder falls back to local article JSON in the dataset's `train/` directory.

Run the regression checks without API calls:

```bash
python -m unittest discover -s tests -v
```

## Generating new responses

The stored responses are sufficient to rebuild embeddings; generating them again is a separate step. New runs can differ from the stored experiment because models and provider services may change.

| Stage | Script | External service |
|---|---|---|
| Download PLOS articles | `download_plos.py` | PLOS; writes PDFs and XML |
| Convert PLOS XML to JSON | `xml_to_json.py` | None; writes `data2/processed/` |
| Select study articles | `prepare_plos.py` | None; copies non-excluded study articles to `data2/train/` |
| Summarize experiments | `extract_experiments_summary.py --dataset plos` | OpenRouter |
| Recover hypotheses | `generate_hypotheses.py --dataset plos` | OpenRouter |
| Generate novel hypotheses | `generate_new_hypotheses.py --dataset plos` | OpenRouter |
| Embed responses | `embed.py --dataset plos --task novel` | OpenRouter |

Generation reads article JSON from `data/train/` or `data2/train/`. The XML converter writes to `data2/processed/`. Run `python prepare_plos.py --check` to validate the inputs and `python prepare_plos.py` to copy the 52 non-excluded study articles into `data2/train/`. Existing files are preserved; conflicting contents produce an error. Prepared PLOS training files are also included in this repository. AI4Mat full texts must be obtained separately from the sources listed in the dataset appendix.

Hypothesis generation preserves valid stored responses and fills missing samples up to ten per model and paper. Readers accept both current model identifiers and legacy identifiers from earlier direct-API runs. The prompts are defined in the generation scripts.

The AI4Mat Anthropic and OpenAI responses were generated through the providers' own APIs before the switch to OpenRouter. Other responses used OpenRouter. Recorded settings appear in `config.GENERATION_SETTINGS` and `paper_assets/tables/models.tex`; the current scripts use OpenRouter for generation.

Maintenance tools:

- `find_garbled_samples.py` reports newline counts and samples for inspection.
- `clean_blank_samples.py` removes invalid responses from result files. It modifies stored data.
- `model_utils.py`, when run directly, makes one test generation call per model.
- `verify_caching.py` makes API calls to inspect Claude prompt-cache usage.

## Metrics

Cosine similarities exclude self-pairs. Similarity statistics are computed per paper and then averaged across papers, so each paper has equal weight. Confidence intervals use 10,000 bootstrap resamples of papers with a fixed random seed.

- **Intra-model similarity:** distinct response pairs from the same model and paper.
- **Model-pair similarity:** all response pairs between two models for the same paper. Heatmap diagonals show intra-model similarity.
- **Intra-provider similarity:** model-pair similarities averaged across distinct models from the same provider.
- **Cross-provider similarity:** model-pair similarities averaged across models from different providers.
- **Relatedness comparison:** responses from the same paper and model, the same paper and different models, or different papers.
- **Diversity:** the [Vendi score](https://arxiv.org/abs/2210.02410) with a cosine kernel over ten hypotheses. The comparison uses ten responses from one model, one response from each of ten randomly selected models for the same paper (200 draws), or responses about ten different papers as a reference.
- **PERMANOVA R²:** the fraction of variation in each paper's embeddings explained by model or provider identity. The analysis uses 199 permutations per paper.

## Data sources and reuse

Model responses and Claude Sonnet 4.6 experiment summaries were generated for this study. PLOS article XML files retain their original [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence information. Full article-level author, copyright, source, and licence information is in [PLOS_ATTRIBUTION.md](PLOS_ATTRIBUTION.md). Preserve that notice when redistributing the article XML, extracted JSON, or PLOS summaries.

AI4Mat full texts are not distributed here. PLOS PDFs are also excluded because the XML is included. Embeddings can be rebuilt from the stored responses. Logs, the preprocessing notebook, and local Python caches are excluded from Git.

Code and tests use the [MIT License](LICENSE). Documentation and study-generated assets use [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Source articles retain their original licences. See [licensing and attribution](LICENSING.md) for the scope and attribution requirements, and [the data-rights review](DATA_RIGHTS_REVIEW.md) for redistribution evidence and unresolved items.

## Citation

```bibtex
@article{bisht2026agentic,
  title = {Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery},
  author = {Bisht, Harshit and Kumar, Vinay and Jablonka, Kevin Maik and Mausam and Krishnan, N. M. Anoop},
  journal = {arXiv preprint arXiv:2605.08956},
  year = {2026},
  doi = {10.48550/arXiv.2605.08956}
}
```
