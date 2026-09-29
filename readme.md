# Hypothesis hivemind experiment

Code for the *hypothesis hivemind* experiment in "Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery" (NeurIPS 2026, Position Paper Track). It covers 12 models from 4 providers, two tasks, and two datasets. Each model draws 10 samples per paper, the samples are embedded with `text-embedding-3-small`, and the analysis compares cosine similarity within and across models and providers.

## Layout

`config.py` is the single source of truth for:

- **Datasets:** `ai4mat` (NeurIPS 2025 AI4Mat, 50 papers) and `plos` (PLOS Biology Apr–Jun 2026, 52 papers after the skip list).
- **Tasks:** `recover` (underlying hypothesis from an experiment summary) and `novel` (novel hypothesis from the full paper).
- **Models:** display names, providers, open-weights flag, and the legacy keys used by the original direct-API AI4Mat runs.

On disk, AI4Mat lives in `data/ results/ embeddings/` and PLOS Biology in `data2/ results2/ embeddings2/`. Only `config.py` knows that mapping.

| Stage | Script | API calls |
|---|---|---|
| PLOS download + XML→JSON | `download_plos.py`, `xml_to_json.py` | PLOS |
| Experiment summaries | `extract_experiments_summary.py [--dataset ...]` | yes |
| Recover hypotheses | `generate_hypotheses.py [--dataset ...]` | yes |
| Novel hypotheses | `generate_new_hypotheses.py [--dataset ...]` | yes |
| Embeddings | `embed.py [--dataset ...] [--task ...] [--check]` | yes (`--check`: no) |
| **Paper figures, tables, numbers** | `make_paper_assets.py [--out DIR]` | **no** |

The generation stages are resumable. A (paper, model) pair that already has 10 valid samples is skipped, including samples stored under a legacy key. `embed.py` re-embeds any entry whose vector count no longer matches its valid-sample count.

Maintenance helpers: `find_garbled_samples.py` (report), `clean_blank_samples.py` (drop invalid entries so the next run regenerates them), `model_utils.py` (smoke test when run directly).

## Reproducing the paper assets

```bash
pip install anthropic openai langchain-openai python-dotenv numpy scipy matplotlib lxml
echo "OPENROUTER_API_KEY=..." > .env

python embed.py --check          # anything stale?
python embed.py                  # fix it (cheap: only stale entries are sent)
python make_paper_assets.py      # -> paper_assets/
```

Each (dataset, task) is analysed only on papers with full coverage, meaning all 12 models have 10 embedded responses. `REPORT.md` lists any excluded papers, and the build fails if fewer than 50 papers qualify (`config.MIN_PAPERS`).

### What `paper_assets/` contains

| File | Use in paper |
|---|---|
| `figures/fig1_model_similarity_ai4mat.pdf` | Fig. 1 (main text) |
| `figures/fig1_model_similarity_plos.pdf` | Appendix: same for PLOS Biology |
| `figures/fig2_intra_model_{ai4mat,plos}.pdf` | Appendix: intra-model similarity distributions (Fig. 2) |
| `figures/fig3_same_vs_different_paper.pdf` | Appendix: embedding sanity check (Fig. 3) |
| `figures/fig4_diversity.pdf` | Effective number of distinct hypotheses: 1 model vs 10 models vs 10 papers (Fig. 4) |
| `tables/similarity_summary.tex` | Intra-model / intra-provider / cross-provider, 95% bootstrap CIs |
| `tables/output_length.tex` | Output length per model, with correlation to similarity |
| `tables/models.tex` | Models, identifiers, open weights, generation settings |
| `appendix_datasets.tex` | Appendix A.1 / A.2 paper lists |
| `numbers.tex` | `\newcommand` macros for every number quoted in the text |
| `tables/*.csv` | Raw numbers behind each figure |
| `REPORT.md` | Data checks and every number, human readable |

Include figures at `width=\linewidth`. They are sized for the NeurIPS text width, so the fonts print at 6–8 pt.

## Metric definitions

All metrics are cosine similarities with self-pairs excluded, computed per paper and then averaged over papers.

- **Intra-model:** pairs of distinct samples from the same model.
- **Model pair (a, b):** all sample pairs across a and b. These fill the off-diagonal cells of Fig. 1; the diagonal is intra-model.
- **Intra-provider / cross-provider:** model-pair similarity averaged over distinct models with the same provider / with different providers.
- **Fig. 3 levels:** same paper and same model (= intra-model); same paper and different models; different papers (any models).
- **Fig. 4 diversity:** the [Vendi score](https://arxiv.org/abs/2210.02410) with a cosine kernel, always over exactly 10 hypotheses: one model's own 10 for a paper, one hypothesis each from 10 random models for the same paper (200 draws), or 10 hypotheses about 10 different papers (a scale reference).
- **PERMANOVA R²** ([Anderson 2001](https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x)): the share of each paper's variation in hypothesis embeddings explained by model or provider identity, with a 199-permutation test on each paper.

## Generation settings

The AI4Mat Anthropic and OpenAI samples predate the move to OpenRouter (git `78c3df4`) and were drawn through the providers' own APIs. Everything else went through OpenRouter. `tables/models.tex` gives the exact settings for each (dataset, model); the source is `config.GENERATION_SETTINGS`.

## Data in this repository

| Path | Contents | Licence |
|---|---|---|
| `results/`, `results2/` | All model outputs, keyed by model | Generated for this study |
| `data/experiments_summary/`, `data2/experiments_summary/` | Methods summaries written by Claude Sonnet 4.6 | Generated for this study |
| `data2/xml/`, `data2/processed/`, `data2/train/` | PLOS Biology articles (JATS XML and extracted text) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite the original articles, listed with DOIs in `paper_assets/appendix_datasets.tex` |
| `paper_assets/` | Every figure, table and number in the paper | Generated for this study |

Not committed:

- **AI4Mat full texts.** The papers carry no open licence. They are listed with their OpenReview links in `paper_assets/appendix_datasets.tex`.
- **PLOS PDFs.** They duplicate the committed XML; `download_plos.py` re-fetches them.
- **Embeddings.** They total about 720 MB; `python embed.py` rebuilds them from `results*/`.

