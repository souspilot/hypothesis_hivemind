# Public release verification

Verified on 9 October 2026 with Python 3.14 on macOS.

## Changes

- Rewrote the README and shortened Python comments and docstrings. Scientific prompts, sampling settings, statistical calculations, and existing responses are preserved.
- Added paper IDs, titles, and source links in `metadata/ai4mat.json` (50 records) and `metadata/plos.json` (57 records). No article bodies are included. The asset builder reads these metadata files before falling back to local article JSON.
- Added `prepare_plos.py` to select the 52 non-excluded study articles from processed PLOS JSON. It supports a validation-only check and preserves existing files, reporting conflicting content as an error.
- Added SHA-256 fingerprints of the embedding model and ordered response texts. Embedding runs detect edits with unchanged sample counts. Analysis rejects mismatched fingerprints where present and retains compatibility with the original study caches.
- Corrected the summary generation identifier to `anthropic/claude-sonnet-4.6`, as listed on [OpenRouter](https://openrouter.ai/anthropic/claude-sonnet-4.6).
- Added pinned analysis dependencies and a full dependency environment, plus regression tests.
- Added MIT licensing for code and CC BY 4.0 licensing for documentation and study-generated assets. Original article licences and attribution are described in `LICENSING.md`.

## Validation

Four regression tests pass. They cover metadata loading without full article text, cache reuse and invalidation after response edits, analysis rejection of stale fingerprints, and PLOS selection with dry-run and conflict behavior.

The full asset builder completed in a temporary output directory using the original saved embeddings. All 14 report, CSV, and LaTeX outputs match the saved versions byte-for-byte, including `REPORT.md`, `numbers.tex`, `appendix_datasets.tex`, and every file in `tables/`. Representative regenerated heatmap and diversity PNGs were visually inspected.

PLOS preparation validates against the current processed and training files with zero copies needed. All 12 model clients and the embedding client construct with the pinned dependencies using a placeholder key; no generation or embedding API calls were made. `pip check` reports no broken requirements.

The stored responses have ten samples accepted by the current validity rule for every configured model on 50 AI4Mat and 52 non-excluded PLOS papers in each task. Summary and response JSON files parse successfully. Original article files, responses, embeddings, and saved paper assets were not modified.

A previous pattern scan of 535 tracked files and 570 blobs in reachable Git history found no common provider-key, GitHub-token, AWS access-key-ID, or private-key-header matches. No environment files, logs, notebooks, or AI4Mat full-text directories were found in that tracked history. This was a pattern scan, not an exhaustive secret audit.

## Reproduction notes

Embeddings remain excluded from Git; a fresh clone must regenerate them using an OpenRouter key before building assets. Full article text is required only to generate new responses, not to build assets from saved responses and embeddings.

Older embedding caches lack fingerprints. The analysis can read them for reproducing the stored experiment, but the next embedding run will regenerate their entries to establish fingerprints. Live generation services were not tested; new responses may differ from the recorded experiment.

The existing compact-figure changes and four `fig1_main_*` files are retained for the release. The licence files, metadata, tests, dependency manifests, and new scripts should be included with the repository changes.

## Redistribution follow-up

Copyright attribution now names the authors of *Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery*. The PLOS article files were checked individually for CC BY 4.0 declarations, and full original attribution is supplied in `PLOS_ATTRIBUTION.md`. See `DATA_RIGHTS_REVIEW.md` for the evidence, the flagged AI4Mat summary, and the account-specific provider terms that remain unverified. The earlier numerical and regression checks do not constitute legal clearance of every source-derived summary.
