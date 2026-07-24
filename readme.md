# Hypothesis Generation & Evaluation Pipeline

This project benchmarks how well different LLMs can recover and extend scientific hypotheses from research papers. It runs a multi-stage pipeline across a corpus of papers, collecting outputs from several models, then computes embedding-based similarity metrics to compare them.

## What it does

Given a directory of parsed research papers (JSON format with title, abstract, and body text), the pipeline does four things in sequence:

1. **Extracts experiment summaries** — strips out everything except the experimental methodology from each paper and saves a concise summary.
2. **Generates underlying hypotheses** — asks each model to infer the core hypothesis a paper's experiments were designed to test by only seeing the experiment summary but not the results or discussion.
3. **Generates novel hypotheses** — gives each model the full paper and asks it to produce a new, testable hypothesis that extends beyond what the paper found.

After generation, a separate visualization script embeds all outputs with `text-embedding-3-small` and produces intra-model and inter-model cosine similarity heatmaps, showing how repetitive or distinct each model's outputs are.

## File overview

```
extract_experiments_summary.py   # Stage 1: calls Claude directly via Anthropic SDK
generate_hypotheses.py           # Stage 2: underlying hypotheses via model_utils
generate_new_hypotheses.py       # Stage 3: novel hypotheses via model_utils
# DEPRECATED generate_testing_plans.py        # Stage 4: testing plans via model_utils
pipeline.py                      # Runs stages 2-3 in sequence (stage 1 is separate)
model_utils.py                   # Unified model interface, prompt caching for Claude
visualize_similarity.py          # Embedding + heatmap generation
```

## Setup

```bash
pip install anthropic langchain langchain-openai python-dotenv matplotlib seaborn numpy lxml
```

Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

## Data format
Papers should live in `data/train/<paper_id>.json` and follow the S2ORC schema, with `title`, and a `pdf_parse` object containing `abstract` and `body_text` arrays of paragraph objects.

## Running
Run stage 1 separately first, since it uses a different interface and produces intermediate files that the rest of the pipeline depends on:

```bash
python extract_experiments_summary.py
```

Then run the rest:
```bash
python pipeline.py
```

Or run individual stages:
```bash
python generate_hypotheses.py
python generate_new_hypotheses.py
python generate_testing_plans.py
```

To generate similarity plots after generation is complete:
```bash
python visualize_similarity.py                        # all three result types
python visualize_similarity.py new_hypotheses         # just one type
```

## Output structure
```
results/
  underlying_hypotheses/<paper_id>.json   # {model_id: [hyp_1, ..., hyp_N]}
  new_hypotheses/<paper_id>.json
  testing_plans/<paper_id>.json

embeddings/
  underlying_hypotheses/<paper_id>.json   # {model_id: [[float, ...], ...]}
  new_hypotheses/<paper_id>.json
  testing_plans/<paper_id>.json

plots/
  underlying_hypotheses/intra_model.png
  underlying_hypotheses/inter_model.png
  new_hypotheses/...
  testing_plans/...
```

## Adding or changing models
Edit the `MODELS` list in `model_utils.py`. Claude models (prefix `anthropic:`) automatically get prompt caching applied, which significantly reduces cost when the same paper is passed to the model N times. OpenAI models go through LangChain with no caching. Any model supported by LangChain's `init_chat_model` can be added as an OpenAI-style entry.