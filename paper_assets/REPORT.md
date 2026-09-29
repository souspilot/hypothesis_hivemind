# Hypothesis hivemind: generated numbers

## Coverage

Papers are analysed only if all 12 models have 10 embedded responses.

- AI4Mat / recover: 50 papers analysed
- AI4Mat / novel: 50 papers analysed
- PLOS Biology / recover: 52 papers analysed
- PLOS Biology / novel: 52 papers analysed

## Similarity summary (mean over papers, 95% bootstrap CI)

| Dataset | Task | Papers | Intra-model | Intra-provider | Cross-provider |
|---|---|---|---|---|---|
| AI4Mat | Recover underlying hypothesis | 50 | 0.860 [0.854, 0.867] | 0.808 [0.800, 0.816] | 0.773 [0.764, 0.783] |
| AI4Mat | Generate novel hypothesis | 50 | 0.720 [0.711, 0.730] | 0.660 [0.650, 0.670] | 0.638 [0.628, 0.647] |
| PLOS Biology | Recover underlying hypothesis | 52 | 0.878 [0.870, 0.887] | 0.826 [0.815, 0.837] | 0.808 [0.797, 0.820] |
| PLOS Biology | Generate novel hypothesis | 52 | 0.761 [0.750, 0.771] | 0.709 [0.699, 0.720] | 0.693 [0.682, 0.704] |

## Same paper vs different papers (Fig. 3 means)

| Dataset | Task | Same paper, same model | Same paper, different models | Different papers |
|---|---|---|---|---|
| AI4Mat | recover | 0.860 | 0.780 | 0.397 |
| AI4Mat | novel | 0.720 | 0.642 | 0.390 |
| PLOS Biology | recover | 0.878 | 0.812 | 0.270 |
| PLOS Biology | novel | 0.761 | 0.696 | 0.302 |

## Output length vs similarity (Pearson r across the 12 models)

| Dataset | Task | r(length, intra-model) | r(length, cross-model) |
|---|---|---|---|
| AI4Mat | recover | 0.48 | 0.60 |
| AI4Mat | novel | 0.74 | 0.32 |
| PLOS Biology | recover | 0.56 | 0.73 |
| PLOS Biology | novel | 0.68 | 0.27 |

## Diversity (Fig. 4)

Distinct ideas = Vendi score (cosine kernel) of 10 hypotheses. 1 model: each model's own 10 hypotheses for a paper. 10 models: one hypothesis each from 10 random models, same paper. 10 papers: 10 hypotheses about 10 different papers (scale reference). R^2 = PERMANOVA share of per-paper variation explained by model / provider (mean [95% CI] over papers).

| Dataset | Task | Ideas, 1 model | Ideas, 10 models | Ideas, 10 papers | Papers where 10 > 1 | R^2 model | R^2 provider | max per-paper p |
|---|---|---|---|---|---|---|---|---|
| AI4Mat | recover | 1.86 [1.82, 1.90] | 2.43 [2.36, 2.49] | 6.36 | 100% | 0.407 [0.393, 0.421] | 0.200 [0.187, 0.214] | 0.005 |
| AI4Mat | novel | 2.90 [2.83, 2.98] | 3.63 [3.54, 3.71] | 6.46 | 100% | 0.281 [0.271, 0.291] | 0.111 [0.106, 0.117] | 0.005 |
| PLOS Biology | recover | 1.75 [1.70, 1.81] | 2.22 [2.13, 2.30] | 7.72 | 100% | 0.400 [0.386, 0.414] | 0.164 [0.154, 0.174] | 0.005 |
| PLOS Biology | novel | 2.56 [2.48, 2.64] | 3.12 [3.03, 3.22] | 7.39 | 100% | 0.275 [0.265, 0.285] | 0.104 [0.097, 0.112] | 0.005 |

