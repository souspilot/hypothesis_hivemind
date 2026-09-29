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

## Embedding variance vs number of models (Fig. 4)

Variance = 1 - |centroid|^2 of the pooled outputs, averaged over 1,000 random model orders. 'Independent' = each model keeps its within-model spread but is only as similar to other models as outputs for different papers are.

| Dataset | Task | 1 model | 12 models | 12 independent models | Equivalent independent models [95% CI] | Share of 12-model variance in 1 model |
|---|---|---|---|---|---|---|
| AI4Mat | recover | 0.126 | 0.212 | 0.563 | 1.22 [1.21, 1.24] | 59% |
| AI4Mat | novel | 0.251 | 0.349 | 0.580 | 1.37 [1.35, 1.40] | 72% |
| PLOS Biology | recover | 0.110 | 0.182 | 0.678 | 1.13 [1.12, 1.14] | 60% |
| PLOS Biology | novel | 0.215 | 0.297 | 0.657 | 1.20 [1.19, 1.21] | 73% |

