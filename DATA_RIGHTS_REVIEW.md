# Data redistribution review

Reviewed on 9 October 2026. This review distinguishes evidence of redistribution permission from rights that cannot be established from the repository and public terms alone.

## Inventory and outcome

| Material | Shared files | Redistribution basis | Review outcome |
|---|---|---|---|
| PLOS Biology source articles | 57 XML and 114 extracted article JSON files | Article-level CC BY 4.0 declarations and PLOS policy | Supported with original attribution, licence notices, and extraction notices |
| PLOS experiment summaries | 57 JSON files | Article adaptation permitted by CC BY 4.0; generated-output rights subject to provider terms | Source-paper permission supported; retain PLOS attribution |
| AI4Mat experiment summaries | 50 JSON files | Generated summaries; source-paper expression retains its original rights | Not fully cleared: one copied-text candidate and source reuse licences not verified |
| Model hypotheses | 212 JSON files across both datasets and tasks | Generated outputs; public provider terms generally allocate output rights to customers or disclaim ownership | No general research-publication prohibition identified in the reviewed terms; actual routed-provider/account terms remain unverified |
| Study figures, tables, metrics, and metadata compilations | `paper_assets/`, `metadata/`, DOI lists | Study-author contributions; source-paper titles and links identify original works | Repo licences cover only rights held by the study authors |
| AI4Mat full texts, PLOS PDFs, embeddings, notebooks, and logs | Excluded from Git | Not part of this release | Exclusions verified for currently tracked files |

The PLOS counts include five skip-listed articles. Their inclusion as source files is separate from their exclusion from the numerical analysis.

## PLOS: affirmative redistribution evidence

Every one of the 57 publisher XML files contains a front-matter licence link to CC BY 4.0, a copyright year and holder, and an attribution notice. Both extracted JSON directories contain titles, DOIs, and author metadata for the same 57 IDs. The extracted JSON omits the licence fields, so [PLOS_ATTRIBUTION.md](PLOS_ATTRIBUTION.md) supplies full author lists, original copyright notices, source links, and licence notices for all records. The README and `data2/README.md` describe the extraction and point to that notice.

[PLOS's licensing policy](https://journals.plos.org/plosbiology/s/licenses-and-copyright) permits copying, redistribution, and modification, including commercial use, with author and source attribution. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) requires retention of supplied attribution, copyright and licence notices, and an indication of modifications. The repository includes the licence text.

Article captions and credit statements were screened for third-party rights exceptions. In `journal.pbio.3003730.xml`, silhouette credits identify CC BY 4.0 and CC0 material. In `journal.pbio.3003765.xml`, fractal credits identify CC BY 4.0 material. No conflicting restrictive licence notice was identified in the supplied XML. Original credits remain in the XML. The repository includes no article figure-image binaries, supplementary attachments, or embedded image data; XML references to them do not distribute those files. Their external datasets and attachments have not been licensed for redistribution by this review.

Conclusion: the included PLOS article XML and extracted text have an explicit redistribution basis, with the attribution and modification notices supplied here. Retain the notices when distributing subsets or individual records.

## AI4Mat: source-expression review remains open

No AI4Mat raw PDFs or full article JSON are tracked, and the earlier reachable-history scan found no full-text directories. The metadata compilation contains only paper IDs, titles, and source links.

The 50 AI4Mat summaries and all AI4Mat hypothesis responses were compared with the locally available source article text for long contiguous word sequences. One summary, `data/experiments_summary/12ZCZVKm7r.json`, contains a 26-word match. No other sample reached the screening threshold of 25 consecutive words. This threshold is a search heuristic, not a legal safe harbour; shorter copying and paraphrasing can also require contextual assessment.

That source is *Universally Converging Representations of Matter Across Scientific Foundation Models*. Its [arXiv record](https://arxiv.org/abs/2512.03750) links to [arXiv's non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html), which authorizes arXiv's distribution and is not a general CC BY reuse grant. OpenReview's forum pages required browser verification and its public note API returned HTTP 403, so a separate reuse licence for the workshop version could not be established. No blanket claim is made that all AI4Mat papers lack open licences.

Scientific facts and ideas are distinct from protected source expression, but this automated comparison does not determine the copyright status or lawful use of any particular passage. Before treating the AI4Mat summaries as fully cleared for public redistribution, resolve the flagged passage through an applicable source licence, author permission, or a contextual legal assessment. Omitting source-derived summaries from the public release is another option. Do not silently rewrite the stored experimental inputs: doing so would alter provenance relative to the existing model responses. No source data was changed during this review.

## Generated outputs: reviewed terms and limits

The relevant public terms checked were the API/business terms rather than consumer-chat terms:

- [OpenAI Services Agreement, §4.1](https://openai.com/policies/services-agreement/) assigns OpenAI's output rights to the customer to the extent permitted by law. This does not establish rights in third-party material reproduced in an output.
- [Anthropic Commercial Terms, §B](https://www.anthropic.com/legal/commercial-terms) similarly allocate output ownership to the customer. Other clauses restrict competitive uses and require appropriate review before sharing.
- [Gemini API terms, “Use of Generated Content”](https://ai.google.dev/gemini-api/terms) say Google does not claim ownership of generated content and leave responsibility for its use with the user. This page alone does not establish the contract of every Google endpoint routed through OpenRouter. [Gemma 4's published model licence](https://ai.google.dev/gemma/apache_2) is Apache 2.0; model-weight licensing is distinct from publication of generated responses.
- [Kimi OpenPlatform terms, §4](https://platform.kimi.ai/docs/agreement/modeluse) disclaim ownership of input and output content and place responsibility on the customer. This does not establish the contract of every third-party hosted Kimi endpoint.
- [OpenRouter terms, §§5 and 6.1](https://openrouter.ai/terms) make the applicable model terms govern output ownership. OpenRouter does not waive model/provider terms. Its [routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection) describes selection among multiple providers.

These provisions support an inference that ordinary publication of research responses is generally permitted under standard terms; they are not a universal redistribution warranty. The saved response JSON identifies models but does not record the actual hosting provider for each OpenRouter request or the contractual terms accepted for that account and date. Private institutional agreements and employer ownership rules are also unavailable here. Consequently, those account-specific rights are not certified by this review. Permission to publish research responses also does not establish permission for every downstream use, such as training competing models.

The repository's CC BY grant covers only rights the publication authors hold. It does not relicense source papers, supply missing permissions, or override separate contractual obligations.

## Remaining release checks

1. Resolve the flagged AI4Mat source-expression issue before treating all summaries as cleared.
2. Confirm that the actual API accounts and routed providers had standard terms allowing research-output publication, without a conflicting private agreement.
3. Include `LICENSING.md`, `PLOS_ATTRIBUTION.md`, `data2/README.md`, and the CC BY licence text with the release.
