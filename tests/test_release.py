"""Regression checks for metadata, preparation, and cache invalidation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analysis
import config
import embed
from config import DATASETS, MODELS, TASKS
from embedding_cache import fingerprint, fingerprints_path
from prepare_plos import prepare


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        root_patch = patch.object(config, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_metadata_without_article_text_and_local_fallback(self):
        dataset = DATASETS["ai4mat"]
        row = {"id": "paper", "title": "Paper title", "url": "https://openreview.net/pdf?id=paper"}
        self.write_json(dataset.metadata_path, [row])
        self.assertFalse(dataset.papers_dir.exists())
        self.assertEqual(analysis.load_paper_metadata(dataset, {"paper"}), [row])
        self.write_json(dataset.papers_dir / "extra.json", {"title": "Extra paper"})
        extra = analysis.load_paper_metadata(dataset, {"extra"})[0]
        self.assertEqual(extra["url"], "https://openreview.net/pdf?id=extra")

    def test_cache_reuse_and_same_count_edit(self):
        dataset, task = DATASETS["ai4mat"], TASKS["novel"]
        key = MODELS[0].legacy_keys[0]
        result = dataset.results_dir(task) / "paper.json"
        cache = dataset.embeddings_dir(task) / "paper.json"
        self.write_json(result, {key: ["First hypothesis", "Second hypothesis"]})
        self.write_json(cache, {key: [[1, 0], [0, 1]]})
        self.assertIn(cache, embed.plan(dataset, task))  # Legacy caches lack fingerprints.

        class Embedder:
            def __init__(self):
                self.calls = 0

            def embed_documents(self, samples):
                self.calls += 1
                return [[1, 0] for _ in samples]

        client = Embedder()
        embed.run(dataset, task, client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(embed.plan(dataset, task), {})
        embed.run(dataset, task, client)
        self.assertEqual(client.calls, 1)
        self.write_json(result, {key: ["Changed hypothesis", "Second hypothesis"]})
        self.assertIn(cache, embed.plan(dataset, task))
        embed.run(dataset, task, client)
        self.assertEqual(client.calls, 2)
        self.assertEqual(embed.plan(dataset, task), {})

    def test_analysis_rejects_changed_fingerprinted_responses(self):
        dataset, task = DATASETS["ai4mat"], TASKS["recover"]
        samples = ["A hypothesis"] * config.N_SAMPLES
        vector = [1.0] + [0.0] * 1535
        results = {model.slug: samples for model in MODELS}
        cache = dataset.embeddings_dir(task) / "paper.json"
        self.write_json(dataset.results_dir(task) / "paper.json", results)
        self.write_json(cache, {model.slug: [vector] * config.N_SAMPLES for model in MODELS})
        with patch.object(analysis, "MIN_PAPERS", 1):
            corpus, excluded = analysis.load_corpus(dataset, task)
            self.assertEqual(corpus.paper_ids, ["paper"])  # Original study caches remain readable.
            self.assertEqual(excluded, {})
            self.write_json(fingerprints_path(cache), {model.slug: fingerprint(samples) for model in MODELS})
            results[MODELS[0].slug] = ["Edited hypothesis"] + samples[1:]
            self.write_json(dataset.results_dir(task) / "paper.json", results)
            with self.assertRaisesRegex(analysis.DataIntegrityError, "response text changed"):
                analysis.load_corpus(dataset, task)

    def test_plos_selection_dry_run_and_conflict(self):
        dataset = DATASETS["plos"]
        rows = [{"id": pid, "title": pid, "url": "https://example.org"} for pid in ("a", "b", "skip")]
        self.write_json(dataset.metadata_path, rows)
        # The real skip-list path is fixed in the dataset object, so supply a temporary dataset.
        dataset = config.Dataset("plos", "PLOS Biology", "Test", "2", self.root / "skip.txt")
        dataset.skip_list.write_text("skip\n")
        source, target = self.root / "processed", self.root / "train"
        record = {"title": "Article", "pdf_parse": {"body_text": [{"text": "Methods"}]}}
        for pid in ("a", "b"):
            self.write_json(source / f"{pid}.json", record)
        with patch.dict(DATASETS, {"plos": dataset}):
            self.assertEqual(prepare(source, target, check=True), 2)
            self.assertFalse(target.exists())
            self.assertEqual(prepare(source, target), 2)
            self.assertEqual(prepare(source, target), 0)
            self.assertEqual(sorted(p.name for p in target.glob("*.json")), ["a.json", "b.json"])
            self.write_json(target / "b.json", {"title": "Changed"})
            original = (target / "b.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare(source, target)
            self.assertEqual((target / "b.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
