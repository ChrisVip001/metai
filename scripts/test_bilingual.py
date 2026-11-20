import gzip
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prepare_bilingual_v2 import prepare as prepare_v2
from prepare_bilingual import TRANSFER, from_subset, key
from report_bilingual_v2 import translation_scores
from score_bilingual import normalized, summary

REPO = Path(__file__).resolve().parents[1]


class BilingualBenchmark(unittest.TestCase):
    def test_frozen_data_and_no_cross_split_sentence(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "data"
            manifest = from_subset(
                REPO / "examples/m2-bilingual-v1/subset.jsonl.gz", out
            )
            self.assertEqual(
                manifest["files"]["questions.jsonl"],
                "7d419c72f473b17dc78d634e0ad3f8a900779b69a5cd1c243af10ed8e32c18cd",
            )
            for name, digest in manifest["files"].items():
                self.assertEqual(
                    hashlib.sha256((out / name).read_bytes()).hexdigest(), digest
                )
            rows = [
                json.loads(s)
                for s in (out / "attribution.jsonl").read_text().splitlines()
            ]
            seen = {key(q) for _, q, _ in TRANSFER}
            ids = set()
            for row in rows:
                for lang in ("zh", "en"):
                    sentence = key(row[lang])
                    self.assertNotIn(sentence, seen)
                    seen.add(sentence)
                    self.assertNotIn(row[lang + "_id"], ids)
                    ids.add(row[lang + "_id"])
                    self.assertTrue(row[lang + "_author"])
            self.assertEqual(Counter(r["split"] for r in rows), manifest["split_pairs"])
            questions = [
                json.loads(s)
                for s in (out / "questions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({q["id"] for q in questions}), 40)
            self.assertTrue(
                all(set(q) == {"id", "instruction", "input"} for q in questions)
            )

    def test_v2_excludes_all_v1_sentences_and_preserves_both_directions(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "data"
            manifest = prepare_v2(
                REPO / "examples/m2-bilingual-v2/subset.jsonl.gz", out
            )
            old = gzip.decompress(
                (REPO / "examples/m2-bilingual-v1/subset.jsonl.gz").read_bytes()
            )
            ids, sentences = set(), set()
            for line in [
                *old.splitlines(),
                *(out / "attribution.jsonl").read_bytes().splitlines(),
            ]:
                row = json.loads(line)
                for lang in ("en", "zh"):
                    index, sentence = row[lang + "_id"], key(row[lang])
                    self.assertNotIn(index, ids)
                    self.assertNotIn(sentence, sentences)
                    ids.add(index)
                    sentences.add(sentence)
            self.assertEqual(
                manifest["files"]["questions.jsonl"],
                "2a01d4ed09d3c2f56c432d65bc6e1e80ca2c92d29d04c83871f45097de1cbd0e",
            )
            questions = [
                json.loads(s)
                for s in (out / "questions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({r["id"] for r in questions}), 56)
            self.assertTrue(
                all(set(q) == {"instruction", "input", "id"} for q in questions)
            )
            training = [
                json.loads(s) for s in (out / "sft.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                sorted(Counter(r["instruction"] for r in training).values()),
                [768, 3072, 3072],
            )
            refs = [
                (r["input"], r["output"])
                for r in training
                if r["instruction"].startswith("Translate")
            ]
            forwards = {
                (r["output"], r["input"])
                for r in training
                if r["instruction"].startswith("把")
            }
            self.assertEqual(set(refs), forwards)

    def test_constant_translation_scores_equally_with_rotated_references(self):
        rows = [
            dict(output=s, model={"answer": "blue box"})
            for s in ["blue box", "red bag", "red hat"]
        ]
        constant = translation_scores(rows, "model")
        self.assertEqual(constant["distinct_answers"], 1)
        self.assertAlmostEqual(
            constant["aligned_bigram_f1"], constant["rotated_bigram_f1"]
        )
        for r in rows:
            r["model"]["answer"] = r["output"]
        aligned = translation_scores(rows, "model")
        self.assertEqual(aligned["aligned_bigram_f1"], 1.0)
        self.assertLess(aligned["rotated_bigram_f1"], 1.0)

    def test_strict_metrics_preserve_extra_text_and_clip_repetitions(self):
        pairs = [
            ("zh-en", "aaaa", "aaa"),
            ("lang-en", " 中文 ", "英文"),
            ("lang-zh", "中文。", "中文"),
        ]
        rows = [
            dict(
                kind=k,
                output=g,
                model=dict(answer=a, stop="eos", exact=normalized(a) == normalized(g)),
            )
            for k, a, g in pairs
        ]
        result = summary(rows, "model")
        self.assertEqual(result["language"], {"exact": 0, "n": 2})
        self.assertAlmostEqual(result["zh-en"]["char_bigram_micro_f1"], 4 / 5)
