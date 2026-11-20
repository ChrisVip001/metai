import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prepare_translation import ARMS, DIRECTIONS, cubes, prepare, sentence
from report_translation import FREEZE, contrast, grammar, judge
from score_bilingual import read_jsonl


class TranslationTests(unittest.TestCase):
    def test_factorial_and_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            prepare(out)
            self.assertEqual(
                json.loads((out / "manifest.json").read_text()),
                json.loads(FREEZE.read_text()),
            )
            questions = read_jsonl(out / "heldout-questions.jsonl")
            self.assertEqual(len(questions), 48)
            self.assertEqual(len({q["input"] for q in questions}), 48)
            self.assertTrue(
                all(set(q) == {"id", "instruction", "input"} for q in questions)
            )
            table = grammar()
            counts = {}
            heldout = {(a, o) for _, a, o, _ in cubes()}
            for name, combinations in zip(ARMS, [6, 18, 6, 18]):
                records = read_jsonl(out / f"{name}.jsonl")
                coords = []
                for r in records:
                    for direction in DIRECTIONS:
                        idx = table.get(
                            (direction, r["output"].casefold().rstrip(".。"))
                        )
                        if idx is not None:
                            coords.append(idx)
                # All arms see the same univariate frequencies and 64 identical natural rows.
                self.assertEqual((len(records), len(coords)), (352, 288))
                pairs = {(s[1], s[2]) for s in coords}
                self.assertEqual(len(pairs), combinations)
                self.assertFalse(pairs & heldout)
                counts[name] = [Counter(x[i] for x in coords) for i in range(4)]
            self.assertTrue(all(c == counts["fixed"] for c in counts.values()))

    def test_scoring_distinguishes_slots_and_requires_semantic_review(self):
        table = grammar()
        slots = (2, 0, 3, 1)
        ref = dict(
            id="example", direction="en-zh", slots=slots, output=sentence(*slots)["zh"]
        )
        res = judge(ref, sentence(2, 1, 3, 1)["zh"], table, {})
        self.assertFalse(res["meaning"])
        self.assertEqual(
            res["slots"], dict(subject=True, action=False, object=True, polarity=True)
        )
        with self.assertRaisesRegex(ValueError, "needs semantic review"):
            judge(ref, "我买了一盒。", table, {})
        paired = []
        for a, n in itertools.product((0, 1), (0, 1)):
            paired.append(
                dict(
                    direction="en-zh",
                    slots=[2, a, 3, n],
                    predictions={
                        "test": {
                            "answer": sentence(2, 0, 3, n)["zh"],
                            "meaning": a == 0,
                            "slots": {"action": a == 0, "polarity": True},
                        }
                    },
                )
            )
        self.assertEqual(contrast(paired, "test", 1)["slot_both_correct"], 0)
        self.assertEqual(contrast(paired, "test", 3)["slot_both_correct"], 2)


if __name__ == "__main__":
    unittest.main()
