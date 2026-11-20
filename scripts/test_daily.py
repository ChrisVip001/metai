import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from daily_translation import ARMS, FREEZE, prepare
from report_daily import STAGES, score, worksheet
from score_bilingual import read_jsonl


class DailyChecks(unittest.TestCase):
    def test_frozen_splits_equal_steps_and_bidirectionality(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(prepare(root), json.loads(FREEZE.read_text()))
            data = root / "data"
            all_q = []
            for split in ("heldout", "reserve"):
                questions = read_jsonl(data / f"{split}-questions.jsonl")
                self.assertTrue(
                    all(set(q) == {"id", "instruction", "input"} for q in questions)
                )
                all_q.extend(q["input"] for q in questions)
            self.assertEqual(len(set(all_q)), 64)
            for arm, unique in zip(ARMS, (384, 3072, 3072)):
                rows = read_jsonl(data / f"{arm}.jsonl")
                pairs = Counter((r["input"], r["output"]) for r in rows)
                self.assertEqual(len(rows) // 16 * 4, 1536)
                self.assertEqual(len(pairs), unique * 2)
                for (a, b), n in pairs.items():
                    self.assertEqual(pairs[b, a], n)
                self.assertFalse(set(all_q) & {r["input"] for r in rows})

    def test_blind_partial_credit_and_unreviewed_answers(self):
        r = dict(
            id="42-en-zh",
            input="I can't swim.",
            output="我不会游泳。",
            criterion="无法游泳",
            group="cannot",
            primary=True,
            pair="42",
            direction="en-zh",
            predictions={s: dict(answer="我不游泳。") for s in STAGES},
        )
        r["predictions"]["focused"] = dict(answer="今天我不会游泳。")
        sheet = worksheet([r])
        self.assertEqual(len(sheet), 2)
        self.assertNotIn("predictions", sheet[0])
        with self.assertRaisesRegex(ValueError, "cover"):
            score([r], [])
        for x in sheet:
            x.update(
                meaning=False,
                detail=x["answer"].startswith("今天"),
                reason="额外加了今天" if x["answer"].startswith("今天") else "丢失能力",
            )
        stats = score([r], sheet)
        self.assertEqual(
            stats["focused"]["primary"], dict(n=1, meaning=0, detail=1, exact=0)
        )
        self.assertEqual(stats["before"]["primary"]["detail"], 0)
