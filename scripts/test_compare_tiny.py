import json
import unittest
from unittest.mock import patch

from compare_tiny import audiovisual, av_eval


class HeldoutTest(unittest.TestCase):
    def test_each_single_modality_is_balanced_within_a_scene(self):
        rows = audiovisual(3100)
        for modality in ("image", "audio"):
            pairs = {}
            for row in rows:
                pairs.setdefault(tuple(row[modality]), []).append(row["answer"])
            self.assertEqual(len(pairs), 2)
            for answers in pairs.values():
                self.assertCountEqual(answers, ["yes", "no"])

    def test_exact_match_preserves_whitespace_and_replacement_characters(self):
        rows = audiovisual(3100)
        output = "\n".join(json.dumps(dict(question="Match?", answer=a, token_ids=[42, 2]))
                           for a in (" yes", "no", "�", "yes"))
        with patch("compare_tiny.subprocess.check_output", return_value=output):
            scores = av_eval("metai", "weights", "inputs.jsonl", rows, 6, 2)
        self.assertEqual(scores["correct"], 2)
        self.assertEqual(scores["replacement"], 1)
        self.assertEqual(scores["quartets_correct"], 0)
