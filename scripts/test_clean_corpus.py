import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from clean_corpus import prepare


class CorpusTest(unittest.TestCase):
    def test_normalized_duplicates_share_one_split(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw"
            source.write_text("café ok\ncafe\u0301   ok\nvalid line\n\n", encoding="utf-8")
            self.assertEqual(prepare(source, root / "a"),
                             dict(train=1, valid=1, filtered=1, duplicates=1))
            prepare(source, root / "b")
            manifest = (root / "a/manifest.jsonl").read_bytes()
            self.assertEqual(manifest, (root / "b/manifest.jsonl").read_bytes())
            rows = [json.loads(line) for line in manifest.splitlines()]
            self.assertIn([1, 2], [row["lines"] for row in rows])
            self.assertNotEqual(rows[0]["sha256"], rows[1]["sha256"])
            with self.assertRaises(FileExistsError):
                prepare(source, root / "a")

    def test_insufficient_distinct_text_publishes_no_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "raw").write_text("hello\nhello\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(root / "raw", root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
