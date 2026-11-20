"""Normalize a UTF-8 line corpus, deduplicate before splitting, write provenance."""
import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path


def prepare(source, output, seed=42, valid_fraction=0.2):
    if not 0 < valid_fraction < 1:
        raise ValueError("valid fraction must be between zero and one")
    if output.exists():
        raise FileExistsError(output)
    unique = {}
    skipped = 0
    with source.open(encoding="utf-8") as reader:
        for line_number, raw in enumerate(reader, 1):
            text = " ".join(unicodedata.normalize("NFC", raw).split())
            if len(text) < 2 or any(unicodedata.category(c) == "Cc" for c in text):
                skipped += 1
                continue
            unique.setdefault(text, []).append(line_number)
    if len(unique) < 2:
        raise ValueError("need two distinct usable lines")
    def key(text):
        return hashlib.sha256(f"{seed}\0{text}".encode()).digest()
    ordered = sorted(unique, key=key)
    n_valid = max(1, min(len(ordered) - 1, round(len(ordered) * valid_fraction)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp) / "corpus"
        root.mkdir()
        with (root / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for split, texts in (("valid", ordered[:n_valid]), ("train", ordered[n_valid:])):
                (root / f"{split}.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
                for text in texts:
                    print(json.dumps(dict(source=source.name, lines=unique[text], split=split,
                                          seed=seed, sha256=hashlib.sha256(text.encode()).hexdigest())), file=manifest)
        os.rename(root, output)
    return dict(train=len(ordered)-n_valid, valid=n_valid, filtered=skipped,
                duplicates=sum(len(lines)-1 for lines in unique.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.seed, args.valid_fraction))
