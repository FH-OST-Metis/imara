from __future__ import annotations

from pathlib import Path
import json
from typing import List


def get_corpus_from_chunks(chunk_dir: Path) -> List[str]:
    """Load text corpus from chunk files produced by this pipeline.

    Each .txt file in ``chunk_dir`` is treated as one text unit.
    """
    if not chunk_dir.exists() or not chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    corpus: List[str] = []
    for path in sorted(chunk_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            corpus.append(text)

    if not corpus:
        raise RuntimeError(f"No non-empty .txt chunk files found in {chunk_dir}")

    print(f"Loaded {len(corpus)} text chunks from {chunk_dir}")

    return corpus


def main() -> None:
    chunk_dir = Path("data/processed/chunks")
    corpus = get_corpus_from_chunks(chunk_dir)

    output_dir = Path("data/graphmert/corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.json"

    with corpus_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved corpus with {len(corpus)} text chunks to {corpus_path}")


if __name__ == "__main__":
    main()
