import argparse
from pathlib import Path
from typing import List

from utils.params import load_params

from docling_core.transforms.chunker import HierarchicalChunker

from docling_core.types import DoclingDocument


def write_chunks_for_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    split_by: str,
) -> None:
    chunker = HierarchicalChunker()

    doc = DoclingDocument.load_from_json(input_file)

    chunks = [chunk.text for chunk in chunker.chunk(doc)]

    stem = input_file.stem
    for idx, chunk in enumerate(chunks):
        out_path = output_dir / f"{stem}_chunk_{idx}.txt"
        out_path.write_text(chunk, encoding="utf-8")


def main(input_dir: Path, output_dir: Path) -> None:
    chunk_params = load_params("chunk")

    chunk_size: int = int(chunk_params.get("chunk_size", 500))
    overlap: int = int(chunk_params.get("overlap", 100))
    split_by: str = str(chunk_params.get("split_by", "section"))

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"No .doctags files found in {input_dir}.")
        return

    for input_file in files:
        write_chunks_for_file(
            input_file.resolve(), output_dir.resolve(), chunk_size, overlap, split_by
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    main(args.input.resolve(), args.output.resolve())
