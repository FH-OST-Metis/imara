from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from utils.params import load_params


def read_doctags(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_doctags_into_sections(text: str) -> List[str]:
    lines = text.splitlines(keepends=True)
    sections: List[str] = []
    current: List[str] = []

    def is_header(line: str) -> bool:
        return "<page_header>" in line or "<section_header_level_" in line

    for line in lines:
        if is_header(line):
            if current:
                sections.append("".join(current))
                current = []
        current.append(line)

    if current:
        sections.append("".join(current))

    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step

    return chunks


def write_chunks_for_file(
    doctags_file: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    split_by: str,
) -> None:
    text = read_doctags(doctags_file)

    if split_by == "section":
        chunks: List[str] = split_doctags_into_sections(text)
    else:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    stem = doctags_file.stem
    for idx, chunk in enumerate(chunks):
        out_path = output_dir / f"{stem}_chunk_{idx}.txt"
        out_path.write_text(chunk, encoding="utf-8")

    print(f"{doctags_file.name}: wrote {len(chunks)} chunks to {output_dir}")


def main(input_dir: Path, output_dir: Path) -> None:
    chunk_params = load_params("chunk")

    chunk_size: int = int(chunk_params.get("chunk_size", 500))
    overlap: int = int(chunk_params.get("overlap", 100))
    split_by: str = str(chunk_params.get("split_by", "section"))

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.doctags"))
    if not files:
        print(f"No .doctags files found in {input_dir}.")
        return

    for input_file in files:
        write_chunks_for_file(input_file, output_dir, chunk_size, overlap, split_by)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main(args.input, args.output)
