import argparse
from pathlib import Path
from typing import List

from utils.params import load_params

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk

from docling_core.types import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel


def write_chunks_for_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    split_by: str,
) -> None:
    chunker = HierarchicalChunker()

    doc = DoclingDocument.load_from_json(input_file)

    stem = input_file.stem
    artifacts_dir = input_file.parent / f"{stem}_artifacts"
    image_paths: List[str] = []
    if artifacts_dir.exists() and artifacts_dir.is_dir():
        for img_path in sorted(artifacts_dir.glob("image_*.png")):
            # store relative path (to the JSON file directory) instead of absolute
            rel_path = img_path.relative_to(input_file.parent.parent.parent.parent)
            image_paths.append(str(rel_path))

    for idx, raw_chunk in enumerate(chunker.chunk(doc)):
        doc_chunk = DocChunk.model_validate(raw_chunk)
        has_picture = any(
            it.label == DocItemLabel.PICTURE for it in doc_chunk.meta.doc_items
        )

        text = raw_chunk.text

        if has_picture and image_paths:
            chunk_with_images = f"{text}\n\n[IMAGES]\n" + "\n".join(image_paths)
        else:
            chunk_with_images = text

        out_path = output_dir / f"{stem}_chunk_{idx}.txt"
        out_path.write_text(chunk_with_images, encoding="utf-8")


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
