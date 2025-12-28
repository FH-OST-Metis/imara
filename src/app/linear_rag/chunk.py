import argparse
from pathlib import Path
from typing import List
import sys
import logging

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app

from utils.params_helper import load_params

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk

from docling_core.types import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

import mlflow
from utils.mlflow_helper import mlflow_connect
from utils.mlflow_helper import get_mlflow_experiment_name
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

files_count = 0
overall_chunks_count = 0


def write_chunks_for_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    split_by: str,
) -> None:
    """
    Process a single JSON file from Docling extraction and create text chunk files.
    
    Each chunk is saved as {stem}_chunk_{idx}.txt with optional image references.
    """
    global files_count, overall_chunks_count

    chunker = HierarchicalChunker()

    doc = DoclingDocument.load_from_json(input_file)

    stem = input_file.stem
    mlflow.log_metric("bytes_processed", input_file.stat().st_size, step=files_count)
    artifacts_dir = input_file.parent / f"{stem}_artifacts"
    image_paths: List[str] = []
    if artifacts_dir.exists() and artifacts_dir.is_dir():
        for img_path in sorted(artifacts_dir.glob("image_*.png")):
            # store relative path (to the JSON file directory) instead of absolute
            rel_path = img_path.relative_to(input_file.parent.parent.parent.parent)
            image_paths.append(str(rel_path))

    chunks_count = 0
    for idx, raw_chunk in enumerate(chunker.chunk(doc)):
        chunks_count += 1
        overall_chunks_count += 1
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

    mlflow.log_metric("chunks_created_in_step", chunks_count, step=files_count)
    mlflow.log_metric("overall_chunks_count", overall_chunks_count, step=files_count)


def main(input_dir: Path, output_dir: Path) -> None:
    """
    Main entry point for LinearRAG chunking stage.
    
    Reads DoclingDocument JSON files from input_dir and creates text chunk files
    in output_dir for subsequent graph indexing.
    """
    global files_count

    chunk_params = load_params("chunk")

    chunk_size: int = int(chunk_params.get("chunk_size", 500))
    mlflow.log_param("chunk_size", chunk_size)
    overlap: int = int(chunk_params.get("overlap", 100))
    mlflow.log_param("overlap", overlap)
    split_by: str = str(chunk_params.get("split_by", "section"))
    mlflow.log_param("split_by", split_by)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        logger.warning(f"No .json files found in {input_dir}.")
        return

    for input_file in files:
        files_count += 1
        write_chunks_for_file(
            input_file.resolve(), output_dir.resolve(), chunk_size, overlap, split_by
        )

    logger.info(f"Processed {files_count} documents into {overall_chunks_count} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LinearRAG chunking stage: split documents into passages"
    )

    parser.add_argument("--input", type=Path, required=True,
                       help="Directory with extracted JSON files")
    parser.add_argument("--output", type=Path, required=True,
                       help="Directory to write chunk .txt files")

    args = parser.parse_args()

    mlflow_connect()
    experiment_name = get_mlflow_experiment_name()
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"linearrag_chunk_{timestamp}"

    # End any active run before starting new one (DVC context issue)
    mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        main(args.input.resolve(), args.output.resolve())
