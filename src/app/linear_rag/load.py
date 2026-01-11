import argparse
from pathlib import Path
import sys
import hashlib
import logging
from typing import Tuple, Optional

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app

from utils.params_helper import load_params

import psycopg2
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values

import mlflow
from utils.mlflow_helper import mlflow_connect
from utils.mlflow_helper import get_mlflow_experiment_name
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_hash_id(text: str, prefix: str) -> str:
    """Compute MD5-based hash ID with namespace prefix (matches LinearRAG convention)."""
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()


def _connect(database_url: str) -> PGConnection:
    """Connect to Postgres database."""
    return psycopg2.connect(database_url)


def _parse_chunk_filename(filename: str) -> Tuple[str, int]:
    """
    Parse chunk filename to extract title and page reference.

    Expected format: {title}_chunk_{idx}.txt

    Returns:
        (title, page_ref) tuple
    """
    stem = filename.replace(".txt", "")
    parts = stem.rsplit("_chunk_", 1)

    if len(parts) == 2:
        title = parts[0]
        page_ref = int(parts[1]) if parts[1].isdigit() else 0
    else:
        title = stem
        page_ref = 0

    return title, page_ref


def _resolve_pic_ref(content: str, title: str, artifacts_dir: Path) -> Optional[str]:
    """
    Resolve image references from chunk content.

    Chunks may contain [IMAGES] section with artifact references.
    This function extracts those references for storage.

    Returns:
        JSON string with picture references or None
    """
    if "[IMAGES]" not in content:
        return None

    # Extract image section
    parts = content.split("[IMAGES]")
    if len(parts) < 2:
        return None

    image_section = parts[1].strip()
    if not image_section:
        return None

    # Parse image references (format: artifact_name.png)
    # Store as JSON array for compatibility with document_chunk schema
    import json

    image_refs = [line.strip() for line in image_section.split("\n") if line.strip()]

    return json.dumps(image_refs) if image_refs else None


def load_chunks(
    chunk_dir: Path, artifacts_dir: Path, conn: PGConnection, batch_size: int = 100
) -> int:
    """
    Load chunk .txt files into document_chunk table.

    This intermediate stage between chunk and index allows:
    - Clean separation: chunk.py creates files, load.py inserts DB records
    - Index.py can then focus purely on graph analysis from database

    Args:
        chunk_dir: Directory containing {title}_chunk_{idx}.txt files
        artifacts_dir: Directory with extraction artifacts (for pic_ref resolution)
        conn: PostgreSQL connection
        batch_size: Number of records to insert per batch

    Returns:
        Number of chunks loaded
    """
    chunk_files = sorted(chunk_dir.glob("*.txt"))

    if not chunk_files:
        logger.warning(f"No .txt chunk files found in {chunk_dir}")
        return 0

    logger.info(f"Found {len(chunk_files)} chunk files to load")

    # Prepare batch data for insertion
    chunk_records = []

    for chunk_file in chunk_files:
        content = chunk_file.read_text(encoding="utf-8")

        # Parse filename for title and page_ref
        title, page_ref = _parse_chunk_filename(chunk_file.name)

        # Remove image references from passage text for hashing
        passage_text = content
        if "[IMAGES]" in content:
            passage_text = content.split("[IMAGES]")[0].strip()

        # Compute passage hash (will be used by index.py for graph building)
        chunk_hash_id = compute_hash_id(passage_text, "passage-")

        # Resolve picture references (if any)
        pic_ref = _resolve_pic_ref(content, title, artifacts_dir)

        chunk_records.append((title, page_ref, content, chunk_hash_id, pic_ref))

    # Batch insert into document_chunk table
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO document_chunk (title, page_ref, content, chunk_hash_id, pic_ref)
            VALUES %s
            """,
            chunk_records,
            page_size=batch_size,
        )

    conn.commit()
    logger.info(f"Loaded {len(chunk_records)} chunks into document_chunk table")

    # Log statistics to MLflow
    mlflow.log_metric("chunks_loaded", len(chunk_records))

    return len(chunk_records)


def main(
    input_dir: Path, artifacts_dir: Path, database_url: str, batch_size: int
) -> None:
    """
    Main entry point for LinearRAG load stage.

    Loads chunk .txt files into document_chunk table in Supabase.
    """
    logger.info(f"Connecting to database: {database_url}")
    conn = _connect(database_url)

    try:
        chunks_loaded = load_chunks(input_dir, artifacts_dir, conn, batch_size)
        logger.info(
            f"LinearRAG load completed successfully: {chunks_loaded} chunks loaded"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LinearRAG load stage: load chunks into document_chunk table"
    )

    parser.add_argument(
        "--input", type=Path, required=True, help="Directory with chunk .txt files"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Directory with extraction artifacts (for pic_ref resolution)",
    )
    parser.add_argument(
        "--database_url", type=str, required=True, help="PostgreSQL database URL"
    )

    args = parser.parse_args()

    # Load parameters
    load_params_dict = load_params("load")
    batch_size = int(load_params_dict.get("batch_size", 100))

    # MLflow tracking
    mlflow_connect()
    experiment_name = get_mlflow_experiment_name()
    mlflow.set_experiment(experiment_name)

    mlflow.log_param("batch_size", batch_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"linearrag_load_{timestamp}"

    # End any active run before starting new one (DVC context issue)
    mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        main(
            args.input.resolve(),
            args.artifacts.resolve(),
            args.database_url,
            batch_size,
        )
