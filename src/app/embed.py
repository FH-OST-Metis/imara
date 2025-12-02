from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import psycopg2
from psycopg2.extensions import connection as PGConnection


def _parse_chunk_filename(path: Path) -> Tuple[str, int, Optional[str]]:
    stem = path.stem
    if "_chunk_" in stem:
        base, idx_str = stem.rsplit("_chunk_", 1)
        try:
            page_ref = int(idx_str)
        except ValueError:
            page_ref = 0
        title = base
    else:
        title = stem
        page_ref = 0

    pic_ref: Optional[str] = None
    return title, page_ref, pic_ref


def _resolve_pic_ref(title: str, artifacts_root: Optional[Path]) -> List[str]:
    if artifacts_root is None:
        return []

    artifacts_dir = artifacts_root / f"{title}_artifacts"
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return []

    pngs = sorted(artifacts_dir.glob("image_*.png"))

    return [str(p) for p in pngs]


def _iter_chunks(
    chunk_dir: Path,
    artifacts_root: Optional[Path] = None,
) -> Iterable[Tuple[str, int, Optional[str], str]]:
    if not chunk_dir.exists() or not chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    for path in sorted(chunk_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        title, page_ref, _ = _parse_chunk_filename(path)
        pic_paths = _resolve_pic_ref(title, artifacts_root)
        pic_ref: Optional[str] = json.dumps(pic_paths) if pic_paths else None
        yield title, page_ref, pic_ref, content


def _connect(database_url: str) -> PGConnection:
    return psycopg2.connect(database_url)


def main(input_dir: Path, artifacts_dir: Path, database_url: str) -> None:
    conn = _connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            for title, page_ref, pic_ref, content in _iter_chunks(
                input_dir, artifacts_dir
            ):
                cur.execute(
                    """
                    insert into document_chunk (title, page_ref, pic_ref, content)
                    values (%s, %s, %s, %s)
                    """,
                    (title, page_ref, pic_ref, content),
                )
        print(f"Inserted chunks from {input_dir} into document_chunk table.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--database_url", type=str, default=None)

    args = parser.parse_args()
    main(
        input_dir=args.input.resolve(),
        artifacts_dir=args.artifacts.resolve(),
        database_url=args.database_url,
    )
