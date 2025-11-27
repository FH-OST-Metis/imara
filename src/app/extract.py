from __future__ import annotations

import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def convert_pdf_to_doctags(input_pdf: Path, output_dir: Path) -> Path:
    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    output_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(str(input_pdf))

    doctags = result.document.export_to_doctags()

    out_path = output_dir / f"{input_pdf.stem}.doctags"
    out_path.write_text(doctags, encoding="utf-8")

    print(f"Converted {input_pdf} -> {out_path}")
    return out_path


def main(input_path: Path, output_dir: Path) -> None:
    convert_pdf_to_doctags(input_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main(args.input, args.output)
