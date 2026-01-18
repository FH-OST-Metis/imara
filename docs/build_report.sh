#!/usr/bin/env bash
set -euo pipefail

# Build the IMARA project report PDF via Pandoc
# Usage: ./build_report.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_DIR="$ROOT_DIR/docs/9. Projektabschluss/Dokumentation"
INPUT_MD="$DOC_DIR/Projekt IMARA Schlussbericht.md"
OUTPUT_PDF="$DOC_DIR/Projekt IMARA Schlussbericht-pandoc.pdf"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc is not installed or not in PATH." >&2
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "Warning: xelatex not found. Pandoc may fail to build the PDF." >&2
fi

cd "$ROOT_DIR"

pandoc "$INPUT_MD" \
  -o "$OUTPUT_PDF" \
  --pdf-engine=xelatex \
  --resource-path="$DOC_DIR"

echo "Report built: $OUTPUT_PDF"