# IMARA IEEE Conference Format Conversion

## Overview

This document has been converted from the original German markdown format to IEEE conference format in English. The conversion includes:

- ✅ Translation from German to English
- ✅ IEEE conference two-column format
- ✅ Proper section structure following IEEE guidelines
- ✅ References formatted in IEEE style
- ✅ Abstract and keywords
- ✅ All content preserved (no shortening or extension)
- ✅ Placeholders for images prepared

## Files Generated

- `IMARA_IEEE_Conference.tex` - Main LaTeX document in IEEE conference format

## Compiling the Document

### Prerequisites

You need a LaTeX distribution installed. Options include:
- **TeX Live** (Linux/Unix)
- **MiKTeX** (Windows)
- **MacTeX** (macOS)

### Compilation Steps

1. **Basic compilation:**
   ```bash
   pdflatex IMARA_IEEE_Conference.tex
   bibtex IMARA_IEEE_Conference
   pdflatex IMARA_IEEE_Conference.tex
   pdflatex IMARA_IEEE_Conference.tex
   ```

2. **Using latexmk (recommended):**
   ```bash
   latexmk -pdf IMARA_IEEE_Conference.tex
   ```

## Adding Images

The document references images that need to be added. Follow these steps:

### Image Referenced in the Document

1. **Docling Architecture** (Figure 1)
   - File name expected: `docling_architecture.png`
   - Location: Same directory as the .tex file
   - Source: From the original document or Docling project documentation

### How to Add Images

1. Place your image files in the same directory as `IMARA_IEEE_Conference.tex`
2. Ensure the filenames match those referenced in the document:
   - `docling_architecture.png`

3. If your images have different names, update the references in the .tex file:
   ```latex
   \includegraphics[width=\columnwidth]{your_image_name.png}
   ```

### Image Format Recommendations

- **Preferred formats:** PNG (for diagrams), PDF (for vector graphics), JPG (for photos)
- **Resolution:** At least 300 DPI for print quality
- **Width:** Images will be scaled to column width automatically
- **File size:** Keep reasonable for document portability

### Image Placeholders

If images are not available, the document will compile with placeholder boxes. The LaTeX compiler will show warnings but will still generate a PDF.

## Document Structure

The IEEE conference format includes:

1. **Title and Authors** - Update with actual institution information
2. **Abstract** - Concise summary of the work
3. **Keywords** - Key terms for indexing
4. **Main Sections:**
   - Introduction
   - State of the Art
   - Methodology
   - Results
   - Discussion
   - Conclusion
5. **References** - IEEE citation style
6. **Appendix** - Glossary of terms

## Customization

### Author Information

Update the author block in the .tex file with actual information:

```latex
\IEEEauthorblockA{\textit{[Institution]} \\
\textit{[Department]}\\
[City], [Country] \\
[email]}
```

### Adding More Figures

To add additional figures, use this template:

```latex
\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{image_name.png}}
\caption{Your caption here}
\label{fig:your_label}
\end{figure}
```

Reference in text: `Fig.~\ref{fig:your_label}`

### Adding More Tables

To add additional tables, use this template:

```latex
\begin{table}[htbp]
\caption{Table Caption}
\begin{center}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Header 1} & \textbf{Header 2} & \textbf{Header 3} \\
\hline
Data 1 & Data 2 & Data 3 \\
\hline
\end{tabular}
\label{tab:your_label}
\end{center}
\end{table}
```

## Translation Notes

The document has been translated from German to English while preserving:
- All technical terminology
- Original structure and organization
- Complete content (no summarization or expansion)
- Scientific accuracy
- References and citations

## IEEE Format Compliance

This document follows IEEE conference paper guidelines:
- Two-column format
- 10pt font size
- Standard IEEE margins
- IEEE citation style
- Proper section numbering
- Figure and table formatting

## Troubleshooting

### Common Issues

1. **Missing IEEEtran.cls:**
   - Download from: http://www.ieee.org/conferences_events/conferences/publishing/templates.html
   - Or install via your LaTeX distribution package manager

2. **Image not found:**
   - Check that image files are in the same directory
   - Verify filenames match exactly (case-sensitive on Linux/Mac)
   - Check file extensions

3. **Bibliography not appearing:**
   - Run bibtex after first pdflatex compilation
   - Then run pdflatex twice more

4. **Overfull hbox warnings:**
   - These are common and usually acceptable
   - Review the PDF to ensure text isn't extending into margins

## Next Steps

1. ✅ Add actual image files to the directory
2. ✅ Update author institution information
3. ✅ Compile the document
4. ✅ Review the PDF output
5. ✅ Make any final adjustments

## Support

For IEEE LaTeX template issues:
- IEEE Author Center: https://ieeeauthorcenter.ieee.org/
- Template downloads: http://www.ieee.org/conferences_events/conferences/publishing/templates.html

For LaTeX help:
- TeX Stack Exchange: https://tex.stackexchange.com/
- Overleaf Documentation: https://www.overleaf.com/learn

---

**Date of Conversion:** January 19, 2026  
**Original Document:** Projekt IMARA Schlussbericht.md (German)  
**Target Format:** IEEE Conference Paper (English)
