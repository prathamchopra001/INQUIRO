# INQUIRO LaTeX Templates

This directory contains LaTeX templates for INQUIRO report generation.

## Available Templates

| Template | Description | Requires External Files |
|----------|-------------|------------------------|
| `plain.tex` | Simple article format | No |
| `arxiv.tex` | arXiv preprint style | No |
| `neurips.tex` | NeurIPS/ML conference format | No (uses approximation) |
| `ieee.tex` | IEEE journal format | No (uses IEEEtran.cls if available) |

All templates work out-of-the-box with INQUIRO. For official conference submissions,
you may want to download the actual style files for pixel-perfect formatting.

## Usage

### From the CLI Wizard

When running `python run_inquiro.py`, select "Markdown + LaTeX/PDF" in the output
format step, then choose your preferred template.

### Programmatically

```python
from src.reports.latex_compiler import compile_report_to_latex

pdf_path = compile_report_to_latex(
    "outputs/run_xxx/report.md",
    "outputs/run_xxx/latex",
    template="arxiv"  # or "plain", "neurips", "ieee"
)
```

## Using Official Conference Templates

### NeurIPS Template

For camera-ready NeurIPS submissions:

1. Download the official NeurIPS style file from:
   https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles

2. Place `neurips_2024.sty` in this directory

3. The compiler will automatically use the official style file

### IEEE Template

For IEEE journal submissions:

1. Download IEEEtran from CTAN:
   https://ctan.org/pkg/ieeetran

2. Place `IEEEtran.cls` in this directory (or install system-wide)

3. The compiler will automatically use the official class file

## Template Placeholders

All templates support these placeholders:
- `{title}` - Paper title
- `{author}` - Author name
- `{institution}` - Institution (if applicable)
- `{date}` - Generation date
- `{body}` - Main content (converted from markdown)
- `{bibliography}` - References section

## Requirements

To compile LaTeX to PDF, you need one of:
- **TeX Live** (Linux/macOS/Windows): https://tug.org/texlive/
- **MiKTeX** (Windows): https://miktex.org/
- **MacTeX** (macOS): https://tug.org/mactex/

If no LaTeX compiler is installed, INQUIRO will generate `.tex` source files
that you can compile manually or in Overleaf.

## Checking LaTeX Availability

```python
from src.reports.latex_compiler import check_latex_installation

status = check_latex_installation()
print(status)
# {'pdflatex': True, 'xelatex': True, 'lualatex': False}
```
