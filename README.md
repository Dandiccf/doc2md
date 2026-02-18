# doc2md

Standalone document-to-markdown conversion pipeline powered by [Docling](https://github.com/docling-project/docling). Extracts text, tables, and images from PDFs, DOCX, PPTX, and more — with optional AI-powered image descriptions via OpenAI.

## Installation

```bash
pip install .
```

Or with uv:

```bash
uv sync
```

### Platform extras

The default install uses macOS Vision OCR (`ocrmac`). For other platforms:

```bash
pip install ".[easyocr]"    # EasyOCR (cross-platform, GPU-accelerated)
pip install ".[rapidocr]"   # RapidOCR
pip install ".[tesseract]"  # Tesseract
pip install ".[all]"        # Everything
```

## Quick start

### One-liner

```python
from doc2md import convert

result = convert("path/to/document.pdf", output_dir="output/")
print(result.markdown_path)
```

### Reusable pipeline

```python
from doc2md import DocumentPipeline, PipelineConfig

pipeline = DocumentPipeline()

for doc in ["report.pdf", "slides.pptx", "data.xlsx"]:
    result = pipeline.convert(doc)
    if result.success:
        print(f"{doc} -> {result.markdown_path}")
```

### URL support

```python
from doc2md import convert

result = convert("https://arxiv.org/pdf/2408.09869")
```

### Custom configuration

```python
from doc2md import convert, PipelineConfig

# Disable image descriptions (no API key needed)
config = PipelineConfig(do_picture_description=False)
result = convert("document.pdf", config=config)

# Use EasyOCR instead of macOS Vision
config = PipelineConfig(ocr_engine="easyocr", ocr_lang=["en"])
result = convert("document.pdf", config=config)
```

## Configuration

Set your OpenAI API key for image descriptions:

```bash
cp .env.example .env
# Edit .env with your key
```

Or pass it directly:

```python
config = PipelineConfig(openai_api_key="sk-...", openai_model="gpt-4o")
```

## Output structure

```
output_dir/
├── output.md          # Markdown with images and descriptions
├── output.json        # Full structured JSON export
├── images/            # Extracted images (picture_000.png, ...)
└── metadata.json      # Conversion metadata (timing, element counts)
```

## Supported formats

PDF, DOCX, PPTX, XLSX, HTML, CSV, Markdown, AsciiDoc, and images (PNG, JPG, TIFF, BMP).
