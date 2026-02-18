# doc2md

Standalone document-to-markdown conversion pipeline powered by [Docling](https://github.com/docling-project/docling). Extracts text, tables, and images from PDFs, DOCX, PPTX, and more — with optional AI-powered image descriptions via OpenAI.

## Installation

From GitHub:

```bash
pip install git+https://github.com/Dandiccf/doc2md.git
```

Or with uv (add to an existing project):

```bash
uv add git+https://github.com/Dandiccf/doc2md.git
```

For local development:

```bash
git clone https://github.com/Dandiccf/doc2md.git
cd doc2md
uv sync
```

### OCR extras

The default install uses Docling's auto-detected OCR. Install an extra for a specific engine:

```bash
pip install "doc2md[ocrmac] @ git+https://github.com/Dandiccf/doc2md.git"    # macOS Vision OCR (best on macOS)
pip install "doc2md[easyocr] @ git+https://github.com/Dandiccf/doc2md.git"   # EasyOCR (cross-platform, GPU)
pip install "doc2md[rapidocr] @ git+https://github.com/Dandiccf/doc2md.git"  # RapidOCR (lightweight)
pip install "doc2md[tesseract] @ git+https://github.com/Dandiccf/doc2md.git" # Tesseract
pip install "doc2md[all] @ git+https://github.com/Dandiccf/doc2md.git"       # Everything
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

# Use a specific OCR engine (requires the matching extra installed)
config = PipelineConfig(ocr_engine="ocrmac", ocr_lang=["en-US"])  # macOS only
config = PipelineConfig(ocr_engine="easyocr", ocr_lang=["en"])    # cross-platform, GPU
```

## Configuration

All parameters are set via `PipelineConfig`. Pass it to `convert()` or `DocumentPipeline()`:

```python
from doc2md import convert, PipelineConfig

config = PipelineConfig(
    ocr_engine="ocrmac",
    table_mode="fast",
    do_picture_description=False,
)
result = convert("document.pdf", config=config)
```

### OCR

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `ocr_engine` | `"auto"` | `"auto"`, `"easyocr"`, `"rapidocr"`, `"tesseract"`, `"ocrmac"` | OCR backend. `"auto"` picks the best available. `"ocrmac"` requires macOS and the `[ocrmac]` extra. |
| `ocr_lang` | `["en"]` | list of str | Language codes. Use `["en-US"]` for `ocrmac`, `["en"]` for others. |
| `force_full_page_ocr` | `False` | bool | Run OCR on every page, not just pages with bitmaps. Useful for scanned documents. |
| `bitmap_area_threshold` | `0.05` | float | Minimum bitmap area (fraction of page) to trigger OCR. |

### Tables

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `table_mode` | `"accurate"` | `"accurate"`, `"fast"` | Table structure recognition model. `"accurate"` is slower but better. |
| `do_cell_matching` | `True` | bool | Match detected cells to table grid structure. |

### Images

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `generate_images` | `True` | bool | Extract images from the document. |
| `images_scale` | `2.0` | float | Scale factor for extracted images. Higher = better quality, larger files. |

### Picture descriptions (AI-powered)

Requires `OPENAI_API_KEY` set in environment or `.env` file. Disabled with `do_picture_description=False` (no API key needed).

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `do_picture_description` | `True` | bool | Generate AI descriptions for images. |
| `do_picture_classification` | `True` | bool | Classify images (logo, chart, photo, etc.) to filter descriptions. |
| `picture_description_provider` | `"openai"` | `"openai"` | API provider for descriptions. |
| `picture_description_prompt` | *"Explain what this image conveys..."* | str | Prompt sent to the vision model. |
| `picture_description_timeout` | `60` | int (seconds) | Timeout per description request. |
| `picture_description_concurrency` | `2` | int | Parallel description requests. |
| `picture_description_scale` | `2.0` | float | Image scale sent to the vision model. |
| `picture_area_threshold` | `0.01` | float | Minimum image area (fraction of page) to describe. |
| `classification_deny` | `["logo", "icon", "signature", "stamp", "qr_code", "bar_code"]` | list of str | Skip descriptions for these image types. |
| `classification_min_confidence` | `0.5` | float | Minimum confidence to apply classification filter. |

### Enrichment

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `do_code_enrichment` | `False` | bool | Detect and annotate code blocks. |
| `do_formula_enrichment` | `False` | bool | Detect and convert mathematical formulas. |

### Input formats

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `allowed_formats` | `["pdf", "image", "docx", "pptx", "xlsx", "html", "csv", "md", "asciidoc"]` | list of str | Document formats the pipeline will accept. |

### OpenAI credentials

| Parameter | Default | Source | Description |
|---|---|---|---|
| `openai_api_key` | `""` | `OPENAI_API_KEY` env var / `.env` | Required when `do_picture_description=True`. |
| `openai_model` | `""` | `OPENAI_MODEL` env var / `.env` (fallback: `"gpt-4o"`) | Vision model for descriptions. |

Set via environment:

```bash
cp .env.example .env
# Edit .env with your key
```

Or pass directly:

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

## Testing

Install test dependencies and run the suite:

```bash
uv sync --extra test
uv run pytest tests/ -v
```

Tests automatically download sample documents (a PDF and an image) on first run. The image-description tests require `OPENAI_API_KEY` and are skipped otherwise:

```bash
OPENAI_API_KEY=sk-... uv run pytest tests/ -v
```

## Supported formats

PDF, DOCX, PPTX, XLSX, HTML, CSV, Markdown, AsciiDoc, and images (PNG, JPG, TIFF, BMP).
