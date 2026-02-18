# doc2md

Standalone document-to-markdown conversion pipeline powered by [Docling](https://github.com/docling-project/docling). Extracts text, tables, and images from PDFs, DOCX, PPTX, and more — with optional AI-powered image descriptions via OpenAI or a local vision model (Ollama, LM Studio).

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
| `image_path_prefix` | `""` | str | URL prefix for image paths in the Markdown output. When set, each image reference becomes `prefix/filename.png` (directory paths are stripped, only the filename is kept). E.g. `"/assets"` produces `![img](/assets/image_000.png)`. Empty means no rewriting (Docling's default paths). |

### Picture descriptions (AI-powered)

Disabled with `do_picture_description=False` (no API key or server needed).

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `do_picture_description` | `True` | bool | Generate AI descriptions for images. |
| `do_picture_classification` | `True` | bool | Classify images (logo, chart, photo, etc.) to filter descriptions. |
| `structured_description` | `False` | bool | Request JSON `{summary, detail}` from the vision model. `summary` becomes concise alt text in the image tag, `detail` becomes a blockquote. Falls back gracefully to plain text if the model doesn't return valid JSON. See [Structured descriptions](#structured-descriptions) below. |
| `picture_description_provider` | `"openai"` | `"openai"`, `"local"` | `"openai"` uses the OpenAI API. `"local"` uses any OpenAI-compatible server (Ollama, LM Studio, etc.). |
| `picture_description_prompt` | *"Explain what this image conveys..."* | str | Prompt sent to the vision model. Overridden automatically when `structured_description=True`. |
| `picture_description_timeout` | `60` | int (seconds) | Timeout per description request. |
| `picture_description_concurrency` | `2` | int | Parallel description requests. |
| `picture_description_scale` | `2.0` | float | Image scale sent to the vision model. |
| `picture_area_threshold` | `0.01` | float | Minimum image area (fraction of page) to describe. |
| `classification_deny` | `["logo", "icon", "signature", "stamp", "qr_code", "bar_code"]` | list of str | Skip descriptions for these image types. |
| `classification_min_confidence` | `0.5` | float | Minimum confidence to apply classification filter. |

#### Structured descriptions

By default, image descriptions are placed as a blockquote after a generic `![Image](url)` tag. When `structured_description=True`, the vision model returns a JSON object with two fields, producing richer Markdown:

```markdown
![A bar chart showing Q3 revenue grew 15% YoY across all regions.](http://example.com/image.png)

> The image presents a detailed bar chart comparing quarterly revenue across four
> global regions. North America leads with $4.2M (+18%), followed by Europe...
```

- **`summary`** becomes the image alt text — concise and atomic with the image tag (stays in the same chunk during text splitting).
- **`detail`** becomes the blockquote — thorough description useful for RAG retrieval.

This uses OpenAI's `response_format: {"type": "json_object"}` for guaranteed valid JSON at no extra cost. If the model returns plain text (e.g. a local model without JSON mode), it falls back to the original behavior (generic alt text + full text blockquote).

```python
config = PipelineConfig(
    structured_description=True,  # requires do_picture_description=True (default)
)
result = convert("report.pdf", config=config)
```

### Enrichment

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `do_code_enrichment` | `False` | bool | Detect and annotate code blocks. |
| `do_formula_enrichment` | `False` | bool | Detect and convert mathematical formulas. |

### Input formats

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `allowed_formats` | `["pdf", "image", "docx", "pptx", "xlsx", "html", "csv", "md", "asciidoc"]` | list of str | Document formats the pipeline will accept. |

### OpenAI provider (`"openai"`, default)

| Parameter | Default | Source | Description |
|---|---|---|---|
| `openai_api_key` | `""` | `OPENAI_API_KEY` env var / `.env` | Required when provider is `"openai"`. |
| `openai_model` | `""` | `OPENAI_MODEL` env var / `.env` (fallback: `"gpt-4o"`) | Vision model for descriptions. |

```bash
cp .env.example .env
# Edit .env with your key
```

```python
config = PipelineConfig(openai_api_key="sk-...", openai_model="gpt-4o")
```

### Local provider (`"local"` — Ollama, LM Studio, etc.)

Any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `local_url` | `"http://localhost:11434/v1/chat/completions"` | str | Server URL. Ollama default shown; LM Studio typically uses `http://localhost:1234/v1/chat/completions`. |
| `local_model` | `""` | str | **Required.** Model name, e.g. `"llava"`, `"granite3.2-vision"`, `"gemma3"`. |
| `local_params` | `{}` | dict | Extra request params, e.g. `{"max_completion_tokens": 4096, "seed": 42}`. |

**Ollama example:**

```python
config = PipelineConfig(
    picture_description_provider="local",
    local_url="http://localhost:11434/v1/chat/completions",
    local_model="llava",
    picture_description_timeout=90,
    picture_description_concurrency=1,
)
result = convert("document.pdf", config=config)
```

**LM Studio example:**

```python
config = PipelineConfig(
    picture_description_provider="local",
    local_url="http://localhost:1234/v1/chat/completions",
    local_model="granite3.2-vision",
    local_params={"max_completion_tokens": 4096, "seed": 42},
    picture_description_timeout=90,
    picture_description_concurrency=1,
)
result = convert("document.pdf", config=config)
```

## Output structure

```
output_dir/
├── output.md               # Markdown with images and descriptions
├── output.json             # Full structured JSON export
├── output_artifacts/       # Images referenced in the Markdown (Docling hash names)
├── images/                 # Clean extracted images (picture_000.png, ...)
└── metadata.json           # Conversion metadata (timing, element counts)
```

`result.images_dir` points to `images/` by default. When `image_path_prefix` is set, it points to `output_artifacts/` instead — these are the files whose names match the rewritten Markdown image references, ready for upload to a remote store.

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
