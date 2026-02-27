# doc2md

Standalone document-to-markdown conversion pipeline with a **dual-engine architecture**: simple text-only PDFs are converted in milliseconds using [PyMuPDF4LLM](https://github.com/pymupdf/RAG), while complex documents (images, tables, scans) use [Docling](https://github.com/docling-project/docling)'s deep-learning pipeline. Extracts text, tables, and images from PDFs, DOCX, PPTX, and more — with optional AI-powered image descriptions via OpenAI or a local vision model (Ollama, LM Studio).

## Installation

From GitHub:

```bash
pip install git+https://github.com/Dandiccf/doc2md.git
```

With the fast PyMuPDF4LLM engine (recommended):

```bash
pip install "doc2md[pymupdf] @ git+https://github.com/Dandiccf/doc2md.git"
```

Upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/Dandiccf/doc2md.git
```

Or with uv (add to an existing project):

```bash
uv add git+https://github.com/Dandiccf/doc2md.git
```

Upgrade with uv:

```bash
uv add --upgrade git+https://github.com/Dandiccf/doc2md.git
```

For local development:

```bash
git clone https://github.com/Dandiccf/doc2md.git
cd doc2md
uv sync
```

### Optional extras

```bash
pip install "doc2md[pymupdf] @ git+https://github.com/Dandiccf/doc2md.git"    # Fast PyMuPDF4LLM engine for text-only PDFs
pip install "doc2md[ocrmac] @ git+https://github.com/Dandiccf/doc2md.git"     # macOS Vision OCR (best on macOS)
pip install "doc2md[easyocr] @ git+https://github.com/Dandiccf/doc2md.git"    # EasyOCR (cross-platform, GPU)
pip install "doc2md[rapidocr] @ git+https://github.com/Dandiccf/doc2md.git"   # RapidOCR (lightweight)
pip install "doc2md[tesseract] @ git+https://github.com/Dandiccf/doc2md.git"  # Tesseract
pip install "doc2md[all] @ git+https://github.com/Dandiccf/doc2md.git"        # Everything
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

## Dual-engine architecture

doc2md automatically selects the best conversion engine for each document:

| Signal | PyMuPDF4LLM | Docling |
|---|---|---|
| Text-only PDF (no images, no tables) | ✅ Fast, rule-based | |
| PDF with images/figures | | ✅ DL layout + extraction |
| PDF with tables (ruled grids) | | ✅ TableFormer |
| Scanned PDF (needs OCR) | | ✅ OCR + DL layout |
| Non-PDF formats (DOCX, PPTX, HTML…) | | ✅ |
| URL source | | ✅ |

The `engine` config option controls this:

```python
from doc2md import convert, PipelineConfig

# Auto mode (default) — analyzes the PDF and picks the best engine
config = PipelineConfig(engine="auto", do_picture_description=False)

# Force a specific engine
config = PipelineConfig(engine="pymupdf4llm", do_picture_description=False)
config = PipelineConfig(engine="docling", do_picture_description=False)
```

In auto mode, a lightweight pre-analyzer samples pages using pypdfium2 to detect text density, image coverage, table grid lines, and scanned pages — adding <100ms overhead. If PyMuPDF4LLM is not installed, auto mode falls back to Docling for everything.

The `metadata.json` output includes an `engine_used` field so you always know which engine was used.

## Configuration

All parameters are set via `PipelineConfig`. Pass it to `convert()` or `DocumentPipeline()`:

```python
from doc2md import convert, PipelineConfig

config = PipelineConfig(
    engine="auto",
    ocr_engine="ocrmac",
    table_mode="fast",
    do_picture_description=False,
)
result = convert("document.pdf", config=config)
```

### Engine

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `engine` | `"auto"` | `"auto"`, `"pymupdf4llm"`, `"docling"` | Conversion engine. `"auto"` analyzes the PDF and picks the fastest engine that can handle it. `"pymupdf4llm"` requires the `[pymupdf]` extra. |

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

### Page breaks

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `page_break_placeholder` | `""` | str | String inserted at each page boundary in the Markdown output. E.g. `"<!-- page-break -->"`. Empty means no page break markers. Works with PDF (pages) and PPTX (slides). Flow-based formats like DOCX and HTML have no fixed pages, so the placeholder won't appear. |

### Picture descriptions (AI-powered)

Works for images in PDFs, DOCX, PPTX, and HTML documents, as well as standalone image files. Disabled with `do_picture_description=False` (no API key or server needed).

Each image description is generated with **document context**: the document title and surrounding text are included in the vision prompt, giving the model information for more accurate interpretations of charts, figures, and diagrams. This context also enables reliable language detection when using `picture_description_lang="auto"`.

| Parameter | Default | Choices / Type | Description |
|---|---|---|---|
| `do_picture_description` | `True` | bool | Generate AI descriptions for images. |
| `do_picture_classification` | `True` | bool | Classify images (logo, chart, photo, etc.) to filter descriptions. |
| `structured_description` | `False` | bool | Request JSON `{summary, detail}` from the vision model. `summary` becomes concise alt text in the image tag, `detail` becomes a blockquote. Falls back gracefully to plain text if the model doesn't return valid JSON. See [Structured descriptions](#structured-descriptions) below. |
| `picture_description_provider` | `"openai"` | `"openai"`, `"local"` | `"openai"` uses the OpenAI API. `"local"` uses any OpenAI-compatible server (Ollama, LM Studio, etc.). |
| `picture_description_prompt` | *"Explain what this image conveys..."* | str | Prompt sent to the vision model. Overridden automatically when `structured_description=True`. |
| `picture_description_lang` | `""` | str | Language for image descriptions. ISO&nbsp;639&#8209;1 code (e.g.&nbsp;`"en"`, `"de"`, `"ja"`) to force a specific language, `"auto"` to match the surrounding document text (falls back to English), or empty (default) for no language instruction. |
| `picture_description_timeout` | `60` | int (seconds) | Timeout per description request. |
| `picture_description_concurrency` | `2` | int | Parallel description requests. |
| `picture_description_scale` | `2.0` | float | Image scale sent to the vision model. |
| `picture_area_threshold` | `0.01` | float | Minimum image area (fraction of page) to describe. |
| `classification_deny` | `["logo", "icon", "signature", "stamp", "qr_code", "bar_code"]` | list of str | Skip descriptions for these image types. |
| `classification_min_confidence` | `0.5` | float | Minimum confidence to apply classification filter. |

#### Picture classification

When `do_picture_classification=True` (default), Docling runs a local classification model on each detected image before the description step. The classifier assigns a label with a confidence score, and images whose label appears in `classification_deny` with confidence above `classification_min_confidence` are **skipped** — no vision API call is made for them. This saves API cost and avoids meaningless descriptions for logos, barcodes, and similar decorative elements.

Classification runs entirely locally (no API key needed) and adds negligible overhead to the conversion.

**Available classification labels:**

| Label | Description |
|---|---|
| `bar_chart` | Bar chart |
| `stacked_bar_chart` | Stacked bar chart |
| `line_chart` | Line chart |
| `pie_chart` | Pie chart |
| `scatter_chart` | Scatter plot |
| `flow_chart` | Flow chart / diagram |
| `heatmap` | Heatmap |
| `map` | Geographic map |
| `stratigraphic_chart` | Stratigraphic chart |
| `natural_image` | Photograph / natural image |
| `remote_sensing` | Satellite / remote sensing image |
| `screenshot` | Screenshot |
| `cad_drawing` | CAD drawing |
| `electrical_diagram` | Electrical / circuit diagram |
| `chemistry_molecular_structure` | Molecular structure |
| `chemistry_markush_structure` | Markush structure |
| `icon` | Small icon |
| `logo` | Logo |
| `signature` | Signature |
| `stamp` | Stamp / seal |
| `qr_code` | QR code |
| `bar_code` | Barcode |
| `picture_group` | Group of images |
| `other` | Unclassified |

**Default deny list:** `["logo", "icon", "signature", "stamp", "qr_code", "bar_code"]` — these are typically decorative or non-informational and don't benefit from AI descriptions.

```python
# Describe everything, including logos and barcodes
config = PipelineConfig(classification_deny=[])

# Only skip QR/barcodes, allow logos and icons
config = PipelineConfig(classification_deny=["qr_code", "bar_code"])

# Require higher confidence before skipping (fewer images skipped)
config = PipelineConfig(classification_min_confidence=0.8)

# Disable classification entirely (all images get described)
config = PipelineConfig(do_picture_classification=False)
```

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

#### Description language

By default, the vision model responds in whatever language it chooses (typically English). Use `picture_description_lang` to control this:

```python
# Force German descriptions
config = PipelineConfig(picture_description_lang="de")

# Auto-detect from surrounding document text (falls back to English)
config = PipelineConfig(picture_description_lang="auto")

# Combine with structured descriptions
config = PipelineConfig(structured_description=True, picture_description_lang="ja")
```

The `"auto"` mode works because each vision prompt includes surrounding text from the document, so the model can match that language. Any [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) code is accepted.

### Standalone image handling

When a standalone image file (JPG, PNG, WEBP, TIFF, BMP) is passed to doc2md, it receives special treatment. Docling's layout pipeline runs OCR but rarely detects the entire image as a describable picture, so plain conversion produces only sparse OCR text. doc2md detects standalone images and makes its own vision API call for the whole image — including the filename and any OCR text as context — producing rich markdown:

```markdown
![A bar chart showing quarterly revenue across four regions.](images/chart.png)

> The chart compares Q1-Q4 revenue for North America, Europe, APAC, and LATAM.
> North America leads at $4.2M (+18% YoY)...

---

Revenue ($M): NA 4.2, EU 3.1, APAC 2.8, LATAM 1.5
```

- The image file is copied to `images/` (or served via `image_path_prefix`)
- With `structured_description=True`: summary becomes alt text, detail becomes blockquote
- Any meaningful OCR text (>10 chars) from Docling is preserved after a `---` separator
- When `do_picture_description=False`: no API call, but the image is still referenced in the markdown with any OCR text
- If the vision API fails, falls back gracefully to image reference + OCR text only

This works with both OpenAI and local providers — no extra configuration needed beyond what you already set for picture descriptions.

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

The OpenAI API key is needed for AI-powered picture descriptions (`do_picture_description=True`, the default) and standalone image analysis. All other features (OCR, table extraction, page breaks, etc.) run locally and need no API key.

| Parameter | Default | Description |
|---|---|---|
| `openai_api_key` | `""` | Required when provider is `"openai"`. Env: `OPENAI_API_KEY`. |
| `openai_model` | `""` | Vision model for descriptions (fallback: `"gpt-4o"`). Env: `OPENAI_MODEL`. |
| `openai_base_url` | `"https://api.openai.com/`<br>`v1/chat/completions"` | Full API endpoint URL (path included — Docling posts directly to it). Override for Azure OpenAI or compatible services. Env: `OPENAI_BASE_URL`. |

All three parameters can be set via environment variables or a `.env` file in your project root. doc2md loads `.env` automatically at import time — no extra setup needed. Values passed directly to `PipelineConfig` take precedence over environment variables.

**`.env` file (recommended):**

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
# OPENAI_BASE_URL=https://...  # only needed for Azure or compatible services
```

Then just use the defaults:

```python
from doc2md import convert

result = convert("report.pdf")  # picks up credentials from .env
```

**Pass credentials in code:**

```python
from doc2md import convert, PipelineConfig

config = PipelineConfig(openai_api_key="sk-...", openai_model="gpt-4o")
result = convert("report.pdf", config=config)
```

**Azure OpenAI example:**

```python
config = PipelineConfig(
    openai_api_key="your-azure-api-key",
    openai_model="gpt-4o",
    openai_base_url="https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-10-21",
)
result = convert("report.pdf", config=config)
```

Or via `.env`:

```dotenv
OPENAI_API_KEY=your-azure-api-key
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-10-21
```

**No API key needed?** Disable picture descriptions:

```python
config = PipelineConfig(do_picture_description=False)
result = convert("report.pdf", config=config)
```

### Local provider (`"local"` — Ollama, LM Studio, etc.)

Any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `local_url` | `"http://localhost:11434/`<br>`v1/chat/completions"` | str | Server URL. Ollama default shown; LM Studio typically uses `http://localhost:1234/v1/chat/completions`. |
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

**Docling engine** (complex documents with images/tables):

```
output_dir/
├── output.md               # Markdown with images and descriptions
├── output.json             # Full structured JSON export
├── output_artifacts/       # Images referenced in the Markdown (Docling hash names)
├── images/                 # Clean extracted images (picture_000.png, ...)
└── metadata.json           # Conversion metadata (timing, element counts, engine_used)
```

**PyMuPDF4LLM engine** (text-only PDFs):

```
output_dir/
├── output.md               # Markdown text
└── metadata.json           # Conversion metadata (timing, element counts, engine_used)
```

`result.images_dir` points to `images/` by default. When `image_path_prefix` is set, it points to `output_artifacts/` instead — these are the files whose names match the rewritten Markdown image references, ready for upload to a remote store. For the PyMuPDF4LLM engine, `result.images_dir` and `result.json_path` are `None`.

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
