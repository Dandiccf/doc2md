"""Complete configuration with all Docling parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    """Configuration for the document conversion pipeline.

    Active parameters are set to the optimal defaults (matching the
    ``with_image_descriptions_openai`` profile from the testbench).
    Commented-out parameters are documented for reference and can be
    uncommented and used as needed.
    """

    # -- OCR ----------------------------------------------------------------
    ocr_engine: str = "auto"  # "auto", "easyocr", "rapidocr", "tesseract", "ocrmac" (macOS only)
    ocr_lang: list[str] = field(default_factory=lambda: ["en"])
    force_full_page_ocr: bool = False
    bitmap_area_threshold: float = 0.05

    # EasyOCR-specific:
    # easyocr_use_gpu: bool = True
    # easyocr_confidence_threshold: float = 0.5
    # easyocr_model_storage_directory: str | None = None

    # RapidOCR-specific:
    # rapidocr_use_gpu: bool = False

    # Tesseract-specific:
    # tesseract_cmd: str | None = None
    # tesseract_psm: int = 3

    # OcrMac-specific:
    # ocrmac_recognition: str = "accurate"
    # ocrmac_framework: str = "vision"

    # -- Tables -------------------------------------------------------------
    table_mode: str = "accurate"  # "accurate" or "fast"
    do_cell_matching: bool = True

    # -- Images -------------------------------------------------------------
    generate_images: bool = True
    images_scale: float = 2.0
    # generate_page_images: bool = False

    # -- Picture description ------------------------------------------------
    do_picture_description: bool = True
    do_picture_classification: bool = True
    picture_description_provider: str = "openai"  # "openai" or "local" (Ollama, LM Studio, etc.)
    picture_description_prompt: str = (
        "Explain what this image conveys. Focus on the meaning, "
        "key findings, and takeaways rather than describing visual "
        "elements like colors or layout."
    )
    picture_description_timeout: int = 60
    picture_description_concurrency: int = 2
    picture_description_scale: float = 2.0
    picture_area_threshold: float = 0.01
    classification_deny: list[str] = field(
        default_factory=lambda: [
            "logo", "icon", "signature", "stamp", "qr_code", "bar_code",
        ]
    )
    classification_min_confidence: float = 0.5
    # classification_allow: list[str] | None = None

    # -- Local provider settings (Ollama, LM Studio, any OpenAI-compatible) -
    local_url: str = "http://localhost:11434/v1/chat/completions"  # Ollama default
    local_model: str = ""  # e.g. "llava", "granite3.2-vision", "gemma3"
    local_params: dict = field(default_factory=dict)  # extra params, e.g. {"max_completion_tokens": 4096, "seed": 42}

    # -- Enrichment ---------------------------------------------------------
    do_code_enrichment: bool = False
    do_formula_enrichment: bool = False

    # -- Layout -------------------------------------------------------------
    # layout_model: str | None = None
    # create_orphan_clusters: bool = False

    # -- Pipeline -----------------------------------------------------------
    # document_timeout: int | None = None
    # accelerator_device: str = "auto"
    # accelerator_num_threads: int | None = None

    # -- PDF backend --------------------------------------------------------
    # pdf_backend: str = "dlparse_v4"

    # -- Batch sizes --------------------------------------------------------
    # ocr_batch_size: int | None = None
    # layout_batch_size: int | None = None
    # table_batch_size: int | None = None

    # -- Allowed input formats ----------------------------------------------
    allowed_formats: list[str] = field(
        default_factory=lambda: [
            "pdf", "image", "docx", "pptx", "xlsx",
            "html", "csv", "md", "asciidoc",
        ]
    )

    # -- OpenAI credentials (populated from env in __post_init__) -----------
    openai_api_key: str = ""
    openai_model: str = ""

    def __post_init__(self) -> None:
        if not self.do_picture_description:
            return
        if self.picture_description_provider == "openai":
            if not self.openai_api_key:
                self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
                if not self.openai_api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY not set in environment or .env file"
                    )
            if not self.openai_model:
                self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif self.picture_description_provider == "local":
            if not self.local_model:
                raise RuntimeError(
                    "local_model must be set when using the 'local' provider "
                    "(e.g. 'llava', 'granite3.2-vision')"
                )

    def to_dict(self) -> dict:
        """Return a serialisable dict of the active configuration."""
        return {
            "ocr_engine": self.ocr_engine,
            "ocr_lang": self.ocr_lang,
            "force_full_page_ocr": self.force_full_page_ocr,
            "bitmap_area_threshold": self.bitmap_area_threshold,
            "table_mode": self.table_mode,
            "do_cell_matching": self.do_cell_matching,
            "generate_images": self.generate_images,
            "images_scale": self.images_scale,
            "do_picture_description": self.do_picture_description,
            "do_picture_classification": self.do_picture_classification,
            "picture_description_provider": self.picture_description_provider,
            "local_url": self.local_url,
            "local_model": self.local_model,
            "do_code_enrichment": self.do_code_enrichment,
            "do_formula_enrichment": self.do_formula_enrichment,
            "allowed_formats": self.allowed_formats,
        }
