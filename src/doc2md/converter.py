"""Core conversion pipeline: builds a Docling converter and runs conversions."""

from __future__ import annotations

import re
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import PictureItem, TableItem
from docling_core.types.doc.labels import PictureClassificationLabel
from PIL import Image as PILImage
from pydantic import AnyUrl

from doc2md.config import PipelineConfig
from doc2md.serializers import (
    DescriptionEnrichedImageDocSerializer,
    _parse_description,
    _sanitize_alt_text,
    strip_think_tags,
)
from doc2md.utils import (
    ConversionMetadata,
    ElementCounts,
    TimingInfo,
    logger,
    timed,
)

# Mapping from string names to PictureClassificationLabel enums
_CLASSIFICATION_LABEL_MAP: dict[str, PictureClassificationLabel] = {
    "logo": PictureClassificationLabel.LOGO,
    "icon": PictureClassificationLabel.ICON,
    "signature": PictureClassificationLabel.SIGNATURE,
    "stamp": PictureClassificationLabel.STAMP,
    "qr_code": PictureClassificationLabel.QR_CODE,
    "bar_code": PictureClassificationLabel.BAR_CODE,
}

# Mapping from string names to InputFormat enums
_INPUT_FORMAT_MAP: dict[str, InputFormat] = {
    "pdf": InputFormat.PDF,
    "image": InputFormat.IMAGE,
    "docx": InputFormat.DOCX,
    "pptx": InputFormat.PPTX,
    "xlsx": InputFormat.XLSX,
    "html": InputFormat.HTML,
    "csv": InputFormat.CSV,
    "md": InputFormat.MD,
    "asciidoc": InputFormat.ASCIIDOC,
}

_IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"})


@dataclass
class ConversionResult:
    """Result of a single document conversion."""

    success: bool = False
    markdown_path: Path | None = None
    json_path: Path | None = None
    images_dir: Path | None = None
    metadata_path: Path | None = None
    metadata: ConversionMetadata = field(default_factory=ConversionMetadata)
    error: str | None = None


class DocumentPipeline:
    """Reusable document conversion pipeline configured via PipelineConfig."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._converter: DocumentConverter | None = None

    def _build_converter(self) -> DocumentConverter:
        """Build a DocumentConverter from the current config."""
        cfg = self.config
        opts = PdfPipelineOptions()

        # OCR configuration
        opts.do_ocr = True
        if cfg.ocr_engine == "easyocr":
            opts.ocr_options = EasyOcrOptions(
                lang=cfg.ocr_lang,
                force_full_page_ocr=cfg.force_full_page_ocr,
                bitmap_area_threshold=cfg.bitmap_area_threshold,
            )
        elif cfg.ocr_engine == "ocrmac":
            opts.ocr_options = OcrMacOptions(
                lang=cfg.ocr_lang,
                force_full_page_ocr=cfg.force_full_page_ocr,
                bitmap_area_threshold=cfg.bitmap_area_threshold,
            )
        else:
            from docling.datamodel.pipeline_options import OcrAutoOptions
            opts.ocr_options = OcrAutoOptions(
                force_full_page_ocr=cfg.force_full_page_ocr,
                bitmap_area_threshold=cfg.bitmap_area_threshold,
            )

        # Table structure
        opts.do_table_structure = True
        opts.table_structure_options = TableStructureOptions(
            do_cell_matching=cfg.do_cell_matching,
            mode=(
                TableFormerMode.ACCURATE
                if cfg.table_mode == "accurate"
                else TableFormerMode.FAST
            ),
        )

        # Image generation
        if cfg.generate_images:
            opts.generate_picture_images = True
            opts.images_scale = cfg.images_scale

        # Picture description via API
        if cfg.do_picture_description:
            opts.enable_remote_services = True
            opts.do_picture_description = True
            opts.generate_picture_images = True
            opts.images_scale = max(opts.images_scale, cfg.picture_description_scale)

            deny_labels = [
                _CLASSIFICATION_LABEL_MAP[label]
                for label in cfg.classification_deny
                if label in _CLASSIFICATION_LABEL_MAP
            ]

            if cfg.picture_description_provider == "local":
                api_url = AnyUrl(cfg.local_url)
                api_headers: dict = {}
                api_params: dict = {"model": cfg.local_model, **cfg.local_params}
            else:
                api_url = AnyUrl("https://api.openai.com/v1/chat/completions")
                api_headers = {"Authorization": f"Bearer {cfg.openai_api_key}"}
                api_params = {"model": cfg.openai_model}

            prompt = cfg.picture_description_prompt
            if cfg.structured_description:
                prompt = (
                    "Analyze this image and respond with a JSON object containing "
                    "exactly two fields:\n"
                    '- "summary": A concise 1-2 sentence description of what the '
                    "image shows and its key message. Use only plain text — letters, "
                    "numbers, periods, commas, hyphens, and spaces. No brackets, "
                    "backslashes, or special markdown characters.\n"
                    '- "detail": A thorough explanation of the image content, key '
                    "findings, data, and takeaways. Focus on meaning rather than "
                    "visual styling."
                )
                api_params["response_format"] = {"type": "json_object"}

            opts.picture_description_options = PictureDescriptionApiOptions(
                url=api_url,
                headers=api_headers,
                params=api_params,
                prompt=prompt,
                timeout=cfg.picture_description_timeout,
                concurrency=cfg.picture_description_concurrency,
                scale=cfg.picture_description_scale,
                picture_area_threshold=cfg.picture_area_threshold,
                classification_deny=deny_labels,
                classification_min_confidence=cfg.classification_min_confidence,
            )

        # Picture classification
        if cfg.do_picture_classification:
            opts.do_picture_classification = True

        # Code and formula enrichment
        if cfg.do_code_enrichment:
            opts.do_code_enrichment = True
        if cfg.do_formula_enrichment:
            opts.do_formula_enrichment = True

        # Map allowed format strings to InputFormat enums
        allowed = [
            _INPUT_FORMAT_MAP[fmt]
            for fmt in cfg.allowed_formats
            if fmt in _INPUT_FORMAT_MAP
        ]

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
        }

        return DocumentConverter(
            allowed_formats=allowed,
            format_options=format_options,
        )

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            self._converter = self._build_converter()
        return self._converter

    @staticmethod
    def _rewrite_image_paths(md_text: str, prefix: str) -> str:
        """Replace Markdown image paths with ``prefix/<filename>``.

        When a prefix is configured (e.g. a URL path for serving images),
        only the filename portion of each image reference is kept — the
        original directory path (which may be an absolute temp-dir path
        from Docling) is stripped.
        """
        prefix = prefix.rstrip("/")
        return re.sub(
            r"(!\[[^\]]*\]\()([^)]+)(\))",
            lambda m: f"{m.group(1)}{prefix}/{Path(m.group(2)).name}{m.group(3)}",
            md_text,
        )

    @staticmethod
    def _is_standalone_image(source: str) -> bool:
        """Return True if the source path has an image file extension."""
        return Path(source).suffix.lstrip(".").lower() in _IMAGE_EXTENSIONS

    def _get_vision_api_config(self) -> tuple[AnyUrl, dict, dict, str]:
        """Extract (api_url, headers, params, prompt) from config.

        Mirrors the logic in ``_build_converter()`` for the picture description
        API but returns the values for direct use with ``api_image_request``.
        """
        cfg = self.config

        if cfg.picture_description_provider == "local":
            api_url = AnyUrl(cfg.local_url)
            headers: dict = {}
            params: dict = {"model": cfg.local_model, **cfg.local_params}
        else:
            api_url = AnyUrl("https://api.openai.com/v1/chat/completions")
            headers = {"Authorization": f"Bearer {cfg.openai_api_key}"}
            params = {"model": cfg.openai_model}

        prompt = cfg.picture_description_prompt
        if cfg.structured_description:
            prompt = (
                "Analyze this image and respond with a JSON object containing "
                "exactly two fields:\n"
                '- "summary": A concise 1-2 sentence description of what the '
                "image shows and its key message. Use only plain text — letters, "
                "numbers, periods, commas, hyphens, and spaces. No brackets, "
                "backslashes, or special markdown characters.\n"
                '- "detail": A thorough explanation of the image content, key '
                "findings, data, and takeaways. Focus on meaning rather than "
                "visual styling."
            )
            params["response_format"] = {"type": "json_object"}

        return api_url, headers, params, prompt

    def _describe_standalone_image(self, image: PILImage.Image) -> str | None:
        """Send a standalone image to the vision API for description.

        Returns the description text, or None on failure.
        """
        from docling.utils.api_image_request import api_image_request

        api_url, headers, params, prompt = self._get_vision_api_config()
        try:
            text, _tokens, _stop = api_image_request(
                image=image,
                prompt=prompt,
                url=api_url,
                timeout=self.config.picture_description_timeout,
                headers=headers,
                **params,
            )
            if text:
                return strip_think_tags(text).strip()
            logger.warning("Vision API returned empty response for standalone image")
            return None
        except Exception as exc:
            logger.warning("Vision API call failed for standalone image: %s", exc)
            return None

    def _build_standalone_image_markdown(
        self,
        source_path: Path,
        docling_md: str,
        description: str | None,
        output_dir: Path,
    ) -> tuple[str, Path]:
        """Build enhanced markdown for a standalone image.

        Returns (markdown_text, images_dir).
        """
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy source image to images/
        dest = images_dir / source_path.name
        shutil.copy2(source_path, dest)

        # Build image reference path
        if self.config.image_path_prefix:
            prefix = self.config.image_path_prefix.rstrip("/")
            img_ref = f"{prefix}/{source_path.name}"
        else:
            img_ref = f"images/{source_path.name}"

        parts: list[str] = []

        if description:
            summary, detail = _parse_description(description)
            if summary:
                alt = _sanitize_alt_text(summary)
            else:
                alt = "Image"
            parts.append(f"![{alt}]({img_ref})")
            if detail:
                blockquote = "\n".join(f"> {line}" for line in detail.splitlines())
                parts.append(blockquote)
        else:
            parts.append(f"![Image]({img_ref})")

        # Append OCR text if meaningful
        ocr_text = docling_md.strip()
        if len(ocr_text) > 10:
            parts.append("---")
            parts.append(ocr_text)

        return "\n\n".join(parts) + "\n", images_dir

    def _export_markdown_with_descriptions(self, doc, md_path: Path) -> Path:
        """Export markdown using a custom serializer that places descriptions after images.

        Returns the artifacts directory containing the referenced image files.
        """
        artifacts_dir, ref_path = doc._get_output_paths(md_path)  # pylint: disable=protected-access
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ref_doc = doc._make_copy_with_refmode(  # pylint: disable=protected-access
            artifacts_dir, ImageRefMode.REFERENCED, None,
            reference_path=ref_path,
        )

        serializer = DescriptionEnrichedImageDocSerializer(
            doc=ref_doc,
            params=MarkdownParams(
                image_mode=ImageRefMode.REFERENCED,
            ),
        )
        md_text = serializer.serialize().text
        if self.config.image_path_prefix:
            md_text = self._rewrite_image_paths(md_text, self.config.image_path_prefix)
        md_path.write_text(md_text, encoding="utf-8")
        return artifacts_dir

    def convert(self, source: str | Path, output_dir: str | Path | None = None) -> ConversionResult:
        """Convert a single document and save all outputs.

        Args:
            source: File path or URL to the document.
            output_dir: Directory for outputs. Auto-generated from source name if None.
        """
        source_str = str(source)

        # Determine if source is a URL
        parsed = urlparse(source_str)
        is_url = parsed.scheme in ("http", "https")

        # Determine source name for metadata and default output dir
        if is_url:
            source_name = Path(parsed.path).name or "document"
        else:
            source_name = Path(source_str).name

        # Auto-generate output_dir if not provided
        if output_dir is None:
            stem = Path(source_name).stem
            output_dir = Path.cwd() / f"{stem}_output"
        output_dir = Path(output_dir)

        result = ConversionResult()
        result.metadata.document = source_name
        result.metadata.config = self.config.to_dict()

        try:
            result.metadata.docling_version = getattr(docling, "__version__", "unknown")
        except Exception:
            result.metadata.docling_version = "unknown"

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            converter = self._get_converter()

            with timed() as timing:
                conv_res = converter.convert(
                    source_str,
                    raises_on_error=True,
                )

            result.metadata.timing = TimingInfo(
                start=timing.start,
                end=timing.end,
                elapsed_seconds=timing.elapsed_seconds,
            )

            # Count elements
            counts = ElementCounts()
            counts.pages = len(conv_res.document.pages)
            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, PictureItem):
                    counts.pictures += 1
                elif isinstance(element, TableItem):
                    counts.tables += 1
                else:
                    counts.text_items += 1
            result.metadata.elements = counts

            # Save markdown
            md_path = output_dir / "output.md"
            cfg = self.config
            if cfg.generate_images or cfg.do_picture_description:
                images_dir = output_dir / "images"
                images_dir.mkdir(exist_ok=True)
                result.images_dir = images_dir

                # Save individual picture images
                pic_idx = 0
                for element, _level in conv_res.document.iterate_items():
                    if isinstance(element, PictureItem):
                        try:
                            img = element.get_image(conv_res.document)
                            if img is not None:
                                img.save(
                                    images_dir / f"picture_{pic_idx:03d}.png",
                                    format="PNG",
                                )
                                pic_idx += 1
                        except Exception as img_err:
                            logger.warning(
                                "Failed to save image %d: %s", pic_idx, img_err,
                            )

                if cfg.do_picture_description:
                    artifacts_dir = self._export_markdown_with_descriptions(
                        conv_res.document, md_path,
                    )
                    # When image_path_prefix is set, callers will serve images
                    # by filename from a remote store — point images_dir to the
                    # artifacts directory whose filenames match the markdown refs.
                    if cfg.image_path_prefix:
                        result.images_dir = artifacts_dir
                else:
                    conv_res.document.save_as_markdown(
                        md_path,
                        image_mode=ImageRefMode.REFERENCED,
                    )
                    if cfg.image_path_prefix:
                        md_text = md_path.read_text(encoding="utf-8")
                        md_text = self._rewrite_image_paths(
                            md_text, cfg.image_path_prefix,
                        )
                        md_path.write_text(md_text, encoding="utf-8")
                        # Point images_dir to the artifacts directory whose
                        # filenames match the rewritten markdown references.
                        artifacts_dir = output_dir / f"{md_path.stem}_artifacts"
                        if artifacts_dir.exists():
                            result.images_dir = artifacts_dir
            else:
                md_text = conv_res.document.export_to_markdown()
                md_path.write_text(md_text, encoding="utf-8")
            result.markdown_path = md_path

            # Standalone image enhancement
            if (
                not is_url
                and self._is_standalone_image(source_str)
                and cfg.do_picture_description
            ):
                source_path = Path(source_str)
                docling_md = md_path.read_text(encoding="utf-8")
                try:
                    image = PILImage.open(source_path).convert("RGB")
                    description = self._describe_standalone_image(image)
                except Exception as img_exc:
                    logger.warning(
                        "Failed to open image for description: %s", img_exc,
                    )
                    description = None
                md_text, images_dir = self._build_standalone_image_markdown(
                    source_path, docling_md, description, output_dir,
                )
                md_path.write_text(md_text, encoding="utf-8")
                result.images_dir = images_dir
            elif (
                not is_url
                and self._is_standalone_image(source_str)
                and not cfg.do_picture_description
            ):
                # No LLM call, but still copy image and add a reference
                source_path = Path(source_str)
                docling_md = md_path.read_text(encoding="utf-8")
                md_text, images_dir = self._build_standalone_image_markdown(
                    source_path, docling_md, None, output_dir,
                )
                md_path.write_text(md_text, encoding="utf-8")
                result.images_dir = images_dir

            # Save JSON export
            json_path = output_dir / "output.json"
            conv_res.document.save_as_json(json_path)
            result.json_path = json_path

            result.success = True
            logger.info(
                "Converted %s in %.1fs (%d pages, %d tables, %d pictures)",
                source_name,
                timing.elapsed_seconds,
                counts.pages,
                counts.tables,
                counts.pictures,
            )

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.metadata.error = result.error
            logger.error("Failed to convert %s: %s", source_name, exc)
            logger.debug(traceback.format_exc())

        # Always save metadata
        metadata_path = output_dir / "metadata.json"
        result.metadata.save(metadata_path)
        result.metadata_path = metadata_path
        return result
