"""Core conversion pipeline: builds a Docling converter and runs conversions."""

from __future__ import annotations

import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    ImageFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    DescriptionMetaField,
    PictureMeta,
    PictureItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel, PictureClassificationLabel
from PIL import Image as PILImage
from pydantic import AnyUrl

from doc2md.config import (
    DEFAULT_PICTURE_DESCRIPTION_PROMPT,
    DEFAULT_STRUCTURED_DETAIL_PROMPT,
    PipelineConfig,
)
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
    VisionUsage,
    logger,
    timed,
)

_PDF_EXTENSIONS = frozenset({"pdf"})

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
        self._converters: dict[bool, DocumentConverter] = {}
        self._last_analysis = None  # cached PdfAnalysis from _resolve_engine

    def _build_converter(self, do_ocr: bool = False) -> DocumentConverter:
        """Build a DocumentConverter from the current config."""
        cfg = self.config
        opts = PdfPipelineOptions()

        # OCR configuration
        opts.do_ocr = do_ocr
        if do_ocr:
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

        # Picture description: images are generated here, but API calls are
        # handled in _describe_document_images() after conversion so that
        # surrounding text and document title can be included as context.
        if cfg.do_picture_description:
            opts.generate_picture_images = True
            opts.images_scale = max(opts.images_scale, cfg.picture_description_scale)

        # Picture classification
        if cfg.do_picture_classification:
            opts.do_picture_classification = True

        # Code and formula enrichment
        if cfg.do_code_enrichment:
            opts.do_code_enrichment = True
        if cfg.do_formula_enrichment:
            opts.do_formula_enrichment = True

        # Pipeline options for non-PDF formats (DOCX, PPTX, HTML) that use SimplePipeline
        simple_opts = ConvertPipelineOptions()
        if cfg.do_picture_classification:
            simple_opts.do_picture_classification = True

        # Map allowed format strings to InputFormat enums
        allowed = [
            _INPUT_FORMAT_MAP[fmt]
            for fmt in cfg.allowed_formats
            if fmt in _INPUT_FORMAT_MAP
        ]

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
            InputFormat.DOCX: WordFormatOption(pipeline_options=simple_opts),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=simple_opts),
            InputFormat.HTML: HTMLFormatOption(pipeline_options=simple_opts),
        }

        return DocumentConverter(
            allowed_formats=allowed,
            format_options=format_options,
        )

    def _get_converter(self, do_ocr: bool = False) -> DocumentConverter:
        if do_ocr not in self._converters:
            self._converters[do_ocr] = self._build_converter(do_ocr)
        return self._converters[do_ocr]

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

    def _get_vision_api_config(self) -> tuple[AnyUrl, dict, dict]:
        """Return ``(api_url, headers, params)`` for direct ``api_image_request`` calls."""
        cfg = self.config

        if cfg.picture_description_provider == "local":
            api_url = AnyUrl(cfg.local_url)
            headers: dict = {}
            params: dict = {"model": cfg.local_model, **cfg.local_params}
        else:
            api_url = AnyUrl(cfg.openai_base_url)
            headers = {"Authorization": f"Bearer {cfg.openai_api_key}"}
            params = {"model": cfg.openai_model}

        if cfg.structured_description:
            params["response_format"] = {"type": "json_object"}

        return api_url, headers, params

    # -- Contextual prompt construction ------------------------------------

    def _build_base_prompt(self) -> str:
        """Return the base vision prompt (plain or structured).

        ``structured_description`` fixes the *shape* of the answer — a JSON
        object with ``summary`` and ``detail`` — while
        ``picture_description_prompt`` says what the model should look for. The
        two are independent, so the custom instruction drives the ``detail``
        field instead of being discarded.

        It used to be discarded: turning on structured descriptions silently
        replaced the caller's prompt with a hardcoded one, which made the two
        options mutually exclusive with nothing to indicate it. A caller could
        set a prompt, see it take effect in plain mode, enable structured mode
        and watch it stop mattering.
        """
        cfg = self.config
        if not cfg.structured_description:
            return cfg.picture_description_prompt

        detail = cfg.picture_description_prompt
        if detail == DEFAULT_PICTURE_DESCRIPTION_PROMPT:
            # Nothing was asked for, so keep the wording this mode has always
            # used rather than swapping in the plain-mode default.
            detail = DEFAULT_STRUCTURED_DETAIL_PROMPT
        return (
            "Analyze this image and respond with a JSON object containing "
            "exactly two fields:\n"
            '- "summary": A concise 1-2 sentence description of what the '
            "image shows and its key message. Use only plain text — letters, "
            "numbers, periods, commas, hyphens, and spaces. No brackets, "
            "backslashes, or special markdown characters.\n"
            f'- "detail": {detail}'
        )

    def _build_contextual_prompt(
        self,
        doc_title: str = "",
        surrounding_text: str = "",
    ) -> str:
        """Build a vision prompt enriched with document context and language."""
        prompt = self._build_base_prompt()

        context_parts: list[str] = []
        if doc_title:
            context_parts.append(f"Document title: {doc_title}")
        if surrounding_text:
            context_parts.append(f"Surrounding text:\n{surrounding_text}")
        if context_parts:
            prompt += "\n\nContext:\n" + "\n".join(context_parts)

        # Language instruction
        lang = self.config.picture_description_lang.strip().lower()
        if lang == "auto":
            prompt += (
                "\n\nIMPORTANT: Respond in the same language as the "
                "surrounding text. If no surrounding text is provided or "
                "the language cannot be determined, use English."
            )
        elif lang:
            prompt += (
                f"\n\nIMPORTANT: Respond in the language specified by "
                f"ISO 639-1 code '{lang}'."
            )

        return prompt

    # -- Classification filter ----------------------------------------------

    def _should_describe(self, item: PictureItem) -> bool:
        """Return False if the item's classification is in the deny list."""
        cfg = self.config
        if not item.meta or not item.meta.classification:
            return True
        deny = set(cfg.classification_deny)
        for pred in item.meta.classification.predictions:
            if (
                pred.class_name in deny
                and pred.confidence is not None
                and pred.confidence >= cfg.classification_min_confidence
            ):
                return False
        return True

    # -- Post-processing image descriptions ---------------------------------

    @staticmethod
    def _get_document_title(doc) -> str:
        """Extract the document title from the first meaningful text element.

        Priority: ``title`` label > first text or heading element (whichever
        comes first in document order) > ``doc.name`` (filename).
        """
        for item, _ in doc.iterate_items():
            if not isinstance(item, TextItem) or not item.text.strip():
                continue
            label = getattr(item, "label", None)
            if label == DocItemLabel.TITLE:
                return item.text.strip()
            if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TEXT):
                return item.text.strip()
        return getattr(doc, "name", "") or ""

    @staticmethod
    def _get_surrounding_text(items: list, idx: int, window: int = 3) -> str:
        """Collect text from elements adjacent to *idx*."""
        parts: list[str] = []
        for j in range(max(0, idx - window), idx):
            el, _ = items[j]
            if isinstance(el, TextItem) and el.text.strip():
                parts.append(el.text.strip())
        for j in range(idx + 1, min(len(items), idx + window + 1)):
            el, _ = items[j]
            if isinstance(el, TextItem) and el.text.strip():
                parts.append(el.text.strip())
        return "\n".join(parts)

    def _describe_document_images(self, doc) -> VisionUsage:
        """Describe embedded images using the vision API with document context.

        Iterates over all ``PictureItem`` elements, builds a per-image prompt
        that includes the document title and surrounding text, and stores the
        result in ``item.meta.description``.

        Returns a :class:`VisionUsage` aggregating the token cost of every
        vision API call made (``call_count`` images, summed ``total_tokens``)
        so the caller can meter the otherwise-invisible LLM spend.
        """
        from docling.utils.api_image_request import api_image_request

        cfg = self.config
        api_url, headers, params = self._get_vision_api_config()

        doc_title = self._get_document_title(doc)
        items_list = list(doc.iterate_items())

        # Collect (item, image, prompt) tuples for all describable pictures
        tasks: list[tuple[PictureItem, PILImage.Image, str]] = []
        for idx, (element, _level) in enumerate(items_list):
            if not isinstance(element, PictureItem):
                continue
            if not self._should_describe(element):
                continue
            img = element.get_image(doc)
            if img is None:
                continue

            surrounding = self._get_surrounding_text(items_list, idx)
            prompt = self._build_contextual_prompt(doc_title, surrounding)
            tasks.append((element, img, prompt))

        if not tasks:
            return VisionUsage()

        def _describe(task: tuple[PictureItem, PILImage.Image, str]) -> int:
            """Describe one image; return the call's total token count (0 on failure)."""
            item, img, prompt = task
            try:
                text, num_tokens, _stop = api_image_request(
                    image=img,
                    prompt=prompt,
                    url=api_url,
                    timeout=cfg.picture_description_timeout,
                    headers=headers,
                    **params,
                )
                if text:
                    text = strip_think_tags(text).strip()
                    if item.meta is None:
                        item.meta = PictureMeta()
                    item.meta.description = DescriptionMetaField(text=text)
                return int(num_tokens or 0)
            except Exception as exc:
                logger.warning("Vision API call failed for image: %s", exc)
                return 0

        with ThreadPoolExecutor(max_workers=cfg.picture_description_concurrency) as pool:
            # pool.map returns per-task token counts in submission order; summing
            # is the thread-safe way to aggregate (no shared mutable accumulator).
            token_counts = list(pool.map(_describe, tasks))

        return VisionUsage(call_count=len(tasks), total_tokens=sum(token_counts))

    # -- Standalone image description ---------------------------------------

    def _describe_standalone_image(
        self,
        image: PILImage.Image,
        doc_title: str = "",
        surrounding_text: str = "",
    ) -> tuple[str | None, int]:
        """Send a standalone image to the vision API with context.

        Returns ``(description, total_tokens)``. ``description`` is the text or
        ``None`` on failure; ``total_tokens`` is the vision call's token cost
        (0 when the call failed or reported no usage).
        """
        from docling.utils.api_image_request import api_image_request

        api_url, headers, params = self._get_vision_api_config()
        prompt = self._build_contextual_prompt(doc_title, surrounding_text)
        try:
            text, num_tokens, _stop = api_image_request(
                image=image,
                prompt=prompt,
                url=api_url,
                timeout=self.config.picture_description_timeout,
                headers=headers,
                **params,
            )
            tokens = int(num_tokens or 0)
            if text:
                return strip_think_tags(text).strip(), tokens
            logger.warning("Vision API returned empty response for standalone image")
            return None, tokens
        except Exception as exc:
            logger.warning("Vision API call failed for standalone image: %s", exc)
            return None, 0

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
                page_break_placeholder=self.config.page_break_placeholder or None,
            ),
        )
        md_text = serializer.serialize().text
        if self.config.image_path_prefix:
            md_text = self._rewrite_image_paths(md_text, self.config.image_path_prefix)
        md_path.write_text(md_text, encoding="utf-8")
        return artifacts_dir

    # -- Page-order workaround ----------------------------------------------

    @staticmethod
    def _has_page_order_violation(doc) -> bool:
        """Detect cross-page element misordering in Docling's document tree.

        Returns ``True`` when iterating items in tree order yields elements
        whose page numbers go backwards — e.g. a page-2 element appears
        before all page-1 elements have been emitted.  This happens when
        Docling's layout model groups items from different pages into the
        same list or section node.
        """
        max_page = 0
        for element, _level in doc.iterate_items():
            prov = getattr(element, "prov", None)
            if not prov:
                continue
            page_no = prov[0].page_no
            if page_no < max_page:
                return True
            max_page = max(max_page, page_no)
        return False

    def _export_markdown_page_ordered(
        self,
        doc,
        page_break_placeholder: str = "",
        image_path_prefix: str = "",
    ) -> tuple[str, dict[str, str]]:
        """Export markdown in strict page order.

        Fallback for documents where Docling's tree structure produces wrong
        element ordering due to cross-page grouping.  Iterates elements
        page-by-page using ``iterate_items(page_no=…)`` and formats them
        as markdown, bypassing the tree-based serializer.

        Returns ``(markdown_text, image_names)``, where ``image_names`` maps each
        pictures's ``self_ref`` to the filename its reference uses.  The caller
        writes the files from that mapping, so the reference and the file it
        names come from *one* enumeration.  They used to come from two — this
        loop, page by page, and a separate tree-order loop for the files — which
        disagree by construction, since disagreeing about order is the entire
        reason this branch exists.
        """
        parts: list[str] = []
        sorted_pages = sorted(doc.pages.keys())
        image_names: dict[str, str] = {}
        pic_idx = 0

        for page_idx, page_no in enumerate(sorted_pages):
            if page_idx > 0 and page_break_placeholder:
                parts.append(page_break_placeholder)
                parts.append("")

            for element, _level in doc.iterate_items(page_no=page_no):
                # -- Pictures --
                if isinstance(element, PictureItem):
                    # An item whose provenance spans two pages is yielded by
                    # *both* page filters (docling matches with `any(prov.page_no
                    # in page_nrs …)`).  Reuse the name so it stays one file with
                    # one set of bytes, referenced from each page it appears on.
                    self_ref = getattr(element, "self_ref", None)
                    if self_ref is not None and self_ref in image_names:
                        filename = image_names[self_ref]
                    else:
                        filename = f"picture_{pic_idx:03d}.png"
                        pic_idx += 1
                        if self_ref is not None:
                            image_names[self_ref] = filename

                    if image_path_prefix:
                        prefix = image_path_prefix.rstrip("/")
                        img_ref = f"{prefix}/{filename}"
                    else:
                        img_ref = f"images/{filename}"

                    desc = ""
                    if (
                        element.meta is not None
                        and hasattr(element.meta, "description")
                        and element.meta.description is not None
                        and element.meta.description.text
                    ):
                        desc = strip_think_tags(
                            element.meta.description.text,
                        ).strip()

                    if desc:
                        summary, detail = _parse_description(desc)
                        alt = _sanitize_alt_text(summary) if summary else "Image"
                        parts.append(f"![{alt}]({img_ref})")
                        if detail:
                            blockquote = "\n".join(
                                f"> {line}" for line in detail.splitlines()
                            )
                            parts.append("")
                            parts.append(blockquote)
                    else:
                        parts.append(f"![Image]({img_ref})")

                    parts.append("")
                    continue

                # -- Tables --
                if isinstance(element, TableItem):
                    # TableItem has no ``text`` field at all, so the getattr
                    # default this used to take was always "" and every table in
                    # a page-misordered document was silently dropped — the whole
                    # grid, not just its formatting.  Tender documents and case
                    # files are exactly the kind that trip the fallback.
                    try:
                        text = element.export_to_markdown(doc).strip()
                    except Exception as tbl_err:  # pragma: no cover - defensive
                        logger.warning("Failed to render table: %s", tbl_err)
                        text = ""
                    if text:
                        parts.append(text)
                    parts.append("")
                    continue

                # -- Text / list items / headings --
                text = getattr(element, "text", "").strip()
                if not text:
                    continue

                label = getattr(element, "label", None)
                if label == DocItemLabel.TITLE:
                    parts.append(f"# {text}")
                elif label == DocItemLabel.SECTION_HEADER:
                    parts.append(f"## {text}")
                elif label == DocItemLabel.LIST_ITEM:
                    marker = getattr(element, "marker", "")
                    if marker:
                        parts.append(f"{marker} {text}")
                    else:
                        parts.append(f"- {text}")
                elif label == DocItemLabel.CAPTION:
                    parts.append(f"*{text}*")
                else:
                    parts.append(text)
                parts.append("")

        return "\n".join(parts), image_names

    @staticmethod
    def _write_page_ordered_images(
        doc,
        images_dir: Path,
        image_names: dict[str, str],
    ) -> None:
        """Write the picture files the page-ordered markdown references.

        ``image_names`` comes from ``_export_markdown_page_ordered`` and is keyed
        on each picture's ``self_ref``, so a file is named by the same pass that
        emitted the reference to it.  This used to be a second loop with its own
        counter over a *different* traversal (tree order, not page order) that
        advanced only on a successful save.  Both drifts are silent: the markdown
        stays well-formed while its figures carry other figures' bytes — and with
        AI descriptions on, the wrong description travels into the alt text and
        the embeddings alongside them.
        """
        for element, _level in doc.iterate_items():
            if not isinstance(element, PictureItem):
                continue
            filename = image_names.get(getattr(element, "self_ref", None))
            if filename is None:
                continue  # nothing references it; writing it would orphan a file
            try:
                img = element.get_image(doc)
                if img is not None:
                    img.save(images_dir / filename, format="PNG")
            except Exception as img_err:
                logger.warning("Failed to save image %s: %s", filename, img_err)
                # A half-written PNG is worse than none: callers upload whatever
                # is in the directory, not whatever the markdown references.
                (images_dir / filename).unlink(missing_ok=True)

    # -- Engine resolution ----------------------------------------------------

    def _resolve_engine(self, source_str: str, is_url: bool) -> str:
        """Decide which conversion engine to use.

        Returns ``"pymupdf4llm"`` or ``"docling"``.
        """
        engine = self.config.engine.lower().strip()

        if engine == "pymupdf4llm":
            try:
                import pymupdf4llm as _  # noqa: F401
            except ImportError:
                raise ImportError(
                    "pymupdf4llm is not installed. "
                    "Install it with: pip install doc2md[pymupdf]"
                ) from None
            return "pymupdf4llm"

        if engine == "docling":
            return "docling"

        # Auto mode
        if is_url:
            return "docling"

        ext = Path(source_str).suffix.lstrip(".").lower()
        if ext not in _PDF_EXTENSIONS:
            return "docling"

        try:
            import pymupdf4llm as _  # noqa: F401
        except ImportError:
            return "docling"

        from doc2md.analyzer import analyze_pdf

        analysis = analyze_pdf(source_str)
        self._last_analysis = analysis
        if analysis.is_scanned:
            return "docling"
        if analysis.has_images:
            return "docling"
        if analysis.has_tables:
            return "docling"

        return "pymupdf4llm"

    def _resolve_do_ocr(self, source_str: str, is_url: bool) -> bool:
        """Decide whether to enable OCR for this document.

        Returns ``True`` when OCR should be enabled.
        """
        if self.config.force_full_page_ocr:
            return True  # explicit full-page OCR always implies do_ocr

        setting = self.config.do_ocr.lower().strip()
        if setting in ("true", "yes", "1"):
            return True
        if setting in ("false", "no", "0"):
            return False

        # Auto mode: enable OCR only when needed

        if is_url:
            return True  # can't pre-analyse remote files

        ext = Path(source_str).suffix.lstrip(".").lower()
        if ext in _IMAGE_EXTENSIONS:
            return True  # standalone images need OCR

        # Use the analysis from _resolve_engine if available
        if self._last_analysis is not None:
            return self._last_analysis.is_scanned

        return False

    # -- PyMuPDF4LLM conversion path ----------------------------------------

    def _convert_pymupdf(
        self,
        source_str: str,
        output_dir: Path,
        source_name: str,
        result: ConversionResult,
    ) -> None:
        """Convert a text-only PDF using pymupdf4llm.

        This is a fast path for simple PDFs without images or complex tables.
        Images and tables are handled by the Docling engine instead.
        """
        import pymupdf4llm

        cfg = self.config
        md_path = output_dir / "output.md"

        with timed() as timing:
            chunks = pymupdf4llm.to_markdown(
                source_str,
                page_chunks=True,
                write_images=False,
                ignore_images=True,
            )

        result.metadata.timing = TimingInfo(
            start=timing.start,
            end=timing.end,
            elapsed_seconds=timing.elapsed_seconds,
        )

        # Join page chunks with optional page-break placeholder
        separator = "\n\n"
        if cfg.page_break_placeholder:
            separator = f"\n\n{cfg.page_break_placeholder}\n\n"
        md_text = separator.join(chunk["text"] for chunk in chunks)

        # Count elements from chunks
        counts = ElementCounts()
        counts.pages = len(chunks)
        result.metadata.elements = counts

        md_path.write_text(md_text, encoding="utf-8")
        result.markdown_path = md_path
        result.json_path = None
        result.success = True

        logger.info(
            "Converted %s (pymupdf4llm) in %.1fs (%d pages)",
            source_name,
            timing.elapsed_seconds,
            counts.pages,
        )

    # -- Docling conversion path --------------------------------------------

    def _convert_docling(
        self,
        source_str: str,
        output_dir: Path,
        source_name: str,
        is_url: bool,
        result: ConversionResult,
    ) -> None:
        """Convert a document using the Docling pipeline."""
        do_ocr = self._resolve_do_ocr(source_str, is_url)
        converter = self._get_converter(do_ocr)
        logger.info("OCR %s for %s", "enabled" if do_ocr else "disabled", source_name)

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

        # Post-processing: describe images with document context
        if cfg.do_picture_description and not self._is_standalone_image(source_str):
            result.metadata.vision_usage = self._describe_document_images(
                conv_res.document,
            )

        # Detect cross-page ordering issues in Docling's document tree.
        _page_misordered = self._has_page_order_violation(conv_res.document)
        if _page_misordered and not conv_res.document.pages:
            # The fallback walks sorted(doc.pages) — with no pages it emits an
            # empty document and loses everything. iterate_items(page_no=…) does
            # not validate the page number, so nothing would have raised.
            logger.warning(
                "Page-order issue in %s but the document has no pages; "
                "keeping the tree-ordered export",
                source_name,
            )
            _page_misordered = False
        if _page_misordered:
            logger.warning(
                "Detected page-order issue in %s; using page-ordered fallback",
                source_name,
            )

        if _page_misordered:
            md_text, image_names = self._export_markdown_page_ordered(
                conv_res.document,
                page_break_placeholder=cfg.page_break_placeholder,
                image_path_prefix=cfg.image_path_prefix,
            )
            md_path.write_text(md_text, encoding="utf-8")
            if cfg.generate_images or cfg.do_picture_description:
                images_dir = output_dir / "images"
                images_dir.mkdir(exist_ok=True)
                result.images_dir = images_dir
                self._write_page_ordered_images(
                    conv_res.document, images_dir, image_names,
                )
        elif cfg.generate_images or cfg.do_picture_description:
            # No separate images/ directory here. Docling's own serializer both
            # writes the files and emits the references, into
            # ``<stem>_artifacts`` under names like image_000000_<hash>.png — so
            # the picture_NNN.png copies this used to write alongside them were
            # never referenced by the markdown, and left result.images_dir
            # pointing at a directory whose filenames matched nothing whenever
            # image_path_prefix was empty.
            if cfg.do_picture_description:
                artifacts_dir = self._export_markdown_with_descriptions(
                    conv_res.document, md_path,
                )
                result.images_dir = artifacts_dir
            else:
                conv_res.document.save_as_markdown(
                    md_path,
                    image_mode=ImageRefMode.REFERENCED,
                    page_break_placeholder=cfg.page_break_placeholder or None,
                )
                if cfg.image_path_prefix:
                    md_text = md_path.read_text(encoding="utf-8")
                    md_text = self._rewrite_image_paths(
                        md_text, cfg.image_path_prefix,
                    )
                    md_path.write_text(md_text, encoding="utf-8")
                artifacts_dir = output_dir / f"{md_path.stem}_artifacts"
                if artifacts_dir.exists():
                    result.images_dir = artifacts_dir
        else:
            md_text = conv_res.document.export_to_markdown(
                page_break_placeholder=cfg.page_break_placeholder or None,
            )
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
            ocr_text = docling_md.strip()
            made_vision_call = False
            vision_tokens = 0
            try:
                image = PILImage.open(source_path).convert("RGB")
                made_vision_call = True
                description, vision_tokens = self._describe_standalone_image(
                    image,
                    doc_title=source_path.stem,
                    surrounding_text=ocr_text if len(ocr_text) > 10 else "",
                )
            except Exception as img_exc:
                logger.warning(
                    "Failed to open image for description: %s", img_exc,
                )
                description = None
            result.metadata.vision_usage = VisionUsage(
                call_count=1 if made_vision_call else 0,
                total_tokens=int(vision_tokens or 0),
            )
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
            "Converted %s (docling) in %.1fs (%d pages, %d tables, %d pictures)",
            source_name,
            timing.elapsed_seconds,
            counts.pages,
            counts.tables,
            counts.pictures,
        )

    # -- Main entry point ---------------------------------------------------

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
            self._last_analysis = None
            engine = self._resolve_engine(source_str, is_url)
            result.metadata.engine_used = engine
            logger.info("Using engine: %s for %s", engine, source_name)

            if engine == "pymupdf4llm":
                self._convert_pymupdf(source_str, output_dir, source_name, result)
            else:
                self._convert_docling(source_str, output_dir, source_name, is_url, result)

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
