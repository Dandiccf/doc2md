"""Integration tests for doc2md conversion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc2md import DocumentPipeline, PipelineConfig, convert


# ---------------------------------------------------------------------------
# Basic conversion (no API key required)
# ---------------------------------------------------------------------------


class TestBasicConversion:
    """Tests that run without an OpenAI API key."""

    @pytest.fixture(scope="class")
    def basic_config(self) -> PipelineConfig:
        return PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
            generate_images=True,
        )

    def test_convert_pdf(self, sample_pdf: Path, output_dir: Path, basic_config: PipelineConfig) -> None:
        """Convert a PDF and verify all expected outputs are produced."""
        result = convert(str(sample_pdf), output_dir=output_dir, config=basic_config)

        assert result.success, f"Conversion failed: {result.error}"
        assert result.markdown_path is not None
        assert result.markdown_path.exists()
        assert result.markdown_path.stat().st_size > 0

        assert result.json_path is not None
        assert result.json_path.exists()

        assert result.metadata_path is not None
        assert result.metadata_path.exists()

        # Check metadata values
        assert result.metadata.document == sample_pdf.name
        assert result.metadata.elements.pages > 0
        assert result.metadata.timing.elapsed_seconds > 0

    def test_convert_image(self, sample_image: Path, output_dir: Path, basic_config: PipelineConfig) -> None:
        """Convert an image file via OCR."""
        result = convert(str(sample_image), output_dir=output_dir, config=basic_config)

        assert result.success, f"Conversion failed: {result.error}"
        assert result.markdown_path is not None
        assert result.markdown_path.exists()
        assert result.markdown_path.read_text(encoding="utf-8").strip()

    def test_pipeline_reuse(self, sample_pdf: Path, sample_image: Path, tmp_path: Path, basic_config: PipelineConfig) -> None:
        """Create one pipeline and convert multiple documents with it."""
        pipeline = DocumentPipeline(config=basic_config)

        r1 = pipeline.convert(sample_pdf, output_dir=tmp_path / "pdf_out")
        r2 = pipeline.convert(sample_image, output_dir=tmp_path / "img_out")

        assert r1.success, f"PDF conversion failed: {r1.error}"
        assert r2.success, f"Image conversion failed: {r2.error}"

    def test_auto_output_dir(self, sample_pdf: Path, basic_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When output_dir is None, an auto-generated directory is used."""
        monkeypatch.chdir(tmp_path)
        result = convert(str(sample_pdf), config=basic_config)

        assert result.success, f"Conversion failed: {result.error}"
        assert result.markdown_path is not None
        # Auto-generated dir should be <stem>_output in cwd
        assert "_output" in str(result.markdown_path.parent)


# ---------------------------------------------------------------------------
# URL conversion (no API key required)
# ---------------------------------------------------------------------------


class TestURLConversion:

    def test_convert_url(self, output_dir: Path) -> None:
        """Convert a document directly from a URL."""
        config = PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
            generate_images=False,
        )
        result = convert(
            "https://arxiv.org/pdf/2206.01062",
            output_dir=output_dir,
            config=config,
        )

        assert result.success, f"URL conversion failed: {result.error}"
        assert result.markdown_path is not None
        assert result.markdown_path.exists()


# ---------------------------------------------------------------------------
# Full pipeline with image descriptions (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


class TestWithImageDescriptions:
    """Tests that require an OpenAI API key (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _skip_without_key(self, has_openai_key: bool) -> None:
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

    def test_convert_with_descriptions(self, sample_pdf: Path, output_dir: Path) -> None:
        """Full pipeline: OCR + images + OpenAI descriptions."""
        config = PipelineConfig()  # defaults = optimal profile with descriptions
        result = convert(str(sample_pdf), output_dir=output_dir, config=config)

        assert result.success, f"Conversion failed: {result.error}"
        assert result.markdown_path is not None

        md_text = result.markdown_path.read_text(encoding="utf-8")
        # Descriptions should appear as blockquotes after images
        if result.metadata.elements.pictures > 0:
            assert ">" in md_text, "Expected blockquote descriptions in markdown"

        assert result.images_dir is not None
        assert result.images_dir.exists()


# ---------------------------------------------------------------------------
# Config tests (no conversion, no API key)
# ---------------------------------------------------------------------------


class TestConfig:

    def test_defaults_match_optimal(self) -> None:
        """PipelineConfig defaults should match the optimal profile values."""
        cfg = PipelineConfig(do_picture_description=False)
        assert cfg.ocr_engine == "ocrmac"
        assert cfg.ocr_lang == ["en-US"]
        assert cfg.table_mode == "accurate"
        assert cfg.generate_images is True
        assert cfg.images_scale == 2.0
        assert cfg.do_cell_matching is True

    def test_override(self) -> None:
        """Config fields can be overridden."""
        cfg = PipelineConfig(
            ocr_engine="easyocr",
            ocr_lang=["en"],
            table_mode="fast",
            do_picture_description=False,
        )
        assert cfg.ocr_engine == "easyocr"
        assert cfg.ocr_lang == ["en"]
        assert cfg.table_mode == "fast"

    def test_to_dict(self) -> None:
        cfg = PipelineConfig(do_picture_description=False)
        d = cfg.to_dict()
        assert d["ocr_engine"] == "ocrmac"
        assert isinstance(d["allowed_formats"], list)

    def test_no_api_key_error_when_descriptions_disabled(self) -> None:
        """No error if picture descriptions are off, even without API key."""
        cfg = PipelineConfig(do_picture_description=False)
        assert cfg.openai_api_key == ""
