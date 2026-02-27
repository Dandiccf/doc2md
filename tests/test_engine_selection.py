"""Tests for engine selection, PDF analysis, and pymupdf4llm conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from doc2md import DocumentPipeline, PipelineConfig
from doc2md.analyzer import PdfAnalysis, analyze_pdf


# ---------------------------------------------------------------------------
# Engine resolution tests (no conversion, no API key)
# ---------------------------------------------------------------------------


class TestEngineResolution:
    """Unit tests for _resolve_engine() logic."""

    def _make_pipeline(self, **kwargs) -> DocumentPipeline:
        return DocumentPipeline(
            config=PipelineConfig(do_picture_description=False, **kwargs),
        )

    def test_explicit_docling(self) -> None:
        pipeline = self._make_pipeline(engine="docling")
        assert pipeline._resolve_engine("/tmp/test.pdf", is_url=False) == "docling"

    def test_explicit_pymupdf4llm(self) -> None:
        """Explicit pymupdf4llm works if installed."""
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            pytest.skip("pymupdf4llm not installed")
        pipeline = self._make_pipeline(engine="pymupdf4llm")
        assert pipeline._resolve_engine("/tmp/test.pdf", is_url=False) == "pymupdf4llm"

    def test_explicit_pymupdf4llm_not_installed(self) -> None:
        """Explicit pymupdf4llm raises ImportError if not installed."""
        pipeline = self._make_pipeline(engine="pymupdf4llm")
        with patch.dict("sys.modules", {"pymupdf4llm": None}):
            with pytest.raises(ImportError, match="pymupdf4llm is not installed"):
                pipeline._resolve_engine("/tmp/test.pdf", is_url=False)

    def test_auto_url_uses_docling(self) -> None:
        pipeline = self._make_pipeline(engine="auto")
        assert pipeline._resolve_engine("https://example.com/doc.pdf", is_url=True) == "docling"

    def test_auto_non_pdf_uses_docling(self) -> None:
        pipeline = self._make_pipeline(engine="auto")
        assert pipeline._resolve_engine("/tmp/doc.docx", is_url=False) == "docling"
        assert pipeline._resolve_engine("/tmp/doc.html", is_url=False) == "docling"
        assert pipeline._resolve_engine("/tmp/doc.pptx", is_url=False) == "docling"

    def test_auto_pymupdf_not_installed_uses_docling(self) -> None:
        pipeline = self._make_pipeline(engine="auto")
        with patch.dict("sys.modules", {"pymupdf4llm": None}):
            assert pipeline._resolve_engine("/tmp/test.pdf", is_url=False) == "docling"

    def test_auto_scanned_pdf_uses_docling(self) -> None:
        pipeline = self._make_pipeline(engine="auto")
        with patch("doc2md.analyzer.analyze_pdf") as mock_analyze:
            mock_analyze.return_value = PdfAnalysis(is_scanned=True)
            try:
                import pymupdf4llm  # noqa: F401
            except ImportError:
                pytest.skip("pymupdf4llm not installed")
            assert pipeline._resolve_engine("/tmp/scan.pdf", is_url=False) == "docling"

    def test_auto_pdf_with_images_uses_docling(self) -> None:
        """PDFs with significant images should route to Docling."""
        pipeline = self._make_pipeline(engine="auto")
        with patch("doc2md.analyzer.analyze_pdf") as mock_analyze:
            mock_analyze.return_value = PdfAnalysis(is_scanned=False, has_images=True)
            try:
                import pymupdf4llm  # noqa: F401
            except ImportError:
                pytest.skip("pymupdf4llm not installed")
            assert pipeline._resolve_engine("/tmp/images.pdf", is_url=False) == "docling"

    def test_auto_pdf_with_tables_uses_docling(self) -> None:
        """PDFs with tables should route to Docling."""
        pipeline = self._make_pipeline(engine="auto")
        with patch("doc2md.analyzer.analyze_pdf") as mock_analyze:
            mock_analyze.return_value = PdfAnalysis(is_scanned=False, has_images=False, has_tables=True)
            try:
                import pymupdf4llm  # noqa: F401
            except ImportError:
                pytest.skip("pymupdf4llm not installed")
            assert pipeline._resolve_engine("/tmp/tables.pdf", is_url=False) == "docling"

    def test_auto_text_only_pdf_uses_pymupdf(self) -> None:
        """Simple text-only PDFs (no images, no tables) should use pymupdf4llm."""
        pipeline = self._make_pipeline(engine="auto")
        with patch("doc2md.analyzer.analyze_pdf") as mock_analyze:
            mock_analyze.return_value = PdfAnalysis(is_scanned=False, has_images=False, has_tables=False)
            try:
                import pymupdf4llm  # noqa: F401
            except ImportError:
                pytest.skip("pymupdf4llm not installed")
            assert pipeline._resolve_engine("/tmp/text.pdf", is_url=False) == "pymupdf4llm"


# ---------------------------------------------------------------------------
# PDF analyzer tests
# ---------------------------------------------------------------------------


class TestPdfAnalyzer:
    """Tests for analyze_pdf()."""

    def test_analyze_text_pdf(self, sample_pdf: Path) -> None:
        """Academic paper PDF should be detected as text-based with images and tables."""
        analysis = analyze_pdf(sample_pdf)
        assert analysis.total_pages > 0
        assert analysis.total_chars > 0
        assert analysis.avg_chars_per_page > 100
        assert analysis.is_scanned is False
        assert analysis.has_images is True  # academic paper has figures
        assert analysis.has_tables is True  # academic paper has data tables

    def test_analyze_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent file should return safe fallback (is_scanned=True)."""
        analysis = analyze_pdf(tmp_path / "nonexistent.pdf")
        assert analysis.is_scanned is True

    def test_analyze_invalid_file(self, tmp_path: Path) -> None:
        """Invalid file should return safe fallback (is_scanned=True)."""
        bad_file = tmp_path / "bad.pdf"
        bad_file.write_text("not a pdf")
        analysis = analyze_pdf(bad_file)
        assert analysis.is_scanned is True


# ---------------------------------------------------------------------------
# PyMuPDF4LLM conversion tests
# ---------------------------------------------------------------------------


class TestPyMuPDFConversion:
    """Integration tests for pymupdf4llm conversion path."""

    @pytest.fixture(autouse=True)
    def _skip_without_pymupdf(self) -> None:
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            pytest.skip("pymupdf4llm not installed")

    def test_convert_pdf_pymupdf(self, sample_pdf: Path, output_dir: Path) -> None:
        """Convert a PDF with pymupdf4llm and verify text-only outputs."""
        config = PipelineConfig(
            engine="pymupdf4llm",
            do_picture_description=False,
            do_picture_classification=False,
        )
        pipeline = DocumentPipeline(config=config)
        result = pipeline.convert(sample_pdf, output_dir=output_dir)

        assert result.success, f"Conversion failed: {result.error}"
        assert result.markdown_path is not None
        assert result.markdown_path.exists()
        assert result.markdown_path.stat().st_size > 0

        # pymupdf4llm text-only path: no JSON export, no images
        assert result.json_path is None
        assert result.images_dir is None

        # Metadata should be saved
        assert result.metadata_path is not None
        assert result.metadata_path.exists()
        assert result.metadata.engine_used == "pymupdf4llm"
        assert result.metadata.elements.pages > 0

    def test_auto_selects_docling_for_pdf_with_images(self, sample_pdf: Path, output_dir: Path) -> None:
        """Auto mode should select docling for a PDF with images (like an academic paper)."""
        config = PipelineConfig(
            engine="auto",
            do_picture_description=False,
            do_picture_classification=False,
        )
        pipeline = DocumentPipeline(config=config)
        result = pipeline.convert(sample_pdf, output_dir=output_dir)

        assert result.success, f"Conversion failed: {result.error}"
        assert result.metadata.engine_used == "docling"

    def test_metadata_includes_engine_used(self, sample_pdf: Path, output_dir: Path) -> None:
        """metadata.json should contain engine_used field."""
        import json

        config = PipelineConfig(
            engine="pymupdf4llm",
            do_picture_description=False,
        )
        pipeline = DocumentPipeline(config=config)
        result = pipeline.convert(sample_pdf, output_dir=output_dir)

        assert result.success
        meta_data = json.loads(result.metadata_path.read_text())
        assert meta_data["engine_used"] == "pymupdf4llm"

    def test_page_break_placeholder(self, sample_pdf: Path, output_dir: Path) -> None:
        """Page break placeholder should appear between pages."""
        config = PipelineConfig(
            engine="pymupdf4llm",
            do_picture_description=False,
            page_break_placeholder="<!-- page-break -->",
        )
        pipeline = DocumentPipeline(config=config)
        result = pipeline.convert(sample_pdf, output_dir=output_dir)

        assert result.success
        md_text = result.markdown_path.read_text(encoding="utf-8")
        if result.metadata.elements.pages > 1:
            assert "<!-- page-break -->" in md_text
