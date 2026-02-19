"""Tests for standalone image description enhancement."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from doc2md import DocumentPipeline, PipelineConfig, convert


# ---------------------------------------------------------------------------
# Unit tests (no API key, no conversion needed)
# ---------------------------------------------------------------------------


class TestIsStandaloneImage:

    @pytest.mark.parametrize("ext", ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"])
    def test_image_extensions(self, ext: str) -> None:
        assert DocumentPipeline._is_standalone_image(f"/path/to/photo.{ext}") is True

    @pytest.mark.parametrize("ext", ["JPG", "PNG", "WEBP", "Jpeg"])
    def test_case_insensitive(self, ext: str) -> None:
        assert DocumentPipeline._is_standalone_image(f"image.{ext}") is True

    @pytest.mark.parametrize("ext", ["pdf", "docx", "html", "txt", "csv", "md"])
    def test_non_image_extensions(self, ext: str) -> None:
        assert DocumentPipeline._is_standalone_image(f"file.{ext}") is False

    def test_no_extension(self) -> None:
        assert DocumentPipeline._is_standalone_image("noext") is False


class TestBuildStandaloneImageMarkdown:

    @pytest.fixture()
    def pipeline(self) -> DocumentPipeline:
        return DocumentPipeline(config=PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
        ))

    @pytest.fixture()
    def source_image(self, tmp_path: Path) -> Path:
        """Create a tiny test image."""
        img = PILImage.new("RGB", (10, 10), "red")
        p = tmp_path / "test_photo.jpg"
        img.save(p, "JPEG")
        return p

    def test_no_description_no_ocr(self, pipeline: DocumentPipeline, source_image: Path, tmp_path: Path) -> None:
        md, images_dir = pipeline._build_standalone_image_markdown(
            source_image, "", None, tmp_path / "out",
        )
        assert "![Image](images/test_photo.jpg)" in md
        assert "---" not in md
        assert (images_dir / "test_photo.jpg").exists()

    def test_no_description_with_ocr(self, pipeline: DocumentPipeline, source_image: Path, tmp_path: Path) -> None:
        md, _ = pipeline._build_standalone_image_markdown(
            source_image, "Some meaningful OCR text here", None, tmp_path / "out",
        )
        assert "![Image](images/test_photo.jpg)" in md
        assert "---" in md
        assert "Some meaningful OCR text here" in md

    def test_no_description_trivial_ocr(self, pipeline: DocumentPipeline, source_image: Path, tmp_path: Path) -> None:
        """OCR text <= 10 chars should be suppressed."""
        md, _ = pipeline._build_standalone_image_markdown(
            source_image, "short", None, tmp_path / "out",
        )
        assert "---" not in md
        assert "short" not in md

    def test_plain_text_description(self, pipeline: DocumentPipeline, source_image: Path, tmp_path: Path) -> None:
        md, _ = pipeline._build_standalone_image_markdown(
            source_image, "", "A detailed analysis of the image content.", tmp_path / "out",
        )
        # Plain text → generic alt, detail in blockquote
        assert "![Image](images/test_photo.jpg)" in md
        assert "> A detailed analysis of the image content." in md

    def test_structured_json_description(self, pipeline: DocumentPipeline, source_image: Path, tmp_path: Path) -> None:
        desc = json.dumps({
            "summary": "A red square image.",
            "detail": "The image shows a solid red square with no other content.",
        })
        md, _ = pipeline._build_standalone_image_markdown(
            source_image, "", desc, tmp_path / "out",
        )
        assert "![A red square image.](images/test_photo.jpg)" in md
        assert "> The image shows a solid red square with no other content." in md

    def test_image_path_prefix(self, source_image: Path, tmp_path: Path) -> None:
        pipeline = DocumentPipeline(config=PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
            image_path_prefix="http://localhost:8000/files/docs/123",
        ))
        md, _ = pipeline._build_standalone_image_markdown(
            source_image, "", None, tmp_path / "out",
        )
        assert "![Image](http://localhost:8000/files/docs/123/test_photo.jpg)" in md

    def test_description_with_ocr(self, pipeline: DocumentPipeline, source_image: Path, tmp_path: Path) -> None:
        desc = json.dumps({
            "summary": "A chart showing growth.",
            "detail": "Revenue grew 25% year over year.",
        })
        md, _ = pipeline._build_standalone_image_markdown(
            source_image, "Revenue: $1.2M  Growth: 25%", desc, tmp_path / "out",
        )
        assert "![A chart showing growth.](images/test_photo.jpg)" in md
        assert "> Revenue grew 25% year over year." in md
        assert "---" in md
        assert "Revenue: $1.2M  Growth: 25%" in md


class TestDescribeStandaloneImage:

    def test_success(self) -> None:
        pipeline = DocumentPipeline(config=PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
        ))
        image = PILImage.new("RGB", (10, 10), "blue")

        mock_stop = MagicMock()
        with patch("doc2md.converter.api_image_request", create=True):
            with patch(
                "docling.utils.api_image_request.api_image_request",
                return_value=("A blue square.", 50, mock_stop),
            ):
                result = pipeline._describe_standalone_image(image)

        assert result == "A blue square."

    def test_strips_think_tags(self) -> None:
        pipeline = DocumentPipeline(config=PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
        ))
        image = PILImage.new("RGB", (10, 10), "blue")

        mock_stop = MagicMock()
        with patch(
            "docling.utils.api_image_request.api_image_request",
            return_value=("<think>reasoning</think>A blue square.", 50, mock_stop),
        ):
            result = pipeline._describe_standalone_image(image)

        assert result == "A blue square."

    def test_api_failure_returns_none(self) -> None:
        pipeline = DocumentPipeline(config=PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
        ))
        image = PILImage.new("RGB", (10, 10), "blue")

        with patch(
            "docling.utils.api_image_request.api_image_request",
            side_effect=ConnectionError("timeout"),
        ):
            result = pipeline._describe_standalone_image(image)

        assert result is None

    def test_empty_response_returns_none(self) -> None:
        pipeline = DocumentPipeline(config=PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
        ))
        image = PILImage.new("RGB", (10, 10), "blue")

        mock_stop = MagicMock()
        with patch(
            "docling.utils.api_image_request.api_image_request",
            return_value=("", 0, mock_stop),
        ):
            result = pipeline._describe_standalone_image(image)

        assert result is None


# ---------------------------------------------------------------------------
# Integration: convert() with standalone image (mocked vision API)
# ---------------------------------------------------------------------------


class TestConvertStandaloneImage:

    @pytest.fixture()
    def source_image(self, tmp_path: Path) -> Path:
        img = PILImage.new("RGB", (100, 100), "green")
        p = tmp_path / "photo.png"
        img.save(p, "PNG")
        return p

    def test_convert_with_description_enabled(self, source_image: Path, output_dir: Path) -> None:
        """Standalone image + do_picture_description=True → vision API called, markdown enhanced."""
        desc = json.dumps({
            "summary": "A green square.",
            "detail": "Solid green image with no other elements.",
        })
        mock_stop = MagicMock()

        with patch(
            "docling.utils.api_image_request.api_image_request",
            return_value=(desc, 80, mock_stop),
        ) as mock_api:
            result = convert(
                str(source_image),
                output_dir=output_dir,
                config=PipelineConfig(
                    do_picture_description=True,
                    do_picture_classification=False,
                    structured_description=True,
                    picture_description_provider="local",
                    local_url="http://fake:11434/v1/chat/completions",
                    local_model="test-model",
                ),
            )

        assert result.success, f"Conversion failed: {result.error}"
        assert mock_api.called

        md = result.markdown_path.read_text(encoding="utf-8")
        assert "![A green square.](images/photo.png)" in md or "![A green square.](" in md
        assert "> Solid green image with no other elements." in md
        assert result.images_dir is not None
        assert (result.images_dir / "photo.png").exists()

    def test_convert_with_description_disabled(self, source_image: Path, output_dir: Path) -> None:
        """Standalone image + do_picture_description=False → no API call, image still copied."""
        result = convert(
            str(source_image),
            output_dir=output_dir,
            config=PipelineConfig(
                do_picture_description=False,
                do_picture_classification=False,
                generate_images=True,
            ),
        )

        assert result.success, f"Conversion failed: {result.error}"

        md = result.markdown_path.read_text(encoding="utf-8")
        assert "![Image](" in md
        assert "photo.png" in md
        assert result.images_dir is not None
        assert (result.images_dir / "photo.png").exists()

    def test_convert_vision_api_failure_fallback(self, source_image: Path, output_dir: Path) -> None:
        """If vision API fails, output should still have image reference + OCR text."""
        with patch(
            "docling.utils.api_image_request.api_image_request",
            side_effect=ConnectionError("server down"),
        ):
            result = convert(
                str(source_image),
                output_dir=output_dir,
                config=PipelineConfig(
                    do_picture_description=True,
                    do_picture_classification=False,
                    picture_description_provider="local",
                    local_url="http://fake:11434/v1/chat/completions",
                    local_model="test-model",
                ),
            )

        assert result.success, f"Conversion failed: {result.error}"

        md = result.markdown_path.read_text(encoding="utf-8")
        # Should still have image reference even without description
        assert "![Image](" in md
        assert "photo.png" in md


# ---------------------------------------------------------------------------
# Integration: real conversion with API key (gated)
# ---------------------------------------------------------------------------


class TestStandaloneImageWithAPI:
    """End-to-end test with the real vision API. Skipped without OPENAI_API_KEY."""

    @pytest.fixture(autouse=True)
    def _skip_without_key(self, has_openai_key: bool) -> None:
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

    def test_real_image_description(self, sample_image: Path, output_dir: Path) -> None:
        config = PipelineConfig(structured_description=True)
        result = convert(str(sample_image), output_dir=output_dir, config=config)

        assert result.success, f"Conversion failed: {result.error}"
        md = result.markdown_path.read_text(encoding="utf-8")

        # Should have an image tag with a descriptive alt text (not just "Image")
        assert "!["  in md
        assert "](images/" in md or "](" in md
        # Should have a blockquote description
        assert "> " in md
        assert result.images_dir is not None
