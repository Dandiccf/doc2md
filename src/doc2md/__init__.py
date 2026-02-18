"""doc2md - Standalone document-to-markdown conversion pipeline."""

from pathlib import Path

__version__ = "0.2.0"

from doc2md.config import PipelineConfig
from doc2md.converter import ConversionResult, DocumentPipeline
from doc2md.utils import setup_logging

__all__ = [
    "__version__",
    "convert",
    "DocumentPipeline",
    "ConversionResult",
    "PipelineConfig",
    "setup_logging",
]


def convert(
    source: str | Path,
    output_dir: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> ConversionResult:
    """One-liner convenience function to convert a document to markdown.

    Args:
        source: File path or URL to the document.
        output_dir: Directory for outputs. Auto-generated if None.
        config: Pipeline configuration. Uses optimal defaults if None.

    Returns:
        ConversionResult with paths to all generated files.
    """
    pipeline = DocumentPipeline(config=config)
    return pipeline.convert(source, output_dir=output_dir)
