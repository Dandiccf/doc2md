"""Helpers: logging, timing, metadata."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package root logger."""
    logging.basicConfig(format=LOG_FORMAT, level=level)
    pkg_logger = logging.getLogger("doc2md")
    pkg_logger.setLevel(level)
    return pkg_logger


logger = setup_logging()


@dataclass
class TimingInfo:
    """Stores elapsed time for a conversion."""

    start: float = 0.0
    end: float = 0.0
    elapsed_seconds: float = 0.0


@contextmanager
def timed() -> Generator[TimingInfo, None, None]:
    """Context manager that records wall-clock time."""
    info = TimingInfo()
    info.start = time.time()
    try:
        yield info
    finally:
        info.end = time.time()
        info.elapsed_seconds = round(info.end - info.start, 3)


@dataclass
class ElementCounts:
    """Counts of document elements found during conversion."""

    pages: int = 0
    tables: int = 0
    pictures: int = 0
    text_items: int = 0


@dataclass
class ConversionMetadata:
    """Full metadata for a single conversion run."""

    document: str = ""
    timing: TimingInfo = field(default_factory=TimingInfo)
    elements: ElementCounts = field(default_factory=ElementCounts)
    error: str | None = None
    docling_version: str = ""
    config: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> ConversionMetadata:
        data = json.loads(path.read_text())
        meta = cls()
        meta.document = data.get("document", "")
        meta.timing = TimingInfo(**data.get("timing", {}))
        meta.elements = ElementCounts(**data.get("elements", {}))
        meta.error = data.get("error")
        meta.docling_version = data.get("docling_version", "")
        meta.config = data.get("config", {})
        return meta
