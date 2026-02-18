"""Shared fixtures for doc2md integration tests."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).parent / "samples"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

SAMPLE_URLS = {
    "academic_paper.pdf": "https://arxiv.org/pdf/2206.01062",
    "sample_image.jpg": (
        "https://www.archives.gov/files/founding-docs/"
        "declaration_of_independence_stone_630.jpg"
    ),
}


def _download(name: str, url: str) -> Path:
    """Download a sample file if not already cached."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    dest = SAMPLES_DIR / name
    if dest.exists():
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "doc2md-tests/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def pytest_configure(config: pytest.Config) -> None:
    """Clear previous test output at the start of each test run."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """Path to a small academic PDF (downloaded once per session)."""
    return _download("academic_paper.pdf", SAMPLE_URLS["academic_paper.pdf"])


@pytest.fixture(scope="session")
def sample_image() -> Path:
    """Path to a sample image for OCR testing."""
    return _download("sample_image.jpg", SAMPLE_URLS["sample_image.jpg"])


@pytest.fixture()
def output_dir(request: pytest.FixtureRequest) -> Path:
    """Persistent output directory named after the test, under tests/test_output/."""
    # e.g. tests/test_output/TestBasicConversion__test_convert_pdf/
    node_name = request.node.name
    cls = request.node.getparent(pytest.Class)
    if cls is not None:
        dir_name = f"{cls.name}__{node_name}"
    else:
        dir_name = node_name
    out = TEST_OUTPUT_DIR / dir_name
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def has_openai_key() -> bool:
    """Whether an OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))
