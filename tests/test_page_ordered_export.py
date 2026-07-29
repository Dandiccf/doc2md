"""Regression tests for the page-ordered export fallback.

``_export_markdown_page_ordered`` runs whenever Docling's layout model groups
elements from different pages into one tree node, so tree order no longer
matches reading order. Two defects lived in that branch:

1. Every table was silently dropped. ``TableItem`` has no ``text`` field, so the
   ``getattr(element, "text", "")`` this used took its default every time —
   losing the whole grid, not just its formatting. Tender documents and case
   files are exactly the kind that trip the fallback.

2. The markdown's image references and the image files on disk came from two
   *different* enumerations — page-by-page for the references, whole-tree for the
   files — so index *n* in the markdown was not index *n* on disk whenever those
   disagreed, which is the entire reason the branch exists. A second drift came
   from the file loop advancing its counter only on a successful save, so one
   unreadable picture shifted every later pairing by one.

Both are silent: no error, no warning, well-formed markdown. With AI image
descriptions on, the wrong description travels into the alt text and the
embeddings along with the wrong picture.

These use synthetic DoclingDocuments rather than a sample PDF: the pairing is
only observable when tree order and page order disagree, and that has to be
constructed deliberately.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from docling_core.types.doc.document import (
    BoundingBox,
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc2md.config import PipelineConfig  # noqa: E402
from doc2md.converter import DocumentPipeline  # noqa: E402

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def _pipeline() -> DocumentPipeline:
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline.config = PipelineConfig()
    return pipeline


def _prov(page_no: int, top: int = 0) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no, bbox=BoundingBox(l=0, t=top, r=10, b=top + 10), charspan=(0, 0)
    )


def _add_picture(doc, colour, page_no, *, extra_pages=()):
    img = PILImage.new("RGB", (4, 4), colour)
    item = doc.add_picture(image=ImageRef.from_pil(img, dpi=72), prov=_prov(page_no))
    for page in extra_pages:
        item.prov.append(_prov(page))
    return item


def _refs(markdown: str) -> list[str]:
    return [Path(t).name for t in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", markdown)]


@pytest.fixture()
def misordered_doc():
    """Two pages whose tree order and page order disagree.

    Tree order is red(p2), green(p1), blue(p2); reading order is
    green, red, blue.
    """
    doc = DoclingDocument(name="misordered")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    doc.add_page(page_no=2, size=Size(width=100, height=100))
    doc.add_text(label="text", text="page one text", prov=_prov(1))
    _add_picture(doc, RED, 2)
    _add_picture(doc, GREEN, 1)
    _add_picture(doc, BLUE, 2)
    doc.add_text(label="text", text="page two text", prov=_prov(2))
    return doc


# --- the pairing --------------------------------------------------------------


def test_every_reference_names_the_file_holding_its_own_pixels(misordered_doc, tmp_path):
    """The real regression guard: compare pixels, not filenames.

    A permutation preserves the *set* of names on both sides, so comparing name
    sets passes before and after the fix and proves nothing.
    """
    pipeline = _pipeline()
    markdown, image_names = pipeline._export_markdown_page_ordered(misordered_doc)
    pipeline._write_page_ordered_images(misordered_doc, tmp_path, image_names)

    # Reading order: green (page 1), then red and blue (page 2).
    expected = [GREEN, RED, BLUE]
    for filename, colour in zip(_refs(markdown), expected, strict=True):
        written = PILImage.open(tmp_path / filename).convert("RGB")
        assert written.getpixel((0, 0)) == colour, (
            f"{filename} is referenced where {colour} belongs but holds "
            f"{written.getpixel((0, 0))}"
        )


def test_an_unsaveable_picture_does_not_shift_the_others(misordered_doc, tmp_path, monkeypatch):
    """The conditional-increment drift.

    The file loop used to advance its counter only on a successful save while the
    markdown loop advanced unconditionally, so one unreadable picture re-pointed
    every subsequent reference.
    """
    pipeline = _pipeline()
    markdown, image_names = pipeline._export_markdown_page_ordered(misordered_doc)

    from docling_core.types.doc.document import PictureItem

    original = PictureItem.get_image

    def _fail_on_green(self, doc=None, **kwargs):
        img = original(self, doc, **kwargs)
        if img is not None and img.convert("RGB").getpixel((0, 0)) == GREEN:
            raise OSError("cannot identify image file")
        return img

    monkeypatch.setattr(PictureItem, "get_image", _fail_on_green)
    pipeline._write_page_ordered_images(misordered_doc, tmp_path, image_names)

    refs = _refs(markdown)
    # The green one is simply missing — a dangling reference, which the platform
    # treats as "images unusable". The other two keep their own bytes.
    assert not (tmp_path / refs[0]).exists()
    for filename, colour in ((refs[1], RED), (refs[2], BLUE)):
        assert PILImage.open(tmp_path / filename).convert("RGB").getpixel((0, 0)) == colour


def test_a_failed_save_leaves_no_truncated_file(misordered_doc, tmp_path, monkeypatch):
    """Callers upload the directory's contents, not the markdown's references."""
    pipeline = _pipeline()
    _markdown, image_names = pipeline._export_markdown_page_ordered(misordered_doc)

    def _explode(self, *args, **kwargs):
        Path(args[0]).write_bytes(b"\x89PNG truncated")
        raise OSError("disk full")

    monkeypatch.setattr(PILImage.Image, "save", _explode)
    pipeline._write_page_ordered_images(misordered_doc, tmp_path, image_names)

    assert list(tmp_path.iterdir()) == []


def test_references_never_point_outside_the_written_files(misordered_doc, tmp_path):
    """Dangling references are the one asymmetry that is always wrong."""
    pipeline = _pipeline()
    markdown, image_names = pipeline._export_markdown_page_ordered(misordered_doc)
    pipeline._write_page_ordered_images(misordered_doc, tmp_path, image_names)

    on_disk = {p.name for p in tmp_path.iterdir()}
    assert set(_refs(markdown)) <= on_disk


def test_a_picture_spanning_two_pages_is_written_once(tmp_path):
    """docling yields an item from *every* page its provenance covers.

    Naming it twice would produce two files for one picture and leave one of them
    orphaned; naming it once keeps refs a subset of the files.
    """
    doc = DoclingDocument(name="spanning")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    doc.add_page(page_no=2, size=Size(width=100, height=100))
    _add_picture(doc, RED, 1, extra_pages=(2,))
    _add_picture(doc, BLUE, 2)

    pipeline = _pipeline()
    markdown, image_names = pipeline._export_markdown_page_ordered(doc)
    pipeline._write_page_ordered_images(doc, tmp_path, image_names)

    on_disk = sorted(p.name for p in tmp_path.iterdir())
    assert len(on_disk) == 2, f"one picture per file expected, got {on_disk}"
    assert set(_refs(markdown)) <= set(on_disk)
    # The spanning picture is referenced on both pages it appears on…
    assert _refs(markdown).count(_refs(markdown)[0]) == 2
    # …and still holds its own pixels.
    assert PILImage.open(tmp_path / _refs(markdown)[0]).convert("RGB").getpixel((0, 0)) == RED


# --- tables -------------------------------------------------------------------


def test_tables_survive_the_page_ordered_fallback():
    """TableItem has no ``text`` attribute, so the old getattr default dropped
    every table in every document that tripped this branch."""
    doc = DoclingDocument(name="tabular")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    doc.add_page(page_no=2, size=Size(width=100, height=100))
    doc.add_text(label="text", text="before", prov=_prov(1))
    cells = [
        TableCell(
            text=t,
            row_span=1,
            col_span=1,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate((("Position", "Betrag"), ("Gutachten", "1.250,00")))
        for c, t in enumerate(row)
    ]
    doc.add_table(
        data=TableData(num_rows=2, num_cols=2, table_cells=cells), prov=_prov(1, top=20)
    )
    doc.add_text(label="text", text="after", prov=_prov(2))

    markdown, _names = _pipeline()._export_markdown_page_ordered(doc)

    assert "Position" in markdown
    assert "Gutachten" in markdown
    assert "1.250,00" in markdown
    # Rendered as a grid, not flattened prose.
    assert "|" in markdown


# --- degenerate documents -----------------------------------------------------


def test_a_document_with_no_pages_produces_no_silent_empty_output():
    """``sorted(doc.pages)`` is empty, so the fallback emits nothing at all.

    ``iterate_items(page_no=…)`` does not validate the page number, so nothing
    raises — the caller must not choose this branch in that state.
    """
    doc = DoclingDocument(name="pageless")
    doc.add_text(label="text", text="content with no page", prov=_prov(1))

    markdown, names = _pipeline()._export_markdown_page_ordered(doc)

    assert markdown == ""
    assert names == {}
    # …which is why the converter falls back to the tree export instead.
    assert "content with no page" in doc.export_to_markdown()
