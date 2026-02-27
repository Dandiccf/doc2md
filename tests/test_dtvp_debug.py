"""Diagnostic and regression test for the DTVP PDF page-order bug.

Docling's layout model can group items from different pages into the same
list node, causing the markdown serializer to output pages out of order.
The ``_has_page_order_violation`` / ``_export_markdown_page_ordered`` fallback
in ``converter.py`` detects and corrects this.

Usage:
    python tests/test_dtvp_debug.py          # quick manual check
    pytest  tests/test_dtvp_debug.py -v      # automated assertions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PDF_PATH = Path(__file__).resolve().parent / "2026-02-24_2455388_dtvp_md3.pdf"
OUTPUT_DIR = Path(__file__).resolve().parent / "test_output" / "dtvp_debug"

_skip = pytest.mark.skipif(not PDF_PATH.exists(), reason="DTVP test PDF not present")


@_skip
class TestDtvpPageOrder:
    """Regression tests for the cross-page ordering fix."""

    def test_page_break_placement(self, tmp_path):
        """Page break must appear AFTER all page-1 content, not after line 1."""
        from doc2md import PipelineConfig, convert

        config = PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
            page_break_placeholder="<!-- page-break -->",
        )
        result = convert(str(PDF_PATH), output_dir=str(tmp_path / "out"), config=config)
        assert result.success
        md = result.markdown_path.read_text(encoding="utf-8")
        lines = md.splitlines()

        # Find the page-break line
        pb_indices = [i for i, l in enumerate(lines) if "<!-- page-break -->" in l]
        assert len(pb_indices) == 1, f"Expected 1 page break, found {len(pb_indices)}"
        pb_line = pb_indices[0]

        # Page 1 has ~51 elements → page break must NOT be at line 0 or 1
        assert pb_line > 10, (
            f"Page break at line {pb_line} — page 1 content is still misplaced"
        )

    def test_page1_content_before_break(self, tmp_path):
        """Key page-1 content must appear before the page break."""
        from doc2md import PipelineConfig, convert

        config = PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
            page_break_placeholder="<!-- page-break -->",
        )
        result = convert(str(PDF_PATH), output_dir=str(tmp_path / "out"), config=config)
        md = result.markdown_path.read_text(encoding="utf-8")

        pb_pos = md.index("<!-- page-break -->")
        before_break = md[:pb_pos]

        # These are all from page 1 and must be before the page break
        for expected in [
            "Robert Koch-Institut",
            "Nordufer 20",
            "13353",
            "Öffentliche Ausschreibung nach UVgO",
            "Angebotsfrist: 23.03.2026",
            "Schulungssoftwarepaket",
            "keine Losaufteilung",
        ]:
            assert expected in before_break, (
                f"Page-1 text '{expected}' missing before page break"
            )

    def test_page2_content_after_break(self, tmp_path):
        """Key page-2 content must appear after the page break."""
        from doc2md import PipelineConfig, convert

        config = PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
            page_break_placeholder="<!-- page-break -->",
        )
        result = convert(str(PDF_PATH), output_dir=str(tmp_path / "out"), config=config)
        md = result.markdown_path.read_text(encoding="utf-8")

        pb_pos = md.index("<!-- page-break -->")
        after_break = md[pb_pos:]

        for expected in [
            "Bestimmungen über die Ausführungsfrist",
            "Sicherheitsleistungen",
            "Zahlungsbedingungen",
            "Bieterfragen können bis zum 16.03.2026",
        ]:
            assert expected in after_break, (
                f"Page-2 text '{expected}' missing after page break"
            )

    def test_no_text_lost(self, tmp_path):
        """All content from both pages must be present in the output."""
        from doc2md import PipelineConfig, convert

        config = PipelineConfig(
            do_picture_description=False,
            do_picture_classification=False,
        )
        result = convert(str(PDF_PATH), output_dir=str(tmp_path / "out"), config=config)
        md = result.markdown_path.read_text(encoding="utf-8")

        # Spot-check content from page 1
        assert "Robert Koch-Institut, Beschaffung" in md
        assert "ZV23_ausschreibung@rki.de" in md
        assert "Learning Management Systems" in md

        # Spot-check content from page 2
        assert "Zuschlagserteilung" in md
        assert "gesamtschuldnerisch haftend" in md
        assert "16.03.2026" in md


# ── Manual runner ──────────────────────────────────────────────────────

def _run_manual():
    from doc2md import PipelineConfig, convert

    config = PipelineConfig(
        do_picture_description=False,
        do_picture_classification=False,
        page_break_placeholder="<!-- page-break -->",
    )
    out = OUTPUT_DIR / "fixed"
    result = convert(str(PDF_PATH), output_dir=str(out), config=config)
    md = result.markdown_path.read_text(encoding="utf-8")
    lines = md.splitlines()

    pb_line = next(
        (i for i, l in enumerate(lines) if "<!-- page-break -->" in l), None,
    )
    print(f"Success: {result.success}")
    print(f"Page break at line: {pb_line}")
    print(f"Lines before break: {pb_line}")
    print(f"Lines after break:  {len(lines) - pb_line - 1}")
    print(f"\n{'='*70}\n{md}")


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)
    _run_manual()
