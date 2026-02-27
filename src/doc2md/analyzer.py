"""Lightweight PDF pre-analysis using pypdfium2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from doc2md.utils import logger


@dataclass
class PdfAnalysis:
    """Result of lightweight PDF pre-analysis."""

    total_pages: int = 0
    total_chars: int = 0
    total_images: int = 0
    avg_chars_per_page: float = 0.0
    avg_image_coverage: float = 0.0
    is_scanned: bool = False
    has_images: bool = False
    has_tables: bool = False


def analyze_pdf(path: str | Path, sample_pages: int = 5) -> PdfAnalysis:
    """Analyze a PDF to determine if it needs OCR/deep-learning processing.

    Samples a subset of pages (first, last, and evenly-spaced middle pages)
    for speed.  A page is considered "scanned" when it has very few text
    characters but significant image coverage.  If >50% of sampled pages
    look scanned, the whole document is flagged as scanned.

    On any failure the safe fallback (``is_scanned=True``) is returned so
    that the caller routes to Docling.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.debug("pypdfium2 not available; assuming scanned PDF")
        return PdfAnalysis(is_scanned=True)

    try:
        doc = pdfium.PdfDocument(str(path))
        total_pages = len(doc)
        if total_pages == 0:
            return PdfAnalysis(is_scanned=True)

        # Pick pages to sample: first, last, and evenly-spaced middle pages
        if total_pages <= sample_pages:
            indices = list(range(total_pages))
        else:
            indices = [0, total_pages - 1]
            step = (total_pages - 1) / (sample_pages - 1)
            for i in range(1, sample_pages - 1):
                idx = int(round(step * i))
                if idx not in indices:
                    indices.append(idx)
            indices = sorted(set(indices))

        total_chars = 0
        total_images = 0
        total_coverage = 0.0
        scanned_count = 0
        pages_with_significant_images = 0
        pages_with_tables = 0

        for page_idx in indices:
            page = doc[page_idx]
            page_w, page_h = page.get_size()
            page_area = page_w * page_h

            # Count text characters
            textpage = page.get_textpage()
            chars = textpage.count_chars()
            total_chars += chars

            # Classify page objects: images, paths (potential table lines)
            img_coverage = 0.0
            page_images = 0
            h_lines = 0
            v_lines = 0
            for obj in page.get_objects():
                if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                    page_images += 1
                    try:
                        bounds = obj.get_bounds()
                        img_w = bounds[2] - bounds[0]
                        img_h = bounds[3] - bounds[1]
                        if page_area > 0:
                            img_coverage += (img_w * img_h) / page_area
                    except Exception:
                        pass
                elif obj.type == pdfium.raw.FPDF_PAGEOBJ_PATH:
                    try:
                        bounds = obj.get_bounds()
                        w = bounds[2] - bounds[0]
                        h = bounds[3] - bounds[1]
                        # Thin horizontal rule (wide and very short)
                        if h < 2 and w > 30:
                            h_lines += 1
                        # Thin vertical rule (tall and very narrow)
                        elif w < 2 and h > 10:
                            v_lines += 1
                    except Exception:
                        pass
            total_images += page_images
            total_coverage += img_coverage

            # A page has "significant" images when coverage exceeds 5%
            if img_coverage > 0.05:
                pages_with_significant_images += 1

            # Table heuristic: a grid needs at least 3 horizontal + 3 vertical rules
            if h_lines >= 3 and v_lines >= 3:
                pages_with_tables += 1

            # Scanned heuristic: very few chars + significant image coverage
            if chars < 20 and img_coverage > 0.3:
                scanned_count += 1

        avg_chars = total_chars / len(indices) if indices else 0.0
        avg_coverage = total_coverage / len(indices) if indices else 0.0
        is_scanned = scanned_count > len(indices) / 2
        has_images = pages_with_significant_images > 0
        has_tables = pages_with_tables > 0

        doc.close()

        return PdfAnalysis(
            total_pages=total_pages,
            total_chars=total_chars,
            total_images=total_images,
            avg_chars_per_page=avg_chars,
            avg_image_coverage=avg_coverage,
            is_scanned=is_scanned,
            has_images=has_images,
            has_tables=has_tables,
        )

    except Exception as exc:
        logger.warning("PDF analysis failed (%s); assuming scanned", exc)
        return PdfAnalysis(is_scanned=True)
